import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# --- 평가 루틴 (로그 변환 썼다면 원공간 MAE도 같이 리포트) ---
def evaluate(model, loader, log_target: bool = True):
    device = next(model.parameters()).device
    model.eval()
    crit = nn.L1Loss()  # MAE (로그 스페이스 기준)
    loss_sum = 0.0
    raw_mae_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb, bb in loader:         # << (X, Y, B) 받음
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()
            bb = bb.to(device, non_blocking=True).long()

            out = model(xb, bb)           # << 글로벌 모델: (x, b_idx)
            if out.dim() == 3 and out.size(-1) == 1:
                out = out.squeeze(-1)     # (B,H,1) -> (B,H)

            loss = crit(out, yb)          # 로그 스페이스 MAE
            bs = xb.size(0)
            loss_sum += loss.item() * bs
            n += bs

            if log_target:
                y_hat_real = torch.expm1(out)
                y_real     = torch.expm1(yb)
                raw_mae = torch.mean(torch.abs(y_hat_real - y_real)).item()
                raw_mae_sum += raw_mae * bs

    log_mae = loss_sum / max(1, n)
    if log_target:
        mae = raw_mae_sum / max(1, n)
        return log_mae, mae
    return log_mae, None

# --- 학습 루틴: DataLoader 기반 / 글로벌 모델 지원 ---
def _resolve_run_name(model, run_name=None):
    if run_name is not None:
        return run_name
    # base 모델 속성 후보들(구현마다 이름이 다를 수 있음)
    base_obj = None
    for attr in ("base", "base_model", "backbone", "model"):
        if hasattr(model, attr):
            base_obj = getattr(model, attr)
            break
    base_name = base_obj.__class__.__name__ if base_obj is not None else "UnknownBase"
    wrapper_name = model.__class__.__name__
    return f"{wrapper_name}-{base_name}"

def train_model(
    model,
    loader_tr, loader_va, loader_te,
    lr=1e-3, weight_decay=1e-4, num_epochs=30,
    early_stop_patience=30, log_target=True,
    ckpt_dir="results", run_name=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- 저장 이름 ---
    run_name = _resolve_run_name(model, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"global_{run_name}-BDG2.pt")
    # print(f"{ckpt_path}")  # 실제 저장 경로 확인용

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val = float("inf"); waited = 0

    for epoch in range(1, num_epochs+1):
        model.train(); train_sum = 0.0; ntr = 0
        for xb, yb, bb in loader_tr:
            xb, yb, bb = xb.to(device).float(), yb.to(device).float(), bb.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            out = model(xb, bb)
            if out.dim()==3 and out.size(-1)==1: out = out.squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item()*xb.size(0); ntr += xb.size(0)
        train_log = train_sum/max(1,ntr)

        # valid
        model.eval(); val_sum = 0.0; nv = 0
        with torch.no_grad():
            for xb, yb, bb in loader_va:
                xb, yb, bb = xb.to(device).float(), yb.to(device).float(), bb.to(device).long()
                out = model(xb, bb)
                if out.dim()==3 and out.size(-1)==1: out = out.squeeze(-1)
                val_sum += criterion(out, yb).item()*xb.size(0); nv += xb.size(0)
        val_log = val_sum/max(1,nv)
        scheduler.step(val_log)

        if val_log < best_val - 1e-6:
            best_val = val_log; waited = 0
            torch.save({"model_state": model.state_dict()}, ckpt_path)
            print(f"[BEST SAVED] {ckpt_path}")
        else:
            waited += 1
            if waited >= early_stop_patience:
                print("Early stopping."); break

        print(f"Epoch {epoch:03d} | train(logMAE)={train_log:.4f} | val(logMAE)={val_log:.4f} (best {best_val:.4f})")

    # ---- test ----
    # 베스트 로드 후 테스트
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
    te_log, te_mae = evaluate(model, loader_te, log_target=log_target)
    if log_target and te_mae is not None:
        print(f"Test: logMAE={te_log:.4f} | MAE={te_mae:.4f}")
    else:
        print(f"Test: logMAE={te_log:.4f}")
