import numpy as np
import torch

# ----------------- 유틸 -----------------
def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _inverse_minmax(z, vmin, vmax):
    '''
    MinMax 역정규화: z*(vmax-vmin) + vmin
    z: (..., H)
    vmin, vmax: 스칼라 또는 (..., H) 브로드캐스터블
    '''
    return z * (vmax - vmin) + vmin

def _inverse_minmax_by_bidx(z, bidx, vmin_by_bidx, vmax_by_bidx):
    '''
    bidx별 MinMax 역정규화(배치 내에서 bidx가 섞여 있어도 처리).
    z: (B, H)
    bidx: (B,)
    vmin_by_bidx, vmax_by_bidx: dict[int] -> array-like(H,) or scalar
    '''
    z = z.clone() if isinstance(z, torch.Tensor) else z.copy()
    # 고유 bidx마다 나눠서 적용
    uniq = torch.unique(bidx).tolist() if isinstance(bidx, torch.Tensor) else np.unique(bidx).tolist()
    for bi in uniq:
        mask = (bidx == bi)
        vmin = vmin_by_bidx[ int(bi) ]
        vmax = vmax_by_bidx[ int(bi) ]
        if isinstance(z, torch.Tensor):
            vmin_t = torch.as_tensor(vmin, dtype=z.dtype, device=z.device)
            vmax_t = torch.as_tensor(vmax, dtype=z.dtype, device=z.device)
            z[mask] = z[mask] * (vmax_t - vmin_t) + vmin_t
        else:
            vmin_n = np.asarray(vmin)
            vmax_n = np.asarray(vmax)
            z[mask] = z[mask] * (vmax_n - vmin_n) + vmin_n
    return z

def compute_metrics(y_hat, y, eps=1e-6, mape_thresh=None):
    '''
    y_hat, y: 원공간 numpy (N,H) 또는 (N,)
    mape_thresh: |y| < mape_thresh 샘플은 MAPE/SMAPE/WAPE 계산에서 제외
    '''
    y_hat = _to_np(y_hat); y = _to_np(y)
    err = y_hat - y
    abs_err = np.abs(err)

    mask = np.isfinite(y_hat) & np.isfinite(y)
    if mape_thresh is not None:
        mask = mask & (np.abs(y) >= mape_thresh)

    if not np.any(mask):
        return {"MAE": np.nan, "RMSE": np.nan, "WAPE": np.nan, "sMAPE": np.nan, "MAPE": np.nan, "coverage": 0.0}

    mae  = float(np.mean(abs_err[mask]))
    rmse = float(np.sqrt(np.mean((err[mask] ** 2))))
    wape = float(100.0 * np.sum(abs_err[mask]) / max(eps, np.sum(np.abs(y[mask]))))
    smape = float(100.0 * np.mean(
        abs_err[mask] / np.maximum(eps, (np.abs(y_hat[mask]) + np.abs(y[mask])) / 2.0)
    ))
    mape = float(100.0 * np.mean(
        abs_err[mask] / np.maximum(eps, np.abs(y[mask]))
    ))
    cov = float(np.mean(mask))
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "sMAPE": smape, "MAPE": mape, "coverage": cov}

def compute_metrics_by_building(preds, trues, bids, eps=1e-6, mape_thresh=None):
    out = {}
    bids = _to_np(bids).astype(int)
    for b in np.unique(bids):
        idx = bids == b
        out[int(b)] = compute_metrics(preds[idx], trues[idx], eps=eps, mape_thresh=mape_thresh)
    return out

# ----------------- 테스트 -----------------
def test_model(
    model,
    loader_te,
    # 학습 시 로그/정규화 여부
    log_target: bool = True,                 # 학습에서 log1p 적용했다면 True
    normalize_target: bool = True,           # 학습에서 MinMax 정규화했다면 True
    # 역정규화 파라미터(둘 중 하나를 채택)
    # 1) 전건물 공통 Min/Max
    target_min=None,                         # float or array-like(H,)
    target_max=None,                         # float or array-like(H,)
    # 2) 건물별(b_idx) Min/Max
    target_min_by_bidx: dict | None = None,  # {int b_idx: float/array(H,)}
    target_max_by_bidx: dict | None = None,  # {int b_idx: float/array(H,)}
    # 지표/필터 옵션
    return_building: bool = False,
    return_metrics: bool = False,
    mape_thresh: float = None,
    eps: float = 1e-6,
    per_building_metrics: bool = False,
    # 특정 건물만 평가
    only_building=None,
    bid2idx: dict = None,
    only_bidx=None
):
    '''
    반환 형태:
      - return_metrics=False:
          return_building=False → (preds, trues)
          return_building=True  → (preds, trues, bids)
      - return_metrics=True:
          return_building=False → (preds, trues, metrics)
          return_building=True  → (preds, trues, bids, metrics[, metrics_by_building])
    '''
    device = next(model.parameters()).device
    model.eval()

    # ---- 허용 빌딩 인덱스 집합 계산 ----
    allow_bidx = None
    if only_building is not None:
        assert bid2idx is not None, "only_building을 쓰려면 bid2idx가 필요합니다."
        if isinstance(only_building, (list, tuple, set)):
            allow_bidx = {int(bid2idx[b]) for b in only_building if b in bid2idx}
        else:
            assert only_building in bid2idx, f"'{only_building}'가 bid2idx에 없습니다."
            allow_bidx = {int(bid2idx[only_building])}
    elif only_bidx is not None:
        if isinstance(only_bidx, (list, tuple, set)):
            allow_bidx = {int(x) for x in only_bidx}
        else:
            allow_bidx = {int(only_bidx)}

    preds_list, trues_list, bids_list = [], [], []

    with torch.no_grad():
        for xb, yb, bb in loader_te:  # xb:(B,L,F), yb:(B,H)/(B,H,1), bb:(B,)
            # --- 빌딩 필터링(테스트 전용) ---
            if allow_bidx is not None:
                mask = torch.zeros_like(bb, dtype=torch.bool)
                for bi in allow_bidx:
                    mask |= (bb == bi)
                if not mask.any():
                    continue
                xb, yb, bb = xb[mask], yb[mask], bb[mask]

            xb = xb.to(device).float()
            yb = yb.to(device).float()
            bb = bb.to(device).long()

            out = model(xb, bb)  # (B,H) or (B,H,1)
            if out.dim() == 3 and out.size(-1) == 1:
                out = out.squeeze(-1)
            if yb.dim() == 3 and yb.size(-1) == 1:
                yb = yb.squeeze(-1)

            # ===== 원공간 복원 =====
            # 학습 순서가 [log1p → MinMax]였으므로,
            # 복원은 [MinMax 역정규화 → expm1] 순서로 진행.
            def _undo_all(z, bidx):
                # 1) MinMax 역정규화
                if normalize_target:
                    if (target_min_by_bidx is not None) and (target_max_by_bidx is not None):
                        z = _inverse_minmax_by_bidx(z, bidx, target_min_by_bidx, target_max_by_bidx)
                    elif (target_min is not None) and (target_max is not None):
                        if isinstance(z, torch.Tensor):
                            vmin = torch.as_tensor(target_min, dtype=z.dtype, device=z.device)
                            vmax = torch.as_tensor(target_max, dtype=z.dtype, device=z.device)
                            z = z * (vmax - vmin) + vmin
                        else:
                            z = _inverse_minmax(z, np.asarray(target_min), np.asarray(target_max))
                    else:
                        # normalize_target=True인데 Min/Max 정보가 없으면 경고적 처리
                        raise ValueError("normalize_target=True이면 target_min/target_max 또는 *_by_bidx를 제공해야 합니다.")
                # 2) 로그 복원
                if log_target:
                    if isinstance(z, torch.Tensor):
                        z = torch.expm1(z)
                    else:
                        z = np.expm1(z)
                return z

            out_rec = _undo_all(out, bb)
            yb_rec  = _undo_all(yb,  bb)

            # 텐서를 넘파이로 모으기
            preds_list.append(out_rec.detach().cpu().numpy() if isinstance(out_rec, torch.Tensor) else out_rec)
            trues_list.append(yb_rec.detach().cpu().numpy() if isinstance(yb_rec, torch.Tensor) else yb_rec)
            if return_building:
                bids_list.append(bb.detach().cpu().numpy())

    if len(preds_list) == 0:
        raise RuntimeError("테스트 세트에서 조건에 맞는 샘플이 없습니다. (only_building/only_bidx 확인)")

    preds = np.concatenate(preds_list, axis=0)  # (N,H)
    trues = np.concatenate(trues_list, axis=0)  # (N,H)

    if not return_metrics:
        if return_building:
            bids = np.concatenate(bids_list, axis=0)  # (N,)
            return preds, trues, bids
        return preds, trues

    # 지표 계산(원공간)
    metrics = compute_metrics(preds, trues, eps=eps, mape_thresh=mape_thresh)

    if return_building:
        bids = np.concatenate(bids_list, axis=0)
        if per_building_metrics:
            metrics_by = compute_metrics_by_building(preds, trues, bids, eps=eps, mape_thresh=mape_thresh)
            return preds, trues, bids, metrics, metrics_by
        return preds, trues, bids, metrics

    return preds, trues, metrics
