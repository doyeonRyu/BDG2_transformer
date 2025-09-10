import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator

def plot_data_by_month(filepath, building_name, year, X_train, X_val):
    """
    Function: plot_data_by_month
        1. csv 파일 불러오기
        2. train/valid/test 경계 인덱스 계산
        3. 시각화
            3.1 y축
            - 왼쪽 y축: electricity (파랑)
            - 오른쪽 y축: Mean Temp (°C) (빨강), Total Rain (mm) (노랑)
            3.2 경계선 표시 (---)
            3.3 x축: month만 표시
            3.4 나머지 시각화
        4. 저장
    Parameters:
        filepath (str): CSV 파일 경로
        building_name (str): 건물 이름
        year (int): 연도
        X_train (np.ndarray): 학습용 입력 데이터 (경계 인덱스 계산용)
        X_val (np.ndarray): 검증용 입력 데이터 (경계 인덱스 계산용)
    Return values:
        None
    """
    # 1. csv 데이터 불러오기
    df = pd.read_csv(filepath)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df = df.sort_values('Date/Time').reset_index(drop=True)

    # 2. train/valid/test 경계 인덱스 계산
    train_end = X_train.shape[0]
    valid_end = train_end + X_val.shape[0]

    # 3. 시각화
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 3.1 왼쪽 y축: electricity (파랑)
    elec_line, = ax1.plot(df['Date/Time'], df['electricity'],
                          color='tab:blue', label='Electricity', linewidth=0.8)
    ax1.set_ylabel('Electricity', color='tab:blue')

    # 오른쪽 y축: Temp (빨강) / Rain (노랑)
    ax2 = ax1.twinx()
    temp_line, = ax2.plot(df['Date/Time'], df['Mean Temp (°C)'],
                          color='tab:red', label='Mean Temp (°C)', linewidth=0.8)
    rain_line, = ax2.plot(df['Date/Time'], df['Total Rain (mm)'],
                          color='tab:orange', label='Total Rain (mm)', linewidth=0.8)
    ax2.set_ylabel('Temperature / Rain', color='tab:red')

    # 3.2 경계선 표시 (---)
    split_line1 = ax1.axvline(df['Date/Time'].iloc[train_end],
                              color='gray', linestyle='--', label='Train/Valid Split', alpha=0.8)
    split_line2 = ax1.axvline(df['Date/Time'].iloc[valid_end],
                              color='black', linestyle='--', label='Valid/Test Split', alpha=0.8)

    # 3.3 x축 라벨은 month만 표시
    ax1.xaxis.set_major_locator(MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(DateFormatter('%m'))
    ax1.set_xlabel('Month')


    # 범례 생성: 모든 라인을 합쳐서 표시
    lines = [elec_line, temp_line, rain_line, split_line1, split_line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # 제목 표시
    title_str = f"{building_name} Electricity / Temperature / Rain Data in {year}"
    plt.title(title_str)
    ax1.grid(True, which='both', axis='y', alpha=0.8)     # Electricity 축만 grid 표시
    plt.tight_layout()
    plt.show()

    # 저장
    fig.savefig(f"plots/png/{building_name}_{year}_data_plot.png", dpi=300)
    print(f"[저장 완료] plots/png/{building_name}_{year}_data_plot.png")

import numpy as np
import matplotlib.pyplot as plt

def _y_to_original(arr, y_scaler, clip01=True):
    """
    y 배열이 어느 도메인(스케일[0..1], 로그, 원단위)에 있는지 추정 후
    '딱 한 번'만 올바른 역변환을 적용하여 원단위로 반환.
    비정상적으로 큰 값은 자동 교정.
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)

    # 0) 기대 가능한 상한(원단위) 계산: train에서 본 로그 최대치의 expm1
    max_orig_expected = float(np.expm1(y_scaler.data_max_[0]))  # 예: 0~27 근방
    hard_upper = max_orig_expected * 1.5  # 여유 버퍼

    # 1) 스케일/로그/원단위 추정
    amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
    is_scaled_like = (-0.05 <= amin) and (amax <= 1.05)
    is_log_like    = (0.0 <= amin) and (amax <= 6.5)   # log1p(27)=3.33, 넉넉히 6.5

    # 2) 도메인별 역변환
    if is_scaled_like:
        a2 = np.clip(a, 0.0, 1.0) if clip01 else a
        logv = y_scaler.inverse_transform(a2).ravel()
        orig = np.expm1(logv)
    elif is_log_like:
        orig = np.expm1(a.ravel())
    else:
        # 이미 원단위라고 간주
        orig = a.ravel()

    # 3) 비정상치 자동 교정: 상한을 심하게 초과하면 ‘로그였다고’ 보고 expm1만 적용
    if np.nanmax(orig) > hard_upper:
        # 두 번째 전략: (스케일처럼) clip→inverse→expm1
        try_scaled = np.expm1(y_scaler.inverse_transform(np.clip(a, 0, 1))).ravel()
        if np.nanmax(try_scaled) <= hard_upper:
            return try_scaled
        # 세 번째 전략: (로그처럼) expm1만
        try_logonly = np.expm1(a.ravel())
        if np.nanmax(try_logonly) <= hard_upper:
            return try_logonly
        # 그래도 높으면 상한으로 클립
        return np.clip(orig, 0, hard_upper)
    return orig

import os
import numpy as np
import matplotlib.pyplot as plt

def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def _find_col_idx(feature_names, target_names=("elec", "electricity")):
    if feature_names is None: 
        return None
    low = [str(c).lower() for c in feature_names]
    for tgt in target_names:
        if tgt.lower() in low:
            return low.index(tgt.lower())
    # 부분일치(예: 'building_elec', 'meter_electricity')
    for i, c in enumerate(low):
        if any(t in c for t in target_names):
            return i
    return None

def plot_forecast(
    model_name,                     # ← 문자열 이름(파일명/타이틀에 사용)
    preds, trues,                  # (N,H) 원공간
    input_window, output_window,
    sample_index=0,
    X_test=None,                   # (N,L,F) 스케일된 입력 (옵션)
    x_scaler=None,                 # MinMax/Standard 스케일러 (옵션)
    feature_names=None,            # 입력 피처명 리스트/Index (옵션)
    elec_col_name=("elec","electricity"),  # 전력 컬럼 탐색 키
    elec_was_log_scaled=False,     # 입력 ‘전력’이 log1p였다면 True → expm1
    save_dir="plots/png"
):
    # 1) 타깃(예측/정답) – 이미 원공간
    p = _to_np(preds)[sample_index].ravel()
    t = _to_np(trues)[sample_index].ravel()

    # 2) 입력 전력 복원(있을 때만)
    input_seq = None
    if X_test is not None and x_scaler is not None and feature_names is not None:
        x_sample_scaled = _to_np(X_test)[sample_index]        # (L,F)
        # inverse_transform은 피처별 독립이므로 전체를 되돌린 뒤 전력 컬럼만 추출
        try:
            x_sample_orig = x_scaler.inverse_transform(x_sample_scaled)
        except Exception:
            # 일부 스케일러는 (1,L,F)→(L,F)만 허용. 차원 맞춤.
            x_sample_orig = x_scaler.inverse_transform(x_sample_scaled.reshape(-1, x_sample_scaled.shape[-1]))
        e_idx = _find_col_idx(feature_names, elec_col_name)
        if e_idx is not None:
            input_seq = x_sample_orig[:, e_idx]
            if elec_was_log_scaled:
                input_seq = np.expm1(input_seq)

    # 3) 플로팅
    L_in, L_out = input_window, output_window
    t_in  = np.arange(L_in)
    t_out = np.arange(L_in, L_in + L_out)

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    if input_seq is not None:
        plt.plot(t_in,  input_seq, label="Input Electricity", linewidth=1.2)
    plt.plot(t_out, t, label="Target (True)", linewidth=1.2)
    plt.plot(t_out, p, label="Prediction", linewidth=1.6, linestyle="--")
    if input_seq is not None:
        plt.axvline(L_in - 0.5, color="gray", linestyle="--", alpha=0.8, label="Input/Output Split")
    plt.title(f"{model_name} — Sample {sample_index} (Input {L_in} → Output {L_out})")
    plt.xlabel("Time step (relative)")
    plt.ylabel("Electricity (original unit)")
    plt.legend(loc="upper left"); plt.grid(True, axis="y", alpha=0.7); plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_sample_{sample_index}_forecast.png")
    plt.savefig(save_path, dpi=300); plt.show()
    print(f"[{model_name} 모델 결과 저장 완료] {save_path}")
