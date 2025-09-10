import numpy as np
import pandas as pd
import torch 
import holidays
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess(elec, weather, input_window, output_window):
    """
    Fuction: load_and_preprocess
        1. target, weather 데이터 불러오기 
        2. 데이터 병합
        3. timestamp를 통한 데이터 정렬
        2. 결측치/이상치 확인
        3. Feature Engineering
        4. 타겟 로그 변환
        5. Train / Valid / Test 분할
        6. MinMax 정규화
        7. 슬라이딩 윈도우 생성
        8. Tensor로 변환
    Parameters:
        target (str): target 데이터 CSV 파일 경로
        weather (str): 날씨 데이터 CSV 파일 경로
        input_window (int): 입력 윈도우 크기
        output_window (int): 출력 윈도우 크기
    Return values:
        X_train_tensor (torch.Tensor): 학습용 입력 데이터
        y_train_tensor (torch.Tensor): 학습용 출력 데이터
        X_val_tensor (torch.Tensor): 검증용 입력 데이터
        y_val_tensor (torch.Tensor): 검증용 출력 데이터
        X_test_tensor (torch.Tensor): 테스트용 입력 데이터
        y_test_tensor (torch.Tensor): 테스트용 출력 데이터
        x_scaler (MinMaxScaler): 입력 데이터 스케일러 (추후 변환용)
        y_scaler (MinMaxScaler): 출력 데이터 스케일러 (추후 변환용)
    """
    # 1. elec, weather 데이터 불러오기

    # electricity 데이터 불러오기
    elec = pd.read_csv('data/electricity.csv', parse_dates=["timestamp"])
    # Panther_office_Hannah 열만 선택
    elec = elec[["timestamp", "Panther_office_Hannah"]]
    # Panther__office_Hannah 열 이름 변경
    elec.rename(columns={"Panther_office_Hannah": "electricity"}, inplace=True)

    # weather 데이터 불러오기
    weather = pd.read_csv('data/weather.txt', sep=',', parse_dates=["timestamp"])
    # Panther 사이트의 데이터만 선택
    weather = weather[weather["site_id"] == "Panther"]
    weather.drop(columns=["site_id"], inplace=True)

    # 2. 데이터 병합

    #   elec, weather 데이터 병합
    data = pd.merge(elec, weather, on="timestamp", how="left")

    # timestamp: 2016년 6월 1일부터 끝까지로 제한
    data = data[data["timestamp"] >= "2016-06-01"]

    # 3. timestamp를 통한 데이터 정렬

    #   중복 제거 및 정렬 
    data = data.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")

    #   항상 1시간 간격으로 시간축 맞춤
    full_index = pd.date_range(start=data["timestamp"].min(),
                            end=data["timestamp"].max(),
                            freq="h")

    #   timestamp를 인덱스로 설정
    data = data.set_index("timestamp", drop=False).reindex(full_index)
    data.index.name = "timestamp"

    # 4. 이상치 / 결측치 처리

    # IQR 이상치 - Nan 처리
    outliers = {}
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64]:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)] 
            # 이상치를 NaN으로 처리
            data.loc[(data[col] < lower_bound) | (data[col] > upper_bound), col] = np.nan

    # print("이상치 개수:")
    # for col, outlier_data in outliers.items():
    #     print(f"{col}   : {len(outlier_data)}")

    # 결측치 확인 (이상치 + 실제 Nan 값)
    # missing_data = data.isnull().sum()
    # print("결측치 개수:")
    # print(missing_data[missing_data > 0])

    # 결측치 처리
    # 1) 선형 보간 -> 앞/뒤 값으로 채우기
    #   해당하는 열: electricity, airTemperature, dewTemperature, seaLvlPressure, windSpeed
    data[["electricity", "airTemperature", "dewTemperature", "seaLvlPressure", "windSpeed"]] = (
        data[["electricity", "airTemperature", "dewTemperature", "seaLvlPressure", "windSpeed"]]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )

    # 2) 앞/뒤 값으로 채우기
    #   해당하는 열: cloudCoverage, windDirection, precipDepth1HR, precipDepth6HR
    data[["cloudCoverage", "windDirection", "precipDepth1HR", "precipDepth6HR"]] = (
        data[["cloudCoverage", "windDirection", "precipDepth1HR", "precipDepth6HR"]]
        .ffill()
        .bfill()
    )
    
    # # 결측치 재확인 - 없음
    # missing_data = data.isnull().sum()
    # print("결측치 개수:")
    # print(missing_data[missing_data > 0])

    # 5. Feature Engineering
    # 5.1 시간 관련 Feature
    data["Date/Time"] = data["timestamp"].dt.date

    # 요일 인코딩 (0: 월요일, 6: 일요일)
    data["Date/Time"] = pd.to_datetime(data.index)
    data["dayofweek"] = data["Date/Time"].dt.dayofweek

    data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)

    # 주말/공휴일 인코딩
    # 주말
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

    # 공휴일 인코딩 (해당 지역의 정확한 위치가 없어 공휴일 제외)

    # 시간대 인코딩
    data["hour"] = data["timestamp"].dt.hour
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    # 계절 인코딩
    data["month"] = data["timestamp"].dt.month
    data["season"] = ((data["month"] - 1) // 3) % 4
    data["season_sin"] = np.sin(2 * np.pi * data["season"] / 4)
    data["season_cos"] = np.cos(2 * np.pi * data["season"] / 4)

    # 5.2 날씨 인코딩
    # cloudCoverage는 0~6 사이의 값이므로 0-1 사이 값으로 정규화
    data["cloudCoverage_norm"] = data["cloudCoverage"] / 6.0
    data.drop(columns=["cloudCoverage"], inplace=True)

    # windDirection
    data["wind_sin"] = np.sin(data["windDirection"] * np.pi / 180)
    data["wind_cos"] = np.cos(data["windDirection"] * np.pi / 180)
    data.drop(columns=["windDirection"], inplace=True)

    # 6. 타겟 로그 변환

    # electricity 로그 변환
    data["electricity"] = np.log1p(data["electricity"])

    # 7. Train / Valid / Test 분할
    # 각 데이터 시작과 끝: 월요일 - 일요일
    # 각 데이터 사이의 gap: 1주일

    data = data.sort_index()

    def next_monday(ts):
        ts = pd.Timestamp(ts)
        return ts if ts.weekday() == 0 else (ts + pd.offsets.Week(weekday=0))

    def prev_sunday(ts):
        ts = pd.Timestamp(ts)
        return ts if ts.weekday() == 6 else (ts - pd.offsets.Week(weekday=6))

    def week_end(ts):
        ts = pd.Timestamp(ts).normalize()
        return ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    # 전체 범위(안전 클립)
    global_start = data.index.min()
    global_end   = data.index.max()

    # 원하는 달 범위(원시 경계)
    train_raw_start = pd.Timestamp('2016-06-01')
    train_raw_end   = pd.Timestamp('2017-05-31')
    valid_raw_start = pd.Timestamp('2017-06-01')
    valid_raw_end   = pd.Timestamp('2017-08-31')
    test_raw_start  = pd.Timestamp('2017-09-01')
    test_raw_end    = pd.Timestamp('2017-12-31')

    # 월-일 경계로 정렬 (월요일 시작, 일요일 종료)
    train_start = next_monday(max(train_raw_start, global_start))
    train_end   = prev_sunday(min(train_raw_end, global_end))

    valid_start_nominal = next_monday(max(valid_raw_start, global_start))
    valid_end   = prev_sunday(min(valid_raw_end, global_end))

    test_start_nominal  = next_monday(max(test_raw_start, global_start))
    test_end    = prev_sunday(min(test_raw_end, global_end))

    # 1주 gap 적용: 앞 구간의 종료 일요일 다음 주(월요일)부터 +1주 비우고 시작
    gap_days = 7

    # train → valid 사이 gap
    # train_end 는 일요일. 그 다음주 월요일 = train_end + 1day
    valid_start_min = next_monday(train_end + pd.Timedelta(days=1)) + pd.Timedelta(days=gap_days)
    valid_start = max(valid_start_nominal, valid_start_min)

    # valid → test 사이 gap
    test_start_min = next_monday(prev_sunday(valid_end) + pd.Timedelta(days=1)) + pd.Timedelta(days=gap_days)
    test_start = max(test_start_nominal, test_start_min)

    # Train / Valid / Test 분할 
    train = data.loc[train_start : week_end(train_end)]
    valid = data.loc[valid_start : week_end(valid_end)]
    test  = data.loc[test_start  : week_end(test_end)]

    print("train:", train_start, "→", train_end, f"({len(train)} rows)")
    print("valid:", valid_start, "→", valid_end, f"({len(valid)} rows)")
    print("test :", test_start,  "→", test_end, f"({len(test)} rows)")

    # 필요 없는 열 제거
    def delect_cols(data):
        data = data.drop(columns=["dayofweek", "Date/Time", "hour", "timestamp", "month", "season"])
        return data

    train = delect_cols(train)
    valid = delect_cols(valid)
    test = delect_cols(test)

    # 8. x, y 분리하기
    def split_x_y(df):
        x = df
        y = df["electricity"].values.astype(np.float32) 
        y = y.reshape(-1, 1)  # y를 2D 배열로 변환 (모델 입력에 맞춤)
        return x, y

    x_train, y_train = split_x_y(train)
    x_val, y_val = split_x_y(valid)
    x_test, y_test = split_x_y(test)

    # 9. MinMax 정규화

    # MinMax 정규화: train에만 fit, valid/test는 transform
    cols_to_scale = ['electricity', 'airTemperature', 'dewTemperature', 'precipDepth1HR', 'precipDepth6HR', 'seaLvlPressure', 'windSpeed']

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train[cols_to_scale] = x_scaler.fit_transform(x_train[cols_to_scale]) # train 
    x_val[cols_to_scale]   = x_scaler.transform(x_val[cols_to_scale])
    x_test[cols_to_scale]  = x_scaler.transform(x_test[cols_to_scale])

    y_train = y_scaler.fit_transform(y_train) # train
    y_val   = y_scaler.transform(y_val)
    y_test  = y_scaler.transform(y_test)

    # 10. 슬라이딩 윈도우 생성
    def sliding_windows(x_scaled, y_scaled, input_window, output_window):
        """
        Function: sliding_windows
            - 슬라이딩 윈도우 생성 과정
        Parameters:
            - x_scaled (np.ndarray): 이미 정규화된 입력 데이터
            - y_scaled (np.ndarray): 이미 정규화된 출력 데이터
            - input_window (int): 입력 윈도우 크기
            - output_window (int): 출력 윈도우 크기
        Return values:
            - X_list (np.ndarray): 입력 윈도우 데이터
            - Y_list (np.ndarray): 출력 윈도우 데이터
        """
        T = x_scaled.shape[0] # 전체 시간 스텝 수 (1시간 단위 1년치 데이터라면 8760)
        X_list, Y_list = [], []

            # x_scaled, y_scaled 형식 변환 (numpy.ndarray)
        x_scaled = np.asarray(x_scaled)
        y_scaled = np.asarray(y_scaled)

        for i in range(0, T - input_window - output_window + 1):
            X_win = x_scaled[i : i + input_window, :] # (input_window, 입력 변수 개수)
            Y_win = y_scaled[i + input_window : i + input_window + output_window, :] # (output_window, 1)
            X_list.append(X_win)
            Y_list.append(Y_win)
        if len(X_list) == 0:
            return (np.empty((0, input_window, x_scaled.shape[1]), dtype=np.float32),
                    np.empty((0, output_window, 1), dtype=np.float32))
        return np.stack(X_list, 0).astype(np.float32), np.stack(Y_list, 0).astype(np.float32)
        # X_list: (N, input_window, 4)
        # Y_list: (N, output_window, 1)
        # N: 윈도우 개수 (for문 수 만큼)

    #   슬라이딩 윈도우 적용
    X_tr_np, y_tr_np = sliding_windows(x_train, y_train, input_window, output_window)
    X_va_np, y_va_np = sliding_windows(x_val,   y_val,   input_window, output_window)
    X_te_np, y_te_np = sliding_windows(x_test,  y_test,  input_window, output_window)

    # 6. Tensor로 변환 (dtype 변환: numpy -> torch.from_numpy)
    X_train_tensor = torch.from_numpy(X_tr_np).to("cuda")  # (N_tr, 168, 4)
    y_train_tensor = torch.from_numpy(y_tr_np).to("cuda")  # (N_tr, 24, 1)

    X_val_tensor   = torch.from_numpy(X_va_np).to("cuda")
    y_val_tensor   = torch.from_numpy(y_va_np).to("cuda")

    X_test_tensor  = torch.from_numpy(X_te_np).to("cuda")
    y_test_tensor  = torch.from_numpy(y_te_np).to("cuda")

    print("Train X:", X_train_tensor.shape, "y:", y_train_tensor.shape)
    print("Valid  X:", X_val_tensor.shape,   "y:", y_val_tensor.shape)
    print("Test  X:", X_test_tensor.shape,   "y:", y_test_tensor.shape)

    return (
        X_train_tensor, y_train_tensor,
        X_val_tensor, y_val_tensor,
        X_test_tensor, y_test_tensor,
        x_scaler, y_scaler
    )