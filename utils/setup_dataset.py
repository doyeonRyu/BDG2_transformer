import pandas as pd
# import argparse # 파일 실행 시 

def setup_dataset(building_name: str):
    """
    Function: setup_dataset
        1. 에너지, 날씨 데이터를 불러오기
        2. 날짜 열 형식 변환
        2. 연도별로 필터링
        6. 최종 열 선택
        7. 결과를 CSV로 저장
    Parameters:
        building_name (str): 건물 이름 (예: Panther_parking_Lorriane)
    Return values:
        total_data (pd.DataFrame): 병합된 데이터프레임
    """
    # 1. 토론토 에너지 데이터 불러오기
    elec_path = "C:/Users/ryudo/OneDrive - gachon.ac.kr/AiCE2/석사논문/Transformer/BDG2/data"
    elec = pd.read_csv(elec_path + "/electricity.csv", parse_dates=["timestamp"]) # parse_dates: timestamp 열을 datetime 형식으로 변환

    elec["Date/Time"] = elec["timestamp"].dt.date

    # 필요한 열만 선택 후 이름 변경
    elec = elec[["timestamp", "Date/Time", f"{building_name}"]]
    elec.rename(columns={f"{building_name}": "electricity"}, inplace=True)

    # 3. 날씨 데이터 불러오기
    # toronto_weather 폴더 경로
    weather_path = r"C:/Users/ryudo/OneDrive - gachon.ac.kr/AiCE2/석사논문/Transformer/BDG2/data/toronto_weather"

    # 파일별로 읽기
    w16 = pd.read_csv(weather_path + "/toronto_weather_2016.csv", parse_dates=["Date/Time"])
    w17 = pd.read_csv(weather_path + "/toronto_weather_2017.csv", parse_dates=["Date/Time"])

    # 합치기
    weather = pd.concat([w16, w17], ignore_index=True)

    # 3.1 날짜 열 형식 변환
    weather["Date/Time"] = weather["Date/Time"].dt.date

    # 4. 기간 필터
    start_date = pd.to_datetime("2016-06-01").date()
    end_date   = pd.to_datetime("2017-12-31").date()

    elec = elec[(elec["Date/Time"] >= start_date) & (elec["Date/Time"] <= end_date)]
    weather = weather[(weather["Date/Time"] >= start_date) & (weather["Date/Time"] <= end_date)]

    # 5. 날짜 기준으로 병합
    total_data = pd.merge(elec, weather, on="Date/Time", how="inner")

    # 6. 최종 열 선택
    total_data = total_data[["timestamp", "Date/Time", "electricity", "Mean Temp (°C)", "Total Rain (mm)"]]

    # 시간 순 정렬
    total_data.sort_values("timestamp", inplace=True)
    total_data.reset_index(drop=True, inplace=True)

    # 7. 결과 저장
    total_data.to_csv(elec_path + f"/toronto_data_{building_name}.csv", index=False)
    print(f"[저장 완료 | Building: {building_name}]: {elec_path}/toronto_data_{building_name}.csv")
    return total_data

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--building_name", type=str, required=True, help="건물 이름 (예: Panther_parking_Lorriane)")
#     args = parser.parse_args()

#     toronto_total_data_preparation(args.building_name)

# # run example: python data_concat.py --building_name Panther_parking_Lorriane