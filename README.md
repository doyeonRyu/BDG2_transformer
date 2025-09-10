# Transfomer 모델들을 사용해 BDG2 데이터 electricity 예측하기


참고   
> https://github.com/buds-lab/building-data-genome-project-2   
- BUDS Lab(University of Toronto)에서 주관한 공공 건물 에너지 사용량 예측 대회 및 연구 데이터셋  
- 1,636개 건물에서 수집된 2년치(2016-2017) 시계열 에너지 데이터로 구성    
- chilled water, gas, electricity, hotwater, irrigation, solar, steam, water 데이터 존재    
 
## main.py
- 건물마다의 모델 구현
- Panther_office_Hannah의 electricity 예측함  
- model 코드 변경되어 현재 개인 건물 예측 불가  

## Global model 구현
- Bobcat 지역의 건물들을 모두 학습하여 건물마다의 electricity를 예측하는 글로벌 모델 생성  
- Bobcat_office_Alissa 건물 electricity 예측함  
- 데이터  
    - long 포맷 데이터: 건물마다의 측정값들을 정렬 (건물별 시계열 데이터 정렬)  
    - metatdata: 건물 특성 정보 (시계열 데이터 아님)  
    - common data: 동일 지역, 날짜를 가지는 데이터 이므로 모두 동일한 값을 갖는 공통 데이터 (시계열 데이터)  
- 결과(test data)     
    ===========================================================================    
    Evaluating Transformer...   
    [ALL] MAE=19.3644, RMSE=28.0330, WAPE=28.58%, sMAPE=35.78%, MAPE=46.40%   
    [Alissa] MAE=20.5013, RMSE=27.6721, WAPE=34.11%, sMAPE=34.74%, MAPE=28.28%   

    Evaluating Informer...   
    [ALL] MAE=15.6382, RMSE=25.3326, WAPE=23.08%, sMAPE=26.85%, MAPE=29.59%  
    [Alissa] MAE=17.6277, RMSE=22.9307, WAPE=29.33%, sMAPE=29.91%, MAPE=29.55%  

    Evaluating Autoformer...  
    [ALL] MAE=17.8297, RMSE=28.8010, WAPE=26.31%, sMAPE=29.23%, MAPE=34.22%  
    [Alissa] MAE=27.2737, RMSE=34.8545, WAPE=45.38%, sMAPE=53.99%, MAPE=42.46%  

    Evaluating FEDformer...  
    [ALL] MAE=16.2593, RMSE=25.5339, WAPE=23.99%, sMAPE=28.13%, MAPE=31.97%  
    [Alissa] MAE=18.3837, RMSE=24.7452, WAPE=30.59%, sMAPE=31.55%, MAPE=27.32%  

    Evaluating PatchTST...    
    [ALL] MAE=12.9850, RMSE=25.5350, WAPE=19.16%, sMAPE=20.50%, MAPE=25.06%  
    [Alissa] MAE=48.8159, RMSE=80.6493, WAPE=81.23%, sMAPE=59.05%, MAPE=86.70%  
