import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 데이터 전처리 함수
def load_and_prepare(sheet_name):
    file_path = '12호기 기동관련 온도_처리.xlsx'
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()  # 열 이름 정리
    df = df.dropna()  # 결측치 제거

    # 시간 데이터를 분으로 변환
    if pd.api.types.is_timedelta64_dtype(df['TARGET_T_MIN']):
        df['TARGET_T_MIN'] = df['TARGET_T_MIN'].dt.total_seconds() / 60

    # 입력 변수: AMBIENT_TEMP_T 포함
    X = df[[
        'AMBIENT_TEMP_D',
        'MS_TEMP_D',
        'RH_BOWL_D',
        'TARGET_T_MIN',
        'AMBIENT_TEMP_T'
    ]]

    # 출력 변수: 두 개 (멀티 아웃풋)
    y = df[['RH_BOWL_T', 'MS_TEMP_T']]

    return X, y

# 모델 학습 및 평가 함수
def train_and_evaluate(sheet_name):
    print(f'--- {sheet_name} 모델 학습 시작 ---')
    X, y = load_and_prepare(sheet_name)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 성능 평가 (RMSE)
    rmse_rh = np.sqrt(mean_squared_error(y_test['RH_BOWL_T'], y_pred[:, 0]))
    rmse_ms = np.sqrt(mean_squared_error(y_test['MS_TEMP_T'], y_pred[:, 1]))

    print(f'{sheet_name} RMSE (RH_BOWL_T): {rmse_rh:.2f}')
    print(f'{sheet_name} RMSE (MS_TEMP_T): {rmse_ms:.2f}')

    return model

# 모델 학습 
model1 = train_and_evaluate('1호기')
model2 = train_and_evaluate('2호기')

# 예측 샘플 입력
sample_input = pd.DataFrame([{
    'AMBIENT_TEMP_D': 26.0,
    'MS_TEMP_D': 538.0,
    'RH_BOWL_D': 612.0,
    'TARGET_T_MIN': 90,
    'AMBIENT_TEMP_T': 25.2
}])

# 예측 수행
predicted = model1.predict(sample_input)[0]
print("1호기 예측 결과:")
print(f"  RH_BOWL_T 예측값: {predicted[0]:.2f}")
print(f"  MS_TEMP_T 예측값: {predicted[1]:.2f}")

predicted = model2.predict(sample_input)[0]
print("2호기 예측 결과:")
print(f"  RH_BOWL_T 예측값: {predicted[0]:.2f}")
print(f"  MS_TEMP_T 예측값: {predicted[1]:.2f}")
