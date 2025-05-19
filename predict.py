import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 시트별 전처리 함수
def load_and_prepare(sheet_name):
    file_path = 'C:/Users/GSDEP_jshwang/Desktop/predict/12호기 기동관련 온도_처리.xlsx'
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 열 이름 정리
    df.columns = df.columns.str.strip()

    # 열 이름 확인 (디버깅 시 사용)
    print(f"[{sheet_name}] 열 목록:", df.columns.tolist())

    # 결측치 제거
    df = df.dropna()

    # Target T Min 변환 (시간 형식일 경우)
    if pd.api.types.is_timedelta64_dtype(df['TARGET_T_MIN']):
        df['TARGET_T_MIN'] = df['TARGET_T_MIN'].dt.total_seconds() / 60

    # 입력 변수 / 타깃 변수
    X = df[[
        'AMBIENT_TEMP_D',
        'MS_TEMP_D',
        'RH_BOWL_D',
        'TARGET_T_MIN',
        'AMBIENT_TEMP_T',
        'MS_TEMP_T'
    ]]
    y = df['RH_BOWL_T']

    return X, y

# 모델 훈련 함수
def train_and_evaluate(sheet_name):
    print(f'--- {sheet_name} 모델 학습 시작 ---')
    X, y = load_and_prepare(sheet_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'{sheet_name} RMSE:', rmse)

    return model

# 1호기 모델 학습
model1 = train_and_evaluate('1호기')

# 2호기 모델 학습
model2 = train_and_evaluate('2호기')

# 예측 입력 예시 (1호기)
sample_input = pd.DataFrame([{
    'AMBIENT_TEMP_D': 26.0,
    'MS_TEMP_D': 538.0,
    'RH_BOWL_D': 612.0,
    'TARGET_T_MIN': 90,
    'AMBIENT_TEMP_T': 25.2,
    'MS_TEMP_T': 536.9
}])

predicted_temp = model1.predict(sample_input)[0]
print("1호기 90분 후 RH BOWL 예측 온도:", predicted_temp)
