import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import joblib
import xgboost as xgb


# ============================
# Configuration (edit as needed)
# ============================
EXCEL_FILE_PATH: str = '12호기 기동관련 온도 처리필요.xlsx'

# Column names in the Excel file
TIME_COL: str = 'Tag'

# Per-sheet exact column mapping based on actual headers
SHEET_COLUMN_MAP: Dict[str, Dict[str, str]] = {
    '1호기': {
        'ambient': 'GSDEPDP.T1.T1-35320JGCTT58_A',
        'ms': 'GSDEPDP.T1.T1-31100JTBTE05_A',
        'rh': 'GSDEPDP.T1.T1-31100JTBTE21_A',
    },
    '2호기': {
        'ambient': 'GSDEPDP.T2.T2-35320JGCTT58_A',
        'ms': 'GSDEPDP.T2.T2-31100JTBTE05_A',
        'rh': 'GSDEPDP.T1.T1-31100JTBTE21_A',
    },
}

# Model output paths
MODEL_PATHS: Dict[str, str] = {
    '1호기': 'model_1.pkl',
    '2호기': 'model_2.pkl',
}

PERFORMANCE_JSON_PATH: str = 'model_performance.json'


def _ensure_required_columns(df: pd.DataFrame, ambient_col: str, ms_col: str, rh_col: str) -> None:
    missing: List[str] = [c for c in [TIME_COL, ambient_col, ms_col, rh_col] if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}. 엑셀 헤더명을 확인하거나 train.py 상단의 컬럼 상수를 실제 이름으로 수정하세요.")


def load_and_prepare(sheet_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Wide Format 엑셀 파일을 읽어 Long Format의 학습 데이터로 변환합니다. (위치 기반 최종판)
    - 컬럼 이름 대신, 위치(iloc)를 기준으로 데이터에 접근하여 안정성을 확보합니다.
    """
    print(f"파일 '{EXCEL_FILE_PATH}'에서 Wide Format 시트 '{sheet_name}' 로딩 중...")

    try:
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name, skiprows=1, header=[0, 1])
    except Exception as e:
        print(f"❌ '{sheet_name}' 시트 로딩 중 심각한 오류 발생: {e}")
        return None, None

    all_samples = []
    DATE_FORMAT = '%y.%m.%d %H:%M'

    print("각 이벤트(행) 데이터 처리 시작...")
    for index, event_row in df.iterrows():
        try:
            # --- 위치 기반 데이터 접근 (가장 중요) ---
            # 0번째 열: 계통분리 시간, 1번째: 대기온도, 2번째: MS Temp, 3번째: RH BOWL
            start_time_str = event_row.iloc[0]
            start_ambient_temp = event_row.iloc[1] # 0시간째 대기온도
            start_ms_temp = event_row.iloc[2]
            start_rh_bowl = event_row.iloc[3]
            
            start_time = pd.to_datetime(start_time_str, format=DATE_FORMAT, errors='coerce')

            if pd.isna(start_time) or pd.isna(start_ms_temp) or pd.isna(start_rh_bowl):
                continue
        except (KeyError, IndexError):
            continue

        # 0시간째 데이터도 학습 샘플에 추가 (계통분리 시점)
        elapsed_hours = 0.0
        elapsed_minutes = 0.0
        elapsed_days = 0.0
        sample = { 'start_ms_temp': start_ms_temp, 'start_rh_bowl': start_rh_bowl, 'elapsed_hours': elapsed_hours, 'ambient_temp': start_ambient_temp, 'ambient_temp_squared': start_ambient_temp ** 2, 'time_temp_interaction': elapsed_hours * start_ambient_temp, 'cooling_rate': 0, 'log_time': 0, 'elapsed_minutes': elapsed_minutes, 'elapsed_days': elapsed_days, 'ms_temp': start_ms_temp, 'rh_bowl': start_rh_bowl }
        all_samples.append(sample)

        # 12hr, 24hr... 데이터 처리
        time_points = df.columns.levels[0]
        # 'Unnamed:' 로 시작하는 초기 4개 컬럼과 점화 컬럼을 제외하고 순회
        time_labels_to_process = [lbl for lbl in time_points if 'hr' in str(lbl)]

        for time_label in time_labels_to_process:
            try:
                elapsed_hours = float(str(time_label).replace('hr', ''))
                
                ambient_temp = event_row[(time_label, '대기온도')]
                ms_temp = event_row[(time_label, 'MS Temp')]
                rh_bowl = event_row[(time_label, 'RH BOWL')]
                
                if pd.isna(ambient_temp) or pd.isna(ms_temp) or pd.isna(rh_bowl):
                    break
            except KeyError:
                continue

            elapsed_minutes = elapsed_hours * 60
            elapsed_days = elapsed_hours / 24
            sample = { 'start_ms_temp': start_ms_temp, 'start_rh_bowl': start_rh_bowl, 'elapsed_hours': elapsed_hours, 'ambient_temp': ambient_temp, 'ambient_temp_squared': ambient_temp ** 2, 'time_temp_interaction': elapsed_hours * ambient_temp, 'cooling_rate': np.sqrt(elapsed_hours), 'log_time': np.log1p(elapsed_hours), 'elapsed_minutes': elapsed_minutes, 'elapsed_days': elapsed_days, 'ms_temp': ms_temp, 'rh_bowl': rh_bowl }
            all_samples.append(sample)

        # 점화 데이터 처리 (이름이 특이하게 읽히므로 위치와 이름 혼합 사용)
        try:
            # 점화 데이터는 보통 뒤쪽에 있으므로, 이름에 '점화'가 포함된 컬럼을 찾습니다.
            ignition_main_col = [col for col in df.columns.levels[0] if '점화' in str(col)]
            if not ignition_main_col:
                ignition_main_col = [col for col in df.columns.levels[0] if '대기온도.1' in str(col)] # 대체 검색

            if ignition_main_col:
                ignition_main_col = ignition_main_col[0] # 첫 번째로 찾은 것 사용
                
                ignition_time_str = event_row[(ignition_main_col, '점화')]
                ignition_ambient_temp = event_row[(ignition_main_col, '대기온도.1')]
                ignition_ms_temp = event_row[(ignition_main_col, 'MS Temp.1')]
                ignition_rh_bowl = event_row[(ignition_main_col, 'RH BOWL.2')]

                ignition_time = pd.to_datetime(ignition_time_str, format=DATE_FORMAT, errors='coerce')
                
                if not (pd.isna(ignition_time) or pd.isna(ignition_ambient_temp) or pd.isna(ignition_ms_temp) or pd.isna(ignition_rh_bowl)):
                    elapsed_timedelta = ignition_time - start_time
                    ignition_elapsed_hours = elapsed_timedelta.total_seconds() / 3600.0
                    
                    if ignition_elapsed_hours > 0 and ignition_elapsed_hours < 1000:
                        elapsed_minutes = ignition_elapsed_hours * 60
                        elapsed_days = ignition_elapsed_hours / 24
                        ignition_sample = { 'start_ms_temp': start_ms_temp, 'start_rh_bowl': start_rh_bowl, 'elapsed_hours': ignition_elapsed_hours, 'ambient_temp': ignition_ambient_temp, 'ambient_temp_squared': ignition_ambient_temp ** 2, 'time_temp_interaction': ignition_elapsed_hours * ignition_ambient_temp, 'cooling_rate': np.sqrt(ignition_elapsed_hours), 'log_time': np.log1p(ignition_elapsed_hours), 'elapsed_minutes': elapsed_minutes, 'elapsed_days': elapsed_days, 'ms_temp': ignition_ms_temp, 'rh_bowl': ignition_rh_bowl }
                        all_samples.append(ignition_sample)
        except (KeyError, TypeError, IndexError):
            pass

    if not all_samples:
        print("⚠️ 처리할 수 있는 유효한 데이터 샘플이 없습니다. 엑셀의 데이터 형식이나 헤더 구조를 다시 확인해주세요.")
        return None, None

    final_df = pd.DataFrame(all_samples)
    feature_cols = [ 'start_ms_temp', 'start_rh_bowl', 'elapsed_hours', 'ambient_temp', 'ambient_temp_squared', 'time_temp_interaction', 'cooling_rate', 'log_time', 'elapsed_minutes', 'elapsed_days' ]
    X = final_df[feature_cols]
    y = final_df[['ms_temp', 'rh_bowl']]

    print(f"데이터 처리 완료. 최종 입력 변수 형태: {X.shape}, 최종 출력 변수 형태: {y.shape}")
    return X, y

def train_and_evaluate(sheet_name: str):
    print(f'--- {sheet_name} 모델 학습 시작 ---')
    X, y = load_and_prepare(sheet_name)
    if X is None or y is None:
        print(f"❌ {sheet_name} 데이터 처리 실패")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1,
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse_ms = float(np.sqrt(mean_squared_error(y_test['ms_temp'], y_pred[:, 0])))
    rmse_rh = float(np.sqrt(mean_squared_error(y_test['rh_bowl'], y_pred[:, 1])))
    avg_rmse = float((rmse_ms + rmse_rh) / 2.0)

    print(f'{sheet_name} RMSE (MS Temp): {rmse_ms:.2f}')
    print(f'{sheet_name} RMSE (RH BOWL): {rmse_rh:.2f}')
    print(f'{sheet_name} 평균 RMSE: {avg_rmse:.2f}')

    return model, {
        'rmse_ms': rmse_ms,
        'rmse_rh': rmse_rh,
        'avg_rmse': avg_rmse,
    }


def main() -> None:
    performances: Dict[str, Dict[str, float]] = {}

    for sheet_name, model_path in MODEL_PATHS.items():
        model, perf = train_and_evaluate(sheet_name)
        if model is None or perf is None:
            print(f"❌ {sheet_name} 모델 학습 실패, 저장을 건너뜁니다.")
            continue

        joblib.dump(model, model_path)
        print(f"✅ {sheet_name} 모델 저장 완료 → {model_path}")

        performances[sheet_name] = perf

    if performances:
        with open(PERFORMANCE_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(performances, f, ensure_ascii=False, indent=2)
        print(f"✅ 모델 성능 지표 저장 완료 → {PERFORMANCE_JSON_PATH}")


if __name__ == '__main__':
    main()


