import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional

from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.isotonic import IsotonicRegression
import joblib
import xgboost as xgb
import optuna


# ============================
# Configuration (edit as needed)
# ============================
EXCEL_FILE_PATH: str = '12호기 기동관련 온도 처리필요.xlsx'

# Training mode
COOLING_ONLY: bool = True  # 점화 이전 냉각 구간만 학습에 사용
INCLUDE_IGNITION_IN_TRAIN: bool = False  # 점화 시점 샘플을 학습에 포함할지 여부
SMOOTH_LABELS: bool = True  # 이벤트별 냉각 타깃 단조(비증가) 스무딩

# Constants (avoid magic numbers)
DATE_FORMAT: str = '%y.%m.%d %H:%M'
ISOTONIC_ANCHOR_WEIGHT: float = 10.0
MAX_VALID_ELAPSED_HOURS: float = 1000.0

# Model output paths
MODEL_PATHS: Dict[str, str] = {
    '1호기': 'model_1.pkl',
    '2호기': 'model_2.pkl',
}

PERFORMANCE_JSON_PATH: str = 'model_performance.json'

# Optuna Tuning
USE_OPTUNA_TUNING: bool = True
OPTUNA_TRIALS: int = 30
OPTUNA_CV_SPLITS: int = 5
XGB_RANDOM_STATE: int = 42

# Asymmetric loss (quantile) option
ASYMMETRIC_LOSS_ENABLED: bool = True  # 호기별로 quantile_alpha를 튜닝하여 적용
QUANTILE_ALPHA: float = 0.45  # 기본값(튜닝 전 초기값)
MBE_PENALTY_WEIGHT_DEFAULT: float = 0.5  # 1호기 외 기본 MBE 가중치

# Custom CV objective mixing RMSE and |MBE|
MBE_PENALTY_WEIGHT: float = 3.0  # final = rmse + weight * |mbe|
TUNE_MBE_WEIGHT: bool = False     # True면 Optuna가 mbe_weight도 함께 탐색

# (롤백) 참고곡선 관련 유틸 제거


def load_and_prepare(sheet_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Wide Format 엑셀 파일을 읽어 Long Format의 학습 데이터로 변환합니다. (위치 기반 최종판)
    - 컬럼 이름 대신, 위치(iloc)를 기준으로 데이터에 접근하여 안정성을 확보합니다.
    """
    print(f"[INFO] 파일 '{EXCEL_FILE_PATH}'에서 Wide Format 시트 '{sheet_name}' 로딩 중...")

    try:
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name, skiprows=1, header=[0, 1])
    except Exception as e:
        print(f"[ERROR] '{sheet_name}' 시트 로딩 중 심각한 오류 발생: {e}")
        return None, None, None

    all_samples = []
    event_id_counter = 0
    # using global DATE_FORMAT

    print("[INFO] 각 이벤트(행) 데이터 처리 시작...")
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

        # 이벤트별 점화 시점 계산 (있으면 냉각 구간 필터링에 사용)
        ignition_elapsed_hours = None
        try:
            ignition_main_col = [col for col in df.columns.levels[0] if '점화' in str(col)]
            if not ignition_main_col:
                ignition_main_col = [col for col in df.columns.levels[0] if '대기온도.1' in str(col)]
            if ignition_main_col:
                ignition_main_col = ignition_main_col[0]
                ignition_time_str = event_row[(ignition_main_col, '점화')]
                ignition_time = pd.to_datetime(ignition_time_str, format=DATE_FORMAT, errors='coerce')
                if not pd.isna(ignition_time):
                    elapsed_timedelta = ignition_time - start_time
                    ignition_elapsed_hours = elapsed_timedelta.total_seconds() / 3600.0
                    if ignition_elapsed_hours <= 0 or ignition_elapsed_hours >= MAX_VALID_ELAPSED_HOURS:
                        ignition_elapsed_hours = None
        except (KeyError, TypeError, IndexError):
            ignition_elapsed_hours = None

        # 이벤트 단위 샘플 수집 리스트 (냉각 구간만)
        event_samples = []

        # 0시간째 데이터도 이벤트 샘플에 추가 (계통분리 시점)
        elapsed_hours = 0.0
        sample0 = { 'event_id': event_id_counter, 'start_ms_temp': start_ms_temp, 'start_rh_bowl': start_rh_bowl, 'elapsed_hours': elapsed_hours, 'ambient_temp': start_ambient_temp, 'ambient_temp_squared': start_ambient_temp ** 2, 'start_temp_ambient_interaction': start_ms_temp * start_ambient_temp, 'time_temp_interaction': elapsed_hours * start_ambient_temp, 'sqrt_elapsed_hours': 0.0, 'log_time': 0, 'ms_temp': start_ms_temp, 'rh_bowl': start_rh_bowl }
        event_samples.append(sample0)

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

            # 점화 이전 냉각 구간만 포함
            if COOLING_ONLY and ignition_elapsed_hours is not None and elapsed_hours >= ignition_elapsed_hours:
                continue

            sample = { 'event_id': event_id_counter, 'start_ms_temp': start_ms_temp, 'start_rh_bowl': start_rh_bowl, 'elapsed_hours': elapsed_hours, 'ambient_temp': ambient_temp, 'ambient_temp_squared': ambient_temp ** 2, 'start_temp_ambient_interaction': start_ms_temp * ambient_temp, 'time_temp_interaction': elapsed_hours * ambient_temp, 'sqrt_elapsed_hours': np.sqrt(elapsed_hours), 'log_time': np.log1p(elapsed_hours), 'ms_temp': ms_temp, 'rh_bowl': rh_bowl }
            event_samples.append(sample)

        # 이벤트 단위 스무딩 적용 또는 원본 사용
        try:
            if SMOOTH_LABELS and len(event_samples) >= 2:
                # elapsed_hours 기준 정렬
                event_df = pd.DataFrame(event_samples)
                event_df = event_df.sort_values('elapsed_hours').reset_index(drop=True)

                x = event_df['elapsed_hours'].to_numpy(dtype=float)
                y_ms = event_df['ms_temp'].to_numpy(dtype=float)
                y_rh = event_df['rh_bowl'].to_numpy(dtype=float)

                # t=0 앵커를 더 강하게 고정하기 위한 가중치 (첫 샘플이 0시간임)
                w = np.ones_like(x)
                w[0] = ISOTONIC_ANCHOR_WEIGHT

                iso_dec = IsotonicRegression(increasing=False, out_of_bounds='clip')
                ms_smooth = iso_dec.fit_transform(x, y_ms, sample_weight=w)
                iso_dec2 = IsotonicRegression(increasing=False, out_of_bounds='clip')
                rh_smooth = iso_dec2.fit_transform(x, y_rh, sample_weight=w)

                # 스무딩된 타깃 반영 후 all_samples에 추가
                event_df['ms_temp'] = ms_smooth
                event_df['rh_bowl'] = rh_smooth
                all_samples.extend(event_df.to_dict(orient='records'))
            else:
                all_samples.extend(event_samples)
        except Exception:
            # 스무딩 실패 시 원본 사용
            all_samples.extend(event_samples)

        # 이벤트 종료 후 event_id 증가
        event_id_counter += 1

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
                    
                    # 냉각 전용 학습에서는 점화 시점 샘플을 포함하지 않음
                    if INCLUDE_IGNITION_IN_TRAIN and ignition_elapsed_hours > 0 and ignition_elapsed_hours < MAX_VALID_ELAPSED_HOURS:
                        ignition_sample = { 'event_id': event_id_counter, 'start_ms_temp': start_ms_temp, 'start_rh_bowl': start_rh_bowl, 'elapsed_hours': ignition_elapsed_hours, 'ambient_temp': ignition_ambient_temp, 'ambient_temp_squared': ignition_ambient_temp ** 2, 'start_temp_ambient_interaction': start_ms_temp * ignition_ambient_temp, 'time_temp_interaction': ignition_elapsed_hours * ignition_ambient_temp, 'sqrt_elapsed_hours': np.sqrt(ignition_elapsed_hours), 'log_time': np.log1p(ignition_elapsed_hours), 'ms_temp': ignition_ms_temp, 'rh_bowl': ignition_rh_bowl }
                        all_samples.append(ignition_sample)
        except (KeyError, TypeError, IndexError):
            pass

    if not all_samples:
        print("[WARN] 처리할 수 있는 유효한 데이터 샘플이 없습니다. 엑셀의 데이터 형식이나 헤더 구조를 다시 확인해주세요.")
        return None, None, None

    final_df = pd.DataFrame(all_samples)
    feature_cols = [ 'start_ms_temp', 'start_rh_bowl', 'elapsed_hours', 'ambient_temp', 'ambient_temp_squared', 'start_temp_ambient_interaction', 'time_temp_interaction', 'sqrt_elapsed_hours', 'log_time' ]
    X = final_df[feature_cols]
    y = final_df[['ms_temp', 'rh_bowl']]
    groups = final_df['event_id'].to_numpy(dtype=int)

    print(f"[INFO] 데이터 처리 완료. 최종 입력 변수 형태: {X.shape}, 최종 출력 변수 형태: {y.shape}, 이벤트 수: {final_df['event_id'].nunique()}")
    return X, y, groups


def _compute_avg_rmse(y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
    """두 타깃의 RMSE 평균을 계산합니다."""
    rmse_ms = float(np.sqrt(mean_squared_error(y_true['ms_temp'], y_pred[:, 0])))
    rmse_rh = float(np.sqrt(mean_squared_error(y_true['rh_bowl'], y_pred[:, 1])))
    return (rmse_ms + rmse_rh) / 2.0


def _compute_avg_abs_mbe(y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
    mbe_ms = float(np.mean(y_pred[:, 0] - y_true['ms_temp'].to_numpy(dtype=float)))
    mbe_rh = float(np.mean(y_pred[:, 1] - y_true['rh_bowl'].to_numpy(dtype=float)))
    return (abs(mbe_ms) + abs(mbe_rh)) / 2.0


def _compute_avg_abs_mbe(y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
    """두 타깃의 |MBE| 평균을 계산합니다."""
    mbe_ms = float(np.mean(y_pred[:, 0] - y_true['ms_temp'].to_numpy(dtype=float)))
    mbe_rh = float(np.mean(y_pred[:, 1] - y_true['rh_bowl'].to_numpy(dtype=float)))
    return (abs(mbe_ms) + abs(mbe_rh)) / 2.0


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
    monotone_constraints: Tuple[int, ...],
    unit_name: str,
) -> Dict[str, float]:
    """
    GroupKFold를 사용하여 Optuna로 XGBRegressor 하이퍼파라미터를 튜닝합니다.
    데이터 누수를 막기 위해 이벤트 단위 그룹 분할을 유지합니다.
    """
    unique_group_count = int(np.unique(groups).size)
    cv_splits = min(OPTUNA_CV_SPLITS, unique_group_count)
    if cv_splits < 2:
        # 분할이 불가능하면 기본값 반환
        return {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1.0,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
        }

    gkf = GroupKFold(n_splits=cv_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 20.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        }

        # Apply asymmetric objective if enabled and tune quantile_alpha (per unit)
        if ASYMMETRIC_LOSS_ENABLED:
            alpha_low, alpha_high = (0.40, 0.60) if unit_name == '2호기' else (0.10, 0.90)
            params.update({
                'objective': 'reg:quantileerror',
                'quantile_alpha': trial.suggest_float('quantile_alpha', alpha_low, alpha_high),
            })
        else:
            params.update({
                'objective': 'reg:squarederror',
            })

        # Per-unit MBE penalty weight: 1호기 고정 5.0, others use default
        penalty_weight = 5.0 if unit_name == '1호기' else 0.5

        fold_scores: List[float] = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            base = xgb.XGBRegressor(
                **params,
                random_state=XGB_RANDOM_STATE,
                n_jobs=-1,
                monotone_constraints=str(monotone_constraints),
            )
            model = MultiOutputRegressor(base)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            rmse_avg = _compute_avg_rmse(y_val, pred)
            mbe_avg_abs = _compute_avg_abs_mbe(y_val, pred)
            final_score = rmse_avg + penalty_weight * mbe_avg_abs
            fold_scores.append(final_score)

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=XGB_RANDOM_STATE))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=1)
    print(f"[INFO] Optuna best value (avg RMSE): {study.best_value:.4f}")
    print(f"[INFO] Optuna best params: {study.best_params}")

    best = study.best_params
    # 기록: 사용된 MBE 가중치 저장
    best['mbe_weight'] = 5.0 if unit_name == '1호기' else 0.5
    return best

def stratified_group_train_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    label_method: str = 'auto',  # 'auto' | 'top_quantile'
    top_quantile: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    이벤트(그룹) 단위를 유지하면서 타깃 분포를 유사하게 맞추기 위한 계층화 분할을 수행합니다.
    - 이벤트별 대표값(평균 ms_temp)을 기준으로 분위수(bin)를 만들고 StratifiedShuffleSplit 적용
    - 각 bin의 최소 그룹 수가 2 미만이면 bin 수를 줄이며 재시도
    - 끝까지 실패하면 GroupShuffleSplit으로 폴백

    반환: (train_indices, test_indices) - 원본 행 인덱스 배열
    """
    # 1) 이벤트 대표값 계산
    event_summary = pd.DataFrame({'event_id': groups, 'ms_temp': y['ms_temp']})
    event_mean_temps = event_summary.groupby('event_id', as_index=False)['ms_temp'].mean()

    # 2) 라벨링 및 분할 전략 선택
    n_groups = event_mean_temps['event_id'].nunique()

    # top_quantile 바이너리 라벨링 분기 (예: 상위 75% 고온 이벤트 = 1)
    if label_method == 'top_quantile':
        if n_groups < 4:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            return train_idx, test_idx

        threshold = float(event_mean_temps['ms_temp'].quantile(top_quantile))
        labels = (event_mean_temps['ms_temp'] >= threshold).astype(int)
        class_counts = labels.value_counts()
        if class_counts.size < 2 or class_counts.min() < 2:
            # 계층화 불가 → 폴백
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            return train_idx, test_idx

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_event_indices, test_event_indices = next(sss.split(event_mean_temps, labels))
    else:
        # auto: 분위수 다중 bin을 이용해 최대한 계층화 시도
        desired_bins = min(n_groups // 2, 10)
        if desired_bins < 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            return train_idx, test_idx

        success = False
        train_event_indices = None
        test_event_indices = None
        for b in range(desired_bins, 1, -1):
            try:
                labels = pd.qcut(event_mean_temps['ms_temp'], q=b, labels=False, duplicates='drop')
            except Exception:
                continue
            counts = pd.Series(labels).value_counts(dropna=True)
            if counts.size < 2 or counts.min() < 2:
                continue
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            try:
                train_event_indices, test_event_indices = next(sss.split(event_mean_temps, labels))
                success = True
                break
            except ValueError:
                continue
        if not success:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            return train_idx, test_idx

    # 4) 이벤트 ID를 원본 행 인덱스로 매핑
    train_event_ids = event_mean_temps.loc[train_event_indices, 'event_id'].unique()
    test_event_ids = event_mean_temps.loc[test_event_indices, 'event_id'].unique()

    original_indices = np.arange(len(X))
    train_idx = original_indices[np.isin(groups, train_event_ids)]
    test_idx = original_indices[np.isin(groups, test_event_ids)]

    return train_idx, test_idx

def train_and_evaluate(sheet_name: str):
    print(f'[INFO] --- {sheet_name} 모델 학습 시작 ---')
    X, y, groups = load_and_prepare(sheet_name)
    if X is None or y is None or groups is None:
        print(f"[ERROR] {sheet_name} 데이터 처리 실패")
        return None, None

    # --- 데이터 분할: 이벤트 단위 계층화 분할 (Stratified Group Split) ---
    print('[INFO] 이벤트 단위 계층화 분할 시작...')
    # 모델 선택 과정에서 테스트 정보 누수를 방지하기 위해 항상 'auto' 사용
    train_idx, test_idx = stratified_group_train_test_split(
        X, y, groups, test_size=0.2, random_state=42, label_method='auto'
    )
    # 최종 데이터셋 분할
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f'[INFO] 데이터 분할 완료. Train: {len(X_train)} 샘플 ({pd.Series(groups[train_idx]).nunique()} 이벤트), Test: {len(X_test)} 샘플 ({pd.Series(groups[test_idx]).nunique()} 이벤트)')
    # --- 데이터 분할 로직 종료 ---

    # (롤백) 참고곡선 피처 사용 안 함

    # 단조 제약 설정 (feature_cols 순서에 맞춤)
    #  start_ms_temp: +1, start_rh_bowl: +1,
    #  elapsed_hours: -1, ambient_temp: +1, ambient_temp_squared: +1,
    #  start_temp_ambient_interaction: +1,
    #  time_temp_interaction: -1 (시간 증가에 따른 상호작용도 감소하도록)
    #  sqrt_elapsed_hours: -1, log_time: -1
    #  → 총 9개
    monotone_constraints = (1, 1, -1, 1, 1, 1, -1, -1, -1)

    # 하이퍼파라미터 튜닝 (학습 세트에서만 수행)
    groups_train = groups[train_idx]
    if USE_OPTUNA_TUNING:
        best_params = tune_hyperparameters(X_train, y_train, groups_train, monotone_constraints, sheet_name)
    else:
        best_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1.0,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
        }

    # Ensure objective settings according to asymmetric option
    if ASYMMETRIC_LOSS_ENABLED:
        # best_params already contains tuned 'quantile_alpha' from study
        best_params.update({'objective': 'reg:quantileerror'})
    else:
        best_params.update({'objective': 'reg:squarederror'})

    base_model = xgb.XGBRegressor(
        **best_params,
        random_state=XGB_RANDOM_STATE,
        n_jobs=-1,
        monotone_constraints=str(monotone_constraints)
    )

    # Baseline: MultiOutputRegressor
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse_ms = float(np.sqrt(mean_squared_error(y_test['ms_temp'], y_pred[:, 0])))
    rmse_rh = float(np.sqrt(mean_squared_error(y_test['rh_bowl'], y_pred[:, 1])))
    avg_rmse = float((rmse_ms + rmse_rh) / 2.0)

    # MAE
    mae_ms = float(mean_absolute_error(y_test['ms_temp'], y_pred[:, 0]))
    mae_rh = float(mean_absolute_error(y_test['rh_bowl'], y_pred[:, 1]))

    # MBE(Mean Bias Error): mean(pred - true). 음수면 과소예측, 양수면 과대예측
    mbe_ms = float(np.mean(y_pred[:, 0] - y_test['ms_temp'].to_numpy(dtype=float)))
    mbe_rh = float(np.mean(y_pred[:, 1] - y_test['rh_bowl'].to_numpy(dtype=float)))

    print(f'[METRIC][BASE] {sheet_name} RMSE (MS Temp): {rmse_ms:.2f}')
    print(f'[METRIC][BASE] {sheet_name} RMSE (RH BOWL): {rmse_rh:.2f}')
    print(f'[METRIC][BASE] {sheet_name} 평균 RMSE: {avg_rmse:.2f}')
    print(f'[METRIC][BASE] {sheet_name} MAE  (MS Temp): {mae_ms:.2f}')
    print(f'[METRIC][BASE] {sheet_name} MAE  (RH BOWL): {mae_rh:.2f}')
    print(f'[METRIC][BASE] {sheet_name} MBE  (MS Temp, pred-true): {mbe_ms:.2f}')
    print(f'[METRIC][BASE] {sheet_name} MBE  (RH BOWL, pred-true): {mbe_rh:.2f}')

    perf = {
        'rmse_ms': rmse_ms,
        'rmse_rh': rmse_rh,
        'avg_rmse': avg_rmse,
        'mae_ms': mae_ms,
        'mae_rh': mae_rh,
        'mbe_ms': mbe_ms,
        'mbe_rh': mbe_rh,
    }

    # 기록: 튜닝된 quantile_alpha(사용 시)
    if ASYMMETRIC_LOSS_ENABLED:
        try:
            perf['quantile_alpha'] = float(base_model.get_xgb_params().get('quantile_alpha', best_params.get('quantile_alpha')))
        except Exception:
            perf['quantile_alpha'] = float(best_params.get('quantile_alpha')) if 'quantile_alpha' in best_params else None

    # 기록: 사용된 MBE 가중치(1호기 5.0 고정, 그 외 기본)
    perf['mbe_weight'] = float(best_params.get('mbe_weight', 5.0 if sheet_name == '1호기' else MBE_PENALTY_WEIGHT_DEFAULT))

    return model, perf


def main() -> None:
    performances: Dict[str, Dict[str, float]] = {}

    for sheet_name, model_path in MODEL_PATHS.items():
        print(f"[INFO] {sheet_name} 단일 학습 시작 (label_method='auto')...")
        model, perf = train_and_evaluate(sheet_name)
        if model is None or perf is None:
            print(f"[ERROR] {sheet_name} 모델 학습 실패, 저장을 건너뜁니다.")
            continue

        joblib.dump(model, model_path)
        print(f"[INFO] {sheet_name} 모델 저장 완료 -> {model_path}")

        # --- Feature Importance 추출 및 저장 ---
        try:
            estimators = getattr(model, 'estimators_', [])
            # booster에서 학습 시 feature_names를 가져옴
            feature_names: List[str] = []
            try:
                if estimators:
                    names = estimators[0].get_booster().feature_names
                    if isinstance(names, (list, tuple)) and len(names) > 0:
                        feature_names = list(names)
            except Exception:
                feature_names = []

            per_target_imps: List[Dict[str, float]] = []
            for est in estimators:
                try:
                    booster = est.get_booster()
                    imp_raw = booster.get_score(importance_type='gain')  # {feature_name: importance}
                except Exception:
                    imp_raw = {}
                # 모든 피처에 대해 값 보정(없으면 0)
                fixed = {name: float(imp_raw.get(name, 0.0)) for name in (feature_names or imp_raw.keys())}
                per_target_imps.append(fixed)

            # 타깃 순서: 0=MS Temp, 1=RH BOWL (학습 y 컬럼 순서 기준)
            ms_imp = per_target_imps[0] if len(per_target_imps) > 0 else {}
            rh_imp = per_target_imps[1] if len(per_target_imps) > 1 else {}
            # 평균 중요도 계산
            all_keys = set(ms_imp.keys()) | set(rh_imp.keys())
            avg_imp = {k: float(ms_imp.get(k, 0.0) + rh_imp.get(k, 0.0)) / 2.0 for k in all_keys}

            importance_payload = {
                'ms_temp': ms_imp,
                'rh_bowl': rh_imp,
                'average': avg_imp,
            }
            imp_path = f"feature_importance_{sheet_name}.json"
            with open(imp_path, 'w', encoding='utf-8') as f:
                json.dump(importance_payload, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 특성 중요도 저장 완료 -> {imp_path}")
        except Exception as e:
            print(f"[WARN] 특성 중요도 저장 실패: {e}")

        performances[sheet_name] = perf

    if performances:
        with open(PERFORMANCE_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(performances, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 모델 성능 지표 저장 완료 -> {PERFORMANCE_JSON_PATH}")


if __name__ == '__main__':
    main()


