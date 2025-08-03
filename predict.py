import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 데이터 전처리 함수
def load_and_prepare(sheet_name):
    file_path = '12호기 기동관련 온도 처리필요.xlsx'
    
    print(f"파일 '{file_path}'에서 시트 '{sheet_name}' 로딩 중...")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"데이터 로드 완료. 형태: {df.shape}")
        print(f"컬럼명: {list(df.columns)}")
        print(f"처음 10행:")
        print(df.head(10))
        
        # 데이터 정리
        df.columns = df.columns.str.strip()
        
        # 실제 데이터가 있는 행 찾기 (Tag 컬럼에 날짜/시간 형식이 있는 행들)
        valid_rows = []
        for idx, row in df.iterrows():
            tag_value = str(row['Tag']) if pd.notna(row['Tag']) else ''
            # 날짜/시간 형식인지 확인 (예: 2017-04-05 21:16:00)
            if '-' in tag_value and ':' in tag_value and len(tag_value) > 10:
                valid_rows.append(idx)
        
        if len(valid_rows) == 0:
            print("⚠️  유효한 날짜/시간 데이터를 찾을 수 없습니다!")
            return None, None
            
        # 유효한 데이터만 선택
        df = df.iloc[valid_rows].reset_index(drop=True)
        print(f"유효한 데이터 행 선택 후 형태: {df.shape}")
        
        # 시간 데이터 처리
        try:
            df['Tag'] = pd.to_datetime(df['Tag'], errors='coerce')
            df = df.dropna(subset=['Tag'])
            print(f"시간 변환 후 데이터 형태: {df.shape}")
        except Exception as e:
            print(f"시간 변환 오류: {e}")
            return None, None
        
        # 시계열 데이터 특성 파악
        # 각 계통분리 이벤트별로 그룹화
        df = df.sort_values('Tag')
        
        # 계통분리 이벤트 찾기 (첫 번째 데이터 포인트를 각 이벤트의 시작점으로 간주)
        events = []
        current_event = []
        
        for idx, row in df.iterrows():
            if len(current_event) == 0:
                # 새로운 이벤트 시작
                current_event = [row]
            else:
                # 같은 이벤트에 추가
                current_event.append(row)
                
                # 96시간(4일) 데이터가 모이면 이벤트 완료
                if len(current_event) >= 8:  # 12시간 간격으로 8개 포인트 = 96시간
                    events.append(current_event)
                    current_event = []
        
        # 마지막 이벤트도 추가
        if len(current_event) > 0:
            events.append(current_event)
        
        print(f"총 {len(events)}개의 계통분리 이벤트 발견")
        
        # 각 이벤트별로 시계열 데이터 처리
        all_samples = []
        
        for event_idx, event_data in enumerate(events):
            event_df = pd.DataFrame(event_data)
            print(f"\n이벤트 {event_idx + 1}: {len(event_df)}개 데이터 포인트")
            
            # 시간 간격 계산 (분 단위)
            event_df = event_df.sort_values('Tag')
            start_time = event_df['Tag'].iloc[0]
            
            # 숫자 컬럼 찾기
            numeric_cols = []
            for col in event_df.columns:
                if col != 'Tag' and col != 'i':
                    try:
                        pd.to_numeric(event_df[col], errors='coerce')
                        if event_df[col].notna().sum() > 0:
                            numeric_cols.append(col)
                    except:
                        continue
            
            print(f"숫자 컬럼들: {numeric_cols}")
            
            # 각 데이터 포인트에 대해 경과 시간 계산
            for idx, row in event_df.iterrows():
                elapsed_minutes = (row['Tag'] - start_time).total_seconds() / 60
                
                # 입력 변수: 경과 시간(분), 외기온도
                # 출력 변수: MS Temp, RH BOWL
                
                if len(numeric_cols) >= 3:  # 외기온도, MS Temp, RH BOWL
                    # 첫 번째 컬럼을 외기온도로 가정 (GSDEPDP.T1.T1-35320JGCTT58_A)
                    ambient_temp_col = numeric_cols[0]
                    ambient_temp = row[ambient_temp_col] if pd.notna(row[ambient_temp_col]) else 0
                    
                    # MS Temp와 RH BOWL 찾기 (숫자 컬럼 중에서)
                    temp_cols = [col for col in numeric_cols if 'TEMP' in col.upper() or 'MS' in col.upper()]
                    bowl_cols = [col for col in numeric_cols if 'BOWL' in col.upper() or 'RH' in col.upper()]
                    
                    ms_temp = None
                    rh_bowl = None
                    
                    if len(temp_cols) > 0:
                        ms_temp = row[temp_cols[0]] if pd.notna(row[temp_cols[0]]) else 0
                    else:
                        # MS Temp 컬럼을 찾지 못한 경우, 두 번째 컬럼 사용
                        ms_temp = row[numeric_cols[1]] if len(numeric_cols) > 1 and pd.notna(row[numeric_cols[1]]) else 0
                    
                    if len(bowl_cols) > 0:
                        rh_bowl = row[bowl_cols[0]] if pd.notna(row[bowl_cols[0]]) else 0
                    else:
                        # RH BOWL 컬럼을 찾지 못한 경우, 세 번째 컬럼 사용
                        rh_bowl = row[numeric_cols[2]] if len(numeric_cols) > 2 and pd.notna(row[numeric_cols[2]]) else 0
                    
                    # 유효한 데이터만 추가
                    if pd.notna(ambient_temp) and pd.notna(ms_temp) and pd.notna(rh_bowl):
                        sample = {
                            'elapsed_minutes': elapsed_minutes,
                            'ambient_temp': ambient_temp,
                            'ms_temp': ms_temp,
                            'rh_bowl': rh_bowl
                        }
                        all_samples.append(sample)
                        print(f"  샘플 추가: {elapsed_minutes:.0f}분, 외기온도: {ambient_temp:.1f}°C, MS: {ms_temp:.1f}°C, RH: {rh_bowl:.1f}°C")
        
        if len(all_samples) == 0:
            print("⚠️  유효한 시계열 샘플을 찾을 수 없습니다!")
            return None, None
        
        # 데이터프레임으로 변환
        samples_df = pd.DataFrame(all_samples)
        print(f"총 {len(samples_df)}개의 시계열 샘플 생성")
        print(f"샘플 데이터 형태: {samples_df.shape}")
        print(f"샘플 데이터 처음 5행:")
        print(samples_df.head())
        
        # 입력 변수: 경과 시간, 외기온도
        X = samples_df[['elapsed_minutes', 'ambient_temp']]
        
        # 출력 변수: MS Temp, RH BOWL (멀티 아웃풋)
        y = samples_df[['ms_temp', 'rh_bowl']]
        
        print(f"최종 입력 변수 형태: {X.shape}")
        print(f"최종 출력 변수 형태: {y.shape}")
        print(f"입력 변수 컬럼: {list(X.columns)}")
        print(f"출력 변수 컬럼: {list(y.columns)}")
        
        return X, y
        
    except Exception as e:
        print(f"❌ 데이터 로딩 오류: {str(e)}")
        raise e

# 모델 학습 및 평가 함수
def train_and_evaluate(sheet_name):
    print(f'--- {sheet_name} 모델 학습 시작 ---')
    X, y = load_and_prepare(sheet_name)
    
    if X is None or y is None:
        print(f"❌ {sheet_name} 데이터 처리 실패")
        return None, None, None

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습 (멀티 아웃풋)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 성능 평가 (RMSE)
    rmse_ms = np.sqrt(mean_squared_error(y_test['ms_temp'], y_pred[:, 0]))
    rmse_rh = np.sqrt(mean_squared_error(y_test['rh_bowl'], y_pred[:, 1]))
    avg_rmse = (rmse_ms + rmse_rh) / 2
    
    print(f'{sheet_name} RMSE (MS Temp): {rmse_ms:.2f}')
    print(f'{sheet_name} RMSE (RH BOWL): {rmse_rh:.2f}')
    print(f'{sheet_name} 평균 RMSE: {avg_rmse:.2f}')

    return model, X.columns.tolist(), avg_rmse  # 모델, 입력 컬럼명 리스트, 평균 RMSE 반환

# 모델 학습 
model1, input_cols1, rmse1 = train_and_evaluate('1호기')
model2, input_cols2, rmse2 = train_and_evaluate('2호기')

# 예측 샘플 입력 (시계열 특성 반영)
if model1 is not None:
    # 계통분리 후 48시간 경과, 외기온도 20도
    sample_input = pd.DataFrame([{
        'elapsed_minutes': 48 * 60,  # 48시간을 분으로
        'ambient_temp': 20.0
    }])
    
    # 예측 수행
    predicted = model1.predict(sample_input)[0]
    print("1호기 예측 결과 (48시간 경과, 외기온도 20°C):")
    print(f"  MS Temp 예측값: {predicted[0]:.2f}°C")
    print(f"  RH BOWL 예측값: {predicted[1]:.2f}°C")
    print(f"  RMSE: {rmse1:.2f}")

if model2 is not None:
    # 계통분리 후 48시간 경과, 외기온도 20도
    sample_input = pd.DataFrame([{
        'elapsed_minutes': 48 * 60,  # 48시간을 분으로
        'ambient_temp': 20.0
    }])
    
    # 예측 수행
    predicted = model2.predict(sample_input)[0]
    print("2호기 예측 결과 (48시간 경과, 외기온도 20°C):")
    print(f"  MS Temp 예측값: {predicted[0]:.2f}°C")
    print(f"  RH BOWL 예측값: {predicted[1]:.2f}°C")
    print(f"  RMSE: {rmse2:.2f}")
