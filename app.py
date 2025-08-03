import streamlit as st
import pandas as pd
from predict import train_and_evaluate

# 페이지 설정 (반드시 가장 먼저 호출되어야 함)
st.set_page_config(page_title="터빈 냉각 온도 예측 시스템", layout="wide")

def calculate_confidence(rmse, max_value=1000):
    """RMSE를 기반으로 신뢰도를 계산 (0-100%)"""
    # RMSE가 낮을수록 신뢰도가 높음
    confidence = max(0, 100 - (rmse / max_value) * 100)
    return round(confidence, 1)

def get_model_performance():
    """모델 성능 정보를 가져오는 함수"""
    try:
        # 1호기 모델 성능
        model1_result = train_and_evaluate('1호기')
        if model1_result and model1_result[0] is not None:
            model1, input_cols1, rmse1 = model1_result
            confidence_1 = calculate_confidence(rmse1)
        else:
            rmse1 = None
            confidence_1 = None
            
        # 2호기 모델 성능
        model2_result = train_and_evaluate('2호기')
        if model2_result and model2_result[0] is not None:
            model2, input_cols2, rmse2 = model2_result
            confidence_2 = calculate_confidence(rmse2)
        else:
            rmse2 = None
            confidence_2 = None
            
        return {
            '1호기': {'rmse': rmse1, 'confidence': confidence_1},
            '2호기': {'rmse': rmse2, 'confidence': confidence_2}
        }
    except Exception as e:
        st.error(f"모델 성능 정보를 가져오는 중 오류: {str(e)}")
        return None

def main():
    # 제목
    st.title("터빈 냉각 온도 예측 시스템")
    st.markdown("---")

    # 호기 선택
    selected_model = st.radio(
        "예측할 호기를 선택하세요",
        ("1호기", "2호기"),
        horizontal=True
    )
    st.markdown("---")

    # 사이드바에 설명 추가
    with st.sidebar:
        st.header("입력 가이드")
        st.markdown("""
        **터빈 냉각 시계열 예측**
        
        - **경과 시간**: 계통분리 후 경과된 시간 (분)
        - **외기온도**: 현재 외기온도 (°C)
        
        **출력 예측값**
        - **MS Temp**: 터빈 MS 온도 (°C)
        - **RH BOWL**: 터빈 RH 보울 온도 (°C)
        """)
        
        st.markdown("---")
        
        # 모델 신뢰도 정보
        st.header("📊 모델 신뢰도")
        
        # 모델 성능 정보 가져오기
        performance = get_model_performance()
        
        if performance:
            for unit, metrics in performance.items():
                if metrics['rmse'] is not None:
                    st.subheader(f"{unit}")
                    
                    # 신뢰도 표시
                    confidence = metrics['confidence']
                    if confidence >= 80:
                        confidence_color = "🟢"
                        confidence_text = "높음"
                    elif confidence >= 60:
                        confidence_color = "🟡"
                        confidence_text = "보통"
                    else:
                        confidence_color = "🔴"
                        confidence_text = "낮음"
                    
                    st.metric(
                        label=f"{confidence_color} 예측 신뢰도",
                        value=f"{confidence}%",
                        delta=f"{confidence_text}"
                    )
                    
                    # RMSE 표시
                    st.metric(
                        label="RMSE",
                        value=f"{metrics['rmse']:.2f}",
                        delta="낮을수록 좋음"
                    )
                    
                    # 신뢰도 설명
                    if confidence >= 80:
                        st.info("✅ 높은 신뢰도: 예측이 매우 정확합니다.")
                    elif confidence >= 60:
                        st.warning("⚠️ 보통 신뢰도: 예측이 적당히 정확합니다.")
                    else:
                        st.error("❌ 낮은 신뢰도: 예측 정확도가 낮습니다.")
                    
                    st.markdown("---")
        else:
            st.warning("모델 성능 정보를 불러올 수 없습니다.")

    # 입력 폼
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # 경과 시간 입력 (시간 단위로 입력받고 분으로 변환)
            elapsed_hours = st.number_input("경과 시간 (시간)", value=48.0, min_value=0.0, step=0.5, help="계통분리 후 경과된 시간")
            elapsed_minutes = elapsed_hours * 60  # 시간을 분으로 변환
            
            # 외기온도 입력
            ambient_temp = st.number_input("외기온도 (°C)", value=20.0, step=0.1, help="현재 외기온도")
        
        with col2:
            st.info("💡 **참고사항**")
            st.markdown("""
            - **입력값**: 경과 시간 + 외기온도
            - **출력값**: MS Temp + RH BOWL 온도
            - **모델**: 시계열 기반 RandomForest
            - **데이터**: 계통분리 후 96시간까지의 시계열 데이터
            """)
        
        submitted = st.form_submit_button("예측 실행")

    # 모델 로드 및 예측
    if 'submitted' in locals() and submitted:
        with st.spinner(f'{selected_model} 모델을 불러오는 중...'):
            try:
                # 선택된 호기 모델 로드
                model_result = train_and_evaluate(selected_model)
                
                if model_result is None or model_result[0] is None:
                    st.error(f"{selected_model} 모델 로딩에 실패했습니다.")
                    return
                
                model, input_columns, rmse = model_result
                
                # 입력 데이터 준비
                input_data = pd.DataFrame([{
                    'elapsed_minutes': elapsed_minutes,
                    'ambient_temp': ambient_temp
                }])
                
                # 예측 실행
                predicted = model.predict(input_data)[0]
                
                # 신뢰도 계산
                confidence = calculate_confidence(rmse)
                
                # 결과 표시
                st.success(f"{selected_model} 예측이 완료되었습니다!")
                st.markdown("### 📊 예측 결과")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("경과 시간", f"{elapsed_hours:.1f}시간")
                with col2:
                    st.metric("외기온도", f"{ambient_temp:.1f}°C")
                with col3:
                    # 신뢰도 표시
                    if confidence >= 80:
                        confidence_color = "🟢"
                        confidence_text = "높음"
                    elif confidence >= 60:
                        confidence_color = "🟡"
                        confidence_text = "보통"
                    else:
                        confidence_color = "🔴"
                        confidence_text = "낮음"
                    
                    st.metric(
                        label=f"{confidence_color} 신뢰도",
                        value=f"{confidence}%",
                        delta=f"{confidence_text}"
                    )
                
                # 예측 온도 결과
                st.markdown("### 🌡️ 온도 예측값")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MS Temp 예측값", f"{predicted[0]:.1f}°C")
                with col2:
                    st.metric("RH BOWL 예측값", f"{predicted[1]:.1f}°C")
                
                # 추가 정보
                with st.expander("📋 입력된 값 보기"):
                    st.json(input_data.iloc[0].to_dict())
                    
                # 예측 결과 상세 정보
                with st.expander("📈 예측 결과 상세"):
                    st.markdown(f"""
                    | 항목 | 값 |
                    |------|-------|
                    | 경과 시간 | {elapsed_hours:.1f}시간 ({elapsed_minutes:.0f}분) |
                    | 외기온도 | {ambient_temp:.1f}°C |
                    | MS Temp 예측값 | {predicted[0]:.1f}°C |
                    | RH BOWL 예측값 | {predicted[1]:.1f}°C |
                    | 모델 | 시계열 RandomForest |
                    | 호기 | {selected_model} |
                    | RMSE | {rmse:.2f} |
                    | 신뢰도 | {confidence}% |
                    """, unsafe_allow_html=True)
                    
                # 모델 성능 정보
                with st.expander("🔍 모델 정보"):
                    st.markdown(f"""
                    **{selected_model} 시계열 모델 정보**
                    - 입력 변수: 경과 시간, 외기온도
                    - 출력 변수: MS Temp, RH BOWL
                    - 데이터 포인트: {len(model.estimators_)} 개의 트리
                    - 최대 깊이: 10
                    - RMSE: {rmse:.2f}
                    - 신뢰도: {confidence}%
                    - 데이터 특성: 계통분리 후 96시간 시계열
                    """)
                    
            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()
