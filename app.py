import streamlit as st
import pandas as pd
from predict import train_and_evaluate

# 페이지 설정 (반드시 가장 먼저 호출되어야 함)
st.set_page_config(page_title="온도 예측 시스템", layout="wide")

def main():
    # 제목
    st.title("호기별 온도 예측 시스템")
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
        - AMBIENT_TEMP_D: 외기 온도(계통분리시점)
        - MS_TEMP_D: MS 온도(계통분리시점)
        - RH_BOWL_D: RH 보울 온도(계통분리시점)
        - TARGET_T_MIN: 소요 시간(계통분리부터 점화까지, 분 단위)
        - AMBIENT_TEMP_T: 외기 온도(점화 시점)
        """)

    # 입력 폼
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            ambient_temp_d = st.number_input("AMBIENT_TEMP_D", value=26.0, step=0.1)
            ms_temp_d = st.number_input("MS_TEMP_D", value=538.0, step=0.1)
            rh_bowl_d = st.number_input("RH_BOWL_D", value=612.0, step=0.1)
        
        with col2:
            target_t_min = st.number_input("TARGET_T_MIN (분)", value=90, min_value=1, step=1)
            ambient_temp_t = st.number_input("AMBIENT_TEMP_T", value=25.2, step=0.1)
        
        submitted = st.form_submit_button("예측 실행")

    # 모델 로드 및 예측
    if 'submitted' in locals() and submitted:
        with st.spinner(f'{selected_model} 모델을 불러오는 중...'):
            try:
                # 선택된 호기 모델 로드
                model = train_and_evaluate(selected_model)
                
                # 입력 데이터 준비
                input_data = pd.DataFrame([{
                    'AMBIENT_TEMP_D': ambient_temp_d,
                    'MS_TEMP_D': ms_temp_d,
                    'RH_BOWL_D': rh_bowl_d,
                    'TARGET_T_MIN': target_t_min,
                    'AMBIENT_TEMP_T': ambient_temp_t
                }])
                
                # 예측 실행
                predicted = model.predict(input_data)[0]
                
                # 결과 표시
                st.success(f"{selected_model} 예측이 완료되었습니다!")
                st.markdown("### 📊 예측 결과")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RH_BOWL_T 예측값", f"{predicted[0]:.2f}°C")
                with col2:
                    st.metric("MS_TEMP_T 예측값", f"{predicted[1]:.2f}°C")
                
                # 추가 정보
                with st.expander("📋 입력된 값 보기"):
                    st.json(input_data.iloc[0].to_dict())
                    
                # 예측 결과 상세 정보
                with st.expander("📈 예측 결과 상세"):
                    st.markdown(f"""
                    | 항목 | 예측값 |
                    |------|-------|
                    | RH_BOWL_T | {predicted[0]:.2f}°C |
                    | MS_TEMP_T | {predicted[1]:.2f}°C |
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()
