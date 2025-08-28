import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import altair as alt

# 학습 시 사용한 피처 순서(트레이닝과 동일하게 유지해야 함)
FEATURE_COLS = [
    # 기본(폴백) 순서 — 추후 모델에서 피처 순서를 알아내면 이를 사용
    'elapsed_minutes', 'elapsed_hours', 'elapsed_days',
    'ambient_temp', 'ambient_temp_squared', 'time_temp_interaction',
    'cooling_rate', 'log_time', 'start_ms_temp', 'start_rh_bowl'
]

def _resolve_feature_order(trained_model) -> list:
    """모델에서 학습 당시 피처 순서를 추론. 실패 시 FEATURE_COLS 사용."""
    try:
        # MultiOutputRegressor 내부 첫 추정기의 Booster에서 feature_names를 가져옴
        est = getattr(trained_model, 'estimators_', [None])[0]
        if est is not None:
            booster = est.get_booster()
            names = booster.feature_names
            if isinstance(names, (list, tuple)) and len(names) > 0:
                return list(names)
    except Exception:
        pass
    return FEATURE_COLS
# 페이지 설정
st.set_page_config(page_title="터빈 냉각 온도 예측 시스템", page_icon="🌡️", layout="wide")

st.title("🌡️ 터빈 냉각 온도 예측 시스템")

# Minimal CSS for modern cards and layout
st.markdown(
    """
    <style>
    .card {padding: 1rem 1.25rem; border: 1px solid rgba(49,51,63,.2); border-radius: 12px; background: rgba(250,250,250,.04);} 
    .kpi {display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px;}
    .badge {display:inline-block; padding:4px 10px; border-radius:999px; background:#1f6feb; color:#fff; font-size:12px}
    .subtle {color: rgba(250,250,250,.65)}
    </style>
    """,
    unsafe_allow_html=True,
)

# 상단 안내
st.markdown(
    """
    모델을 선택하고 예측 조건을 입력하세요. 초기 온도를 함께 제공하면 정확도가 향상됩니다.
    """,
    help="MS/RH 초기 온도는 냉각 시작 시점의 실제 측정값입니다.",
)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


def load_performance() -> dict:
    try:
        with open('model_performance.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        # 성능 파일이 없으면 빈 딕셔너리 반환
        return {}


# 신뢰도 계산 함수
def calculate_confidence(rmse, max_value=1000):
    """RMSE를 기반으로 신뢰도를 계산합니다 (0-100%)"""
    if rmse is None or rmse <= 0:
        return 0
    confidence = max(0, min(100, (1 - rmse / max_value) * 100))
    return round(confidence, 1)


# 신뢰도 등급 결정
def get_confidence_rating(confidence):
    """신뢰도에 따른 등급을 반환합니다"""
    if confidence >= 80:
        return "🟢 높음", "green"
    elif confidence >= 60:
        return "🟡 보통", "orange"
    else:
        return "🔴 낮음", "red"


performance_data = load_performance()
with st.container():
    # 상단: 모델 선택 + 성능 배지
    top_left, top_right = st.columns([1,1])
    with top_left:
        selected_model = st.radio("모델 선택", ["1호기", "2호기"], horizontal=True)
    with top_right:
        if performance_data and selected_model in performance_data:
            avg_rmse = performance_data[selected_model].get('avg_rmse')
            if avg_rmse is not None:
                confidence = calculate_confidence(avg_rmse)
                rating, _ = get_confidence_rating(confidence)
                st.markdown(f"<span class='badge'>RMSE {avg_rmse:.2f} · 신뢰도 {confidence}% {rating}</span>", unsafe_allow_html=True)


with st.container():
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### 입력 조건")
        with st.form("predict_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                elapsed_hours = st.number_input("경과 시간 (시간)", min_value=0.0, max_value=96.0, value=48.0, step=1.0)
                start_ms_temp = st.number_input("초기 MS Temp (°C)", value=500.0, step=1.0)
            with c2:
                ambient_temp = st.number_input("외기온도 (°C)", min_value=-50.0, max_value=50.0, value=20.0, step=0.1)
                start_rh_bowl = st.number_input("초기 RH BOWL Temp (°C)", value=300.0, step=1.0)

            submitted = st.form_submit_button("🚀 예측 실행", use_container_width=True)

        st.markdown("<div class='subtle'>초기 온도를 제공하면 예측 정확도가 향상됩니다.</div>", unsafe_allow_html=True)

    with right:
        st.markdown("#### 예측 결과")
        if 'submitted' in locals() and submitted:
            model_path = 'model_1.pkl' if selected_model == '1호기' else 'model_2.pkl'
            try:
                model = load_model(model_path)
            except Exception as e:
                st.error(f"모델 로드 실패: {e}")
            else:
                elapsed_minutes = elapsed_hours * 60
                elapsed_days = elapsed_hours / 24
                input_data = {
                    'elapsed_minutes': elapsed_minutes,
                    'elapsed_hours': elapsed_hours,
                    'elapsed_days': elapsed_days,
                    'ambient_temp': ambient_temp,
                    'ambient_temp_squared': ambient_temp ** 2,
                    'time_temp_interaction': elapsed_hours * ambient_temp,
                    'cooling_rate': elapsed_hours ** 0.5,
                    'log_time': np.log1p(elapsed_hours),
                    'start_ms_temp': start_ms_temp,
                    'start_rh_bowl': start_rh_bowl,
                }
                expected_cols = _resolve_feature_order(model)
                input_df = pd.DataFrame([input_data]).reindex(columns=expected_cols)
                prediction = model.predict(input_df)[0]

                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<div class='kpi'>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("MS Temp 예측값", f"{prediction[0]:.2f}°C")
                    with c2:
                        st.metric("RH BOWL 예측값", f"{prediction[1]:.2f}°C")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # 예측 곡선(계통분리~지정 시간) 시각화
                st.markdown("#### 냉각 곡선 (예측)")
                num_points = max(2, int(np.ceil(elapsed_hours)) + 1)
                hours = np.linspace(0.0, float(elapsed_hours), num_points)

                curve_df = pd.DataFrame({
                    'elapsed_hours': hours,
                    'elapsed_minutes': hours * 60.0,
                    'elapsed_days': hours / 24.0,
                    'ambient_temp': float(ambient_temp),
                    'ambient_temp_squared': float(ambient_temp) ** 2.0,
                    'time_temp_interaction': hours * float(ambient_temp),
                    'cooling_rate': np.sqrt(hours),
                    'log_time': np.log1p(hours),
                    'start_ms_temp': float(start_ms_temp),
                    'start_rh_bowl': float(start_rh_bowl),
                })

                curve_pred = model.predict(curve_df.reindex(columns=expected_cols))
                curve_df['MS Temp 예측'] = curve_pred[:, 0]
                curve_df['RH BOWL 예측'] = curve_pred[:, 1]

                plot_df = curve_df.melt(
                    id_vars=['elapsed_hours'],
                    value_vars=['MS Temp 예측', 'RH BOWL 예측'],
                    var_name='시리즈',
                    value_name='온도(°C)'
                )

                chart = (
                    alt.Chart(plot_df)
                    .mark_line()
                    .encode(
                        x=alt.X('elapsed_hours:Q', title='경과 시간 (시간)'),
                        y=alt.Y('온도(°C):Q', title='예측 온도 (°C)'),
                        color=alt.Color('시리즈:N', title=''),
                        tooltip=['elapsed_hours', '온도(°C)', '시리즈']
                    )
                    .properties(height=280)
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)

                st.markdown("**입력 요약**")
                st.write(
                    {
                        "경과 시간(시간)": elapsed_hours,
                        "외기온도(°C)": ambient_temp,
                        "초기 MS Temp(°C)": start_ms_temp,
                        "초기 RH BOWL Temp(°C)": start_rh_bowl,
                    }
                )

                if performance_data and selected_model in performance_data:
                    avg_rmse = performance_data[selected_model].get('avg_rmse')
                    if avg_rmse is not None:
                        confidence = calculate_confidence(avg_rmse)
                        rating, _ = get_confidence_rating(confidence)
                        st.markdown(f"<span class='badge'>RMSE {avg_rmse:.2f} · 신뢰도 {confidence}% {rating}</span>", unsafe_allow_html=True)


# 메인 페이지 정보
st.markdown("---")
st.markdown("""
### 📚 시스템 설명
이 시스템은 터빈 냉각 과정의 복잡한 물리적 특성을 반영한 예측 모델을 사용합니다.

**주요 개선사항:**
- 🎯 **비선형 특성**: 외기온도의 제곱항으로 온도 변화의 비선형성 반영
- ⏰ **시간-온도 상호작용**: 경과 시간과 외기온도의 복합 효과 모델링  
- 📉 **쿨링 레이트**: 시간의 제곱근으로 쿨링 속도 변화 특성 반영
- 📊 **로그 스케일**: 초기 급격한 변화 후 점진적 안정화 특성 반영
- 🤖 **XGBoost**: 더 정교한 앙상블 모델로 복잡한 패턴 학습

**입력 변수:**
- 경과 시간 (분, 시간, 일 단위)
- 외기온도 및 비선형 효과
- 시간-온도 상호작용
- 쿨링 레이트 특성
- 로그 스케일 시간
- 초기 MS/RH BOWL 온도

**출력 변수:**
- MS Temp (메인 스팀 온도)
- RH BOWL (RH 보울 온도)
""")


