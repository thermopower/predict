import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import altair as alt
import os
from typing import Optional

# 학습 시 사용한 피처 순서(트레이닝과 동일하게 유지해야 함)
FEATURE_COLS = [
    # 기본(폴백) 순서 — 학습 시 사용한 순서에 맞춤
    'start_ms_temp', 'start_rh_bowl', 'elapsed_hours', 'ambient_temp',
    'ambient_temp_squared', 'start_temp_ambient_interaction', 'time_temp_interaction', 'sqrt_elapsed_hours', 'log_time'
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

# RH BOWL 지수 스무딩 알파(외부 파일 없이 코드에 내장)
RH_BOWL_SMOOTHING_ALPHA: float = 0.75

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
    /* Hide number input spinners */
    input[type=number]::-webkit-outer-spin-button,
    input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
    input[type=number] { -moz-appearance: textfield; }
    /* Hide Streamlit number_input +/- stepper buttons */
    .stNumberInput button { display: none !important; }
    .stNumberInput svg { display: none !important; }
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


def exponential_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """지수 스무딩. alpha∈(0,1)일 때만 적용, 그 외는 원본 반환."""
    try:
        a = float(alpha)
    except Exception:
        return values
    if not (0.0 < a < 1.0):
        return values
    if values is None or len(values) == 0:
        return values
    smoothed = values.astype(float).copy()
    for i in range(1, len(smoothed)):
        smoothed[i] = a * smoothed[i] + (1.0 - a) * smoothed[i - 1]
    return smoothed

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
                elapsed_hours_str = st.number_input("경과 시간 (시간)", value=48.0, min_value=0.0, max_value=96.0, step=1.0, format="%.2f", key="elapsed_hours")
                start_ms_temp_str = st.number_input("초기 MS Temp (°C)", value=500.0, step=1.0, format="%.2f", key="start_ms_temp")
            with c2:
                ambient_temp_str = st.number_input("외기온도 (°C)", value=20.0, min_value=-50.0, max_value=50.0, step=0.1, format="%.2f", key="ambient_temp")
                start_rh_bowl_str = st.number_input("초기 RH BOWL Temp (°C)", value=300.0, step=1.0, format="%.2f", key="start_rh_bowl")

            submitted = st.form_submit_button("🚀 예측 실행", use_container_width=True)

        st.markdown("<div class='subtle'>초기 온도를 제공하면 예측 정확도가 향상됩니다.</div>", unsafe_allow_html=True)

    with right:
        st.markdown("#### 예측 결과")
        if 'submitted' in locals() and submitted:
            # 입력값 파싱 및 검증
            # number_input 사용으로 이미 float 보장
            elapsed_hours_val = float(elapsed_hours_str)
            start_ms_temp_val = float(start_ms_temp_str)
            ambient_temp_val = float(ambient_temp_str)
            start_rh_bowl_val = float(start_rh_bowl_str)

            errors: list[str] = []
            if not (0.0 <= elapsed_hours_val <= 96.0):
                errors.append("경과 시간은 0~96 사이의 숫자를 입력하세요.")
            if not (-50.0 <= ambient_temp_val <= 50.0):
                errors.append("외기온도는 -50~50 사이의 숫자를 입력하세요.")

            if errors:
                for msg in errors:
                    st.error(msg)
                st.stop()
            model_path = 'model_1.pkl' if selected_model == '1호기' else 'model_2.pkl'
            try:
                model = load_model(model_path)
            except Exception as e:
                st.error(f"모델 로드 실패: {e}")
            else:
                input_data = {
                    'elapsed_hours': elapsed_hours_val,
                    'ambient_temp': ambient_temp_val,
                    'ambient_temp_squared': ambient_temp_val ** 2,
                    'start_temp_ambient_interaction': start_ms_temp_val * ambient_temp_val,
                    'time_temp_interaction': elapsed_hours_val * ambient_temp_val,
                    'sqrt_elapsed_hours': elapsed_hours_val ** 0.5,
                    'log_time': np.log1p(elapsed_hours_val),
                    'start_ms_temp': start_ms_temp_val,
                    'start_rh_bowl': start_rh_bowl_val,
                }
                # (롤백) 참고 피처 생성 없음
                expected_cols = _resolve_feature_order(model)
                input_df = pd.DataFrame([input_data]).reindex(columns=expected_cols)
                prediction = model.predict(input_df)[0]

                with st.container():
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("MS Temp 예측값", f"{prediction[0]:.2f}°C")
                    with c2:
                        st.metric("RH BOWL 예측값", f"{prediction[1]:.2f}°C")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # 예측 곡선(계통분리~지정 시간) 시각화
                st.markdown("#### 냉각 곡선 (예측)")
                num_points = max(2, int(np.ceil(elapsed_hours_val)) + 1)
                hours = np.linspace(0.0, float(elapsed_hours_val), num_points)

                curve_df = pd.DataFrame({
                    'elapsed_hours': hours,
                    'ambient_temp': float(ambient_temp_val),
                    'ambient_temp_squared': float(ambient_temp_val) ** 2.0,
                    'start_temp_ambient_interaction': float(start_ms_temp_val) * float(ambient_temp_val),
                    'time_temp_interaction': hours * float(ambient_temp_val),
                    'sqrt_elapsed_hours': np.sqrt(hours),
                    'log_time': np.log1p(hours),
                    'start_ms_temp': float(start_ms_temp_val),
                    'start_rh_bowl': float(start_rh_bowl_val),
                })
                # (롤백) 참고 피처 생성 없음

                curve_pred = model.predict(curve_df.reindex(columns=expected_cols))

                # 단조 감소(비증가) 보정 전: 예측 추출
                ms = curve_pred[:, 0].astype(float)
                rh = curve_pred[:, 1].astype(float)

                # RH BOWL 지수 스무딩 적용 (코드 내 상수)
                rh = exponential_smooth(rh, RH_BOWL_SMOOTHING_ALPHA)

                # 시작 시점(0시간) 예측값을 입력 초기 온도로 앵커링
                if len(ms) > 0:
                    ms[0] = float(start_ms_temp_val)
                    rh[0] = float(start_rh_bowl_val)

                # 시간이 지날수록 온도가 올라가지 않도록 누적 최소값 적용
                ms_mono = np.minimum.accumulate(ms)
                rh_mono = np.minimum.accumulate(rh)

                curve_df['MS Temp 예측'] = ms_mono
                curve_df['RH BOWL 예측'] = rh_mono

                plot_df = curve_df.melt(
                    id_vars=['elapsed_hours'],
                    value_vars=['MS Temp 예측', 'RH BOWL 예측'],
                    var_name='시리즈',
                    value_name='온도(°C)'
                )

                x_zoom = alt.selection_interval(bind='scales', encodings=['x'], translate=False)
                chart = (
                    alt.Chart(plot_df)
                    .mark_line()
                    .encode(
                        x=alt.X('elapsed_hours:Q', scale=alt.Scale(domain=[0, float(elapsed_hours_val)]), axis=alt.Axis(title='경과 시간 (시간)', titlePadding=28, labelPadding=8)),
                        y=alt.Y('온도(°C):Q', title='예측 온도 (°C)'),
                        color=alt.Color('시리즈:N', title=''),
                        tooltip=['elapsed_hours', '온도(°C)', '시리즈']
                    )
                    .properties(height=340)
                    .add_selection(x_zoom)
                    .configure_axisX(titlePadding=28, labelPadding=8)
                )
                st.altair_chart(chart, use_container_width=True)

                # 예측 결과 섹션에서는 신뢰도 배지를 표시하지 않습니다 (요청사항 반영)


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

**입력 변수:**
- 경과 시간 (시간 단위)
- 초기 MS/RH BOWL 온도

**출력 변수:**
- MS Temp (메인 스팀 온도)
- RH BOWL (RH 보울 온도)
""")

st.markdown("---")
st.markdown("#### 🔎 특성 중요도 (학습 결과)")

# 중요도 로드 및 시각화
def _load_importance(unit: str) -> Optional[pd.DataFrame]:
    path = f"feature_importance_{unit}.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        avg = payload.get('average', {})
        if not avg:
            return None
        df = pd.DataFrame([
            {'feature': k, 'importance': float(v)} for k, v in avg.items()
        ])
        # 중요도 정렬 상위만 노출 (상위 12개)
        df = df.sort_values('importance', ascending=False).head(12)
        return df
    except Exception:
        return None

unit_for_imp = selected_model if 'selected_model' in locals() else '1호기'
imp_df = _load_importance(unit_for_imp)
if imp_df is None:
    st.info("학습 단계에서 중요도 파일을 찾을 수 없습니다. 학습을 먼저 실행해 주세요.")
else:
    chart = (
        alt.Chart(imp_df)
        .mark_bar()
        .encode(
            x=alt.X('importance:Q', title='중요도 (gain 평균)'),
            y=alt.Y('feature:N', sort='-x', title='피처'),
            tooltip=['feature', 'importance']
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


