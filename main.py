import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 데이터셋 (AI 학습용)
data = {
    "학생수": [
        30, 28, 35, 22, 25, 31, 34, 20, 27, 26, 32, 29, 21, 33, 23, 24, 26, 30,
        28, 25, 22, 27, 29, 31, 20, 21, 32, 34, 26, 25
    ],
    "수업시간": [
        45, 50, 60, 30, 40, 55, 50, 30, 35, 40, 60, 50, 30, 55, 35, 40, 45, 60,
        50, 40, 35, 45, 50, 60, 30, 40, 55, 60, 45, 35
    ],
    "창문": [
        "닫힘", "닫힘", "닫힘", "열림", "열림", "닫힘", "닫힘", "열림", "닫힘", "열림", "닫힘", "닫힘",
        "열림", "닫힘", "열림", "열림", "닫힘", "닫힘", "닫힘", "열림", "열림", "닫힘", "닫힘", "닫힘",
        "열림", "열림", "닫힘", "닫힘", "닫힘", "열림"
    ],
    "냉난방": [
        "냉방", "난방", "냉방", "난방", "난방", "냉방", "냉방", "난방", "난방", "냉방", "냉방", "난방",
        "난방", "냉방", "난방", "난방", "냉방", "냉방", "난방", "냉방", "난방", "냉방", "냉방", "난방",
        "난방", "난방", "냉방", "냉방", "냉방", "난방"
    ],
    "외부온도": [
        32, 20, 33, 18, 15, 30, 29, 13, 25, 31, 34, 21, 14, 32, 17, 16, 33, 30,
        19, 31, 18, 26, 22, 30, 15, 17, 32, 33, 28, 19
    ],
    "환기여부": [
        "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "Yes", "No", "No",
        "Yes", "No", "No", "Yes", "Yes", "No", "No", "Yes", "No", "Yes", "No",
        "Yes", "No", "Yes", "Yes", "No", "No", "No", "Yes"
    ]
}
df = pd.DataFrame(data)

st.title("스마트 교실 환기 판단 시스템")

with st.form("input_form"):
    students = st.slider("학생 수", 20, 35, 30)
    duration = st.slider("수업 시간 (분)", 30, 60, 45, step=5)
    window = st.selectbox("창문 상태", ["열림", "닫힘"])
    mode = st.radio("냉난방 모드", ["냉방", "난방", "없음"])
    temperature = st.slider("외부 온도 (℃)", -10, 40, 22)
    submitted = st.form_submit_button("예측하기")

if submitted:
    st.markdown("---")

    # 수치 기반 계산
    co2_rate = 0.005  # 1인당 분당 CO2 (L/sec)
    co2_total = co2_rate * students * duration * 60
    heat_loss = abs(temperature - 22) if mode != "없음" else 0
    loss_factor = 1.0 if window == "열림" else 0.3
    heat_loss *= loss_factor
    carbon_emission = co2_total * 0.000001 * 1.96  # kgCO2 환산

    needs_vent_by_co2 = co2_total > 9000
    needs_vent_by_loss = (mode != "없음" and heat_loss > 5)

    st.subheader("① 수치 기반 판단")

    # ✅ 수치 판단 결과 강조 박스 (제일 위에)
    if needs_vent_by_co2:
        st.success("→ CO₂ 기준 초과: 환기 필요")
    else:
        st.info("→ CO₂ 기준 미만: 환기 불필요")
    st.write(f"- 누적 탄소배출량: **{round(carbon_emission, 3)} kgCO₂**")
    with st.expander("탄소배출량이란?"):
        st.markdown(
            "CO₂ 총 배출량을 질량 단위로 환산한 값입니다. 공기 중 농도뿐 아니라 환경에 미치는 영향을 정량적으로 파악할 수 있도록 도와줍니다."
        )
        st.markdown("예: **1kgCO₂는 에어컨 약 30분 사용 시 발생하는 양과 비슷합니다.**")

    st.write(f"- 총 CO₂ 배출량: **{int(co2_total):,}L** (기준: 9000L)")
    with st.expander("총 CO₂ 배출량이란?"):
        st.markdown(
            "수업 시간 동안 학생들이 숨 쉬며 실내에 배출하는 이산화탄소의 양을 리터 단위로 나타낸 값입니다. 값이 높을수록 공기 질이 나빠질 가능성이 커지고, 환기가 필요해집니다."
        )
        st.markdown("※ 9000L은 일반적인 교실 환기 권장 기준선입니다.")
        st.markdown("예: **학생 30명이 1시간 수업 시 약 1만 L 이상 배출될 수 있습니다.**")

    st.write(f"- 열손실 지수: **{round(heat_loss, 2)}** (기준: 5 이상)")
    with st.expander("열손실 지수란?"):
        st.markdown(
            "냉난방 중 외부로 빠져나가는 열의 크기를 수치화한 지표입니다. 값이 클수록 에너지 낭비가 크며, 환기를 꺼리는 요인이 될 수 있습니다."
        )
        st.markdown("예: **열손실 지수가 8 이상이면 창문을 열 경우 에어컨 효과가 크게 줄어들 수 있습니다.**")

    # AI 판단
    X = df[["학생수", "수업시간", "창문", "냉난방", "외부온도"]]
    X_encoded = pd.get_dummies(X)
    y = df["환기여부"]
    model = DecisionTreeClassifier()
    model.fit(X_encoded, y)

    input_df = pd.DataFrame([{
        "학생수": students,
        "수업시간": duration,
        "창문": window,
        "냉난방": mode,
        "외부온도": temperature
    }])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X_encoded.columns,
                                          fill_value=0)

    ai_prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

    st.subheader("② AI 판단 결과")
    if ai_prediction == "Yes":
        st.success("→ AI 판단: 환기 필요")
    else:
        st.info("→ AI 판단: 환기 불필요")
    st.caption(f"(환기 필요 확률: {round(proba[1]*100,1)}%)")

    st.markdown("**AI 추정 이유:**")
    reasons = []
    if window == "닫힘":
        reasons.append("창문이 닫혀 있어 공기 정체 가능성")
    if duration >= 50:
        reasons.append("수업 시간이 길어 CO₂ 누적 가능성 있음")
    if mode == "냉방" and temperature >= 28:
        reasons.append("냉방 중 외부 고온 → 열손실 우려")
    if mode == "난방" and temperature <= 15:
        reasons.append("난방 중 외부 저온 → 에너지 낭비 가능성")
    if not reasons:
        reasons.append("입력 조건이 비교적 안정적임")
    for r in reasons:
        st.markdown(f"- {r}")

    # 판단 비교
    st.markdown("---")
    st.subheader("③ 수치 vs AI 판단 비교")
    score_co2 = "환기 필요" if needs_vent_by_co2 else "불필요"
    score_ai = "환기 필요" if ai_prediction == "Yes" else "불필요"
    st.write(f"- 수치 판단: {score_co2}")
    st.write(f"- AI 판단: {score_ai}")

    if score_ai == score_co2:
        st.success("→ 두 판단이 일치합니다. 신뢰도 높은 결과입니다.")
    else:
        st.warning("→ 판단 불일치: 상황을 고려하여 적절히 조절하세요.")

    st.markdown("**④ 추천 행동:**")
    if score_ai == "환기 필요" or score_co2 == "환기 필요":
        st.write("지금은 환기가 권장됩니다. 수업 종료 전이라면 창문 일부 개방으로 시작해보세요.")
    else:
        st.write("지금은 환기를 급히 할 필요는 없습니다. 다음 수업 전 짧게 환기하면 충분합니다.")
