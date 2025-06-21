import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# 페이지 설정
st.set_page_config(page_title="전기차와 충전기 분석", layout="wide")

# 제목
st.title("🔌 전기차 보급률과 충전기 수의 관계 분석")

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("merged_ev_charger.csv", encoding='cp949')
    return df

df = load_data()

# 데이터 미리보기
st.subheader("📊 데이터 미리보기")
st.dataframe(df)

# 산점도 + 회귀선 시각화 (Plotly)
st.subheader("📈 Plotly 기반 회귀 시각화")
fig = px.scatter(
    df,
    x="충전기수",
    y="전기차등록대수",
    title="충전기 수 vs 전기차 등록대수",
    labels={"충전기수": "충전기 수", "전기차등록대수": "전기차 등록대수"},
    trendline="ols",  # 회귀선 추가
    template="plotly_white"
)
st.plotly_chart(fig)

# 선형 회귀 분석 (수치 출력용)
X = df[['충전기수']]
y = df['전기차등록대수']
model = LinearRegression().fit(X, y)

# 회귀계수 및 상관계수 출력
st.markdown("### 📌 분석 요약")
st.write(f"**회귀계수 (기울기)**: {model.coef_[0]:,.2f}")
st.write(f"**절편**: {model.intercept_:,.2f}")
corr = df['충전기수'].corr(df['전기차등록대수'])
st.write(f"**피어슨 상관계수**: {corr:.3f}")

#상관계수가 높다 (r ≥ 0.7) → 강한 선형 관계

#회귀모델의 기울기가 양(+)이다 → 충전기 수가 늘수록 등록대수가 증가함

#(선택) R² 값이 높다면 설명력도 확보됨 → 추후 추가 가능

st.caption("※ 데이터는 전국 합계 기준입니다.")
