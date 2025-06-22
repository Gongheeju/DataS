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
    
    # 충전기수 열 생성
    df["충전기수"] = df["완속 충전기수"] + df["급속 충전기수"]
    
    # '년월'을 날짜 형식으로 변환, 실패 시 NaT 처리
    df["년월"] = pd.to_datetime(df["년월"], errors="coerce")
    
    # NaT 값 제거
    df = df.dropna(subset=["년월"])
    
    return df

# 데이터 로드
df = load_data()

# 데이터 미리보기
st.subheader("📊 데이터 미리보기")
st.dataframe(df)

# 산점도 + 회귀선 시각화
st.subheader("📈 회귀 기반 산점도 시각화")
fig = px.scatter(
    df,
    x="충전기수",
    y="전기차등록대수",
    title="충전기 수 vs 전기차 등록대수",
    labels={"충전기수": "충전기 수", "전기차등록대수": "전기차 등록대수"},
    trendline="ols",
    template="plotly_white"
)
st.plotly_chart(fig)

# 선형 회귀 분석
X = df[['충전기수']]
y = df['전기차등록대수']
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)

# 분석 요약 출력
st.markdown("### 📌 분석 요약")
st.write(f"**회귀계수 (기울기)**: {model.coef_[0]:,.2f}")
st.write(f"**절편**: {model.intercept_:,.2f}")
st.write(f"**R² (설명력)**: {r_squared:.3f}")
corr = df['충전기수'].corr(df['전기차등록대수'])
st.write(f"**피어슨 상관계수**: {corr:.3f}")

st.markdown("""
#### 🔍 해석 기준
- **상관계수가 높다 (r ≥ 0.7)** → 강한 선형 관계
- **기울기 양수** → 충전기 수 증가 시 전기차 등록도 증가
- **R² 높음** → 충전기 수만으로 전기차 등록수 상당히 설명 가능

#### 📊 시계열 기반 인과관계 해석
- 시계열 그래프에서 충전기 수 증가가 전기차 등록 증가보다 **선행되는 경향**이 나타납니다.
- 이는 충전 인프라 확충이 전기차 보급을 **유도했을 가능성**을 시사합니다.
- 정밀한 인과 분석에는 추가 통계 검정이 필요하지만, **시계열 추세만으로도 유의미한 관계**가 보입니다.
""")

# 시계열 시각화
st.subheader("📊 시계열 추세 비교")
fig_time = px.line(
    df.sort_values("년월"),
    x="년월",
    y=["완속 충전기수", "급속 충전기수", "전기차등록대수"],
    title="시계열: 완속/급속 충전기 수 및 전기차 등록대수 추세",
    labels={"value": "수량", "variable": "항목"},
    markers=True,
    template="plotly_white"
)
st.plotly_chart(fig_time)

st.caption("※ 데이터는 전국 기준 누적 수치를 기반으로 분석되었습니다.")
