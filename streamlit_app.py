import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 페이지 설정
st.set_page_config(page_title="전기차와 충전기 분석", layout="wide")

# 제목
st.title("🔌 전기차 보급률과 충전기 수의 관계 분석")

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("merged_ev_charger.csv", encoding='cp949')  # 같은 폴더에 있어야 함
    return df

df = load_data()

# 데이터 미리보기
st.subheader("📊 데이터 미리보기")
st.dataframe(df)

# 산점도 시각화
st.subheader("🔍 충전기 수 vs 전기차 등록대수 (산점도)")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x='충전기수', y='전기차등록대수', ax=ax1)
plt.xlabel("충전기 수")
plt.ylabel("전기차 등록대수")
st.pyplot(fig1)

# 회귀선 시각화
st.subheader("📈 선형 회귀 분석")
X = df[['충전기수']]
y = df['전기차등록대수']
model = LinearRegression().fit(X, y)
pred = model.predict(X)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.regplot(x='충전기수', y='전기차등록대수', data=df, ax=ax2, line_kws={"color": "red"})
plt.xlabel("충전기 수")
plt.ylabel("전기차 등록대수")
st.pyplot(fig2)

# 회귀계수 및 상관계수 출력
st.markdown("### 📌 분석 요약")
st.write(f"**회귀계수 (기울기)**: {model.coef_[0]:,.2f}")
st.write(f"**절편**: {model.intercept_:,.2f}")
corr = df['충전기수'].corr(df['전기차등록대수'])
st.write(f"**피어슨 상관계수**: {corr:.3f}")

st.caption("※ 데이터는 전국 합계 기준입니다.")
