# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 업로드
ev_file = st.file_uploader("201007_202506_전기차등록현황.xlsx", type=["xlsx"])
charger_file = st.file_uploader("202412년_지역별_전기차_충전기_구축현황(누적).xlsx", type=["xlsx"])

if ev_file and charger_file:
    ev_df = pd.read_csv(ev_file) if ev_file.name.endswith('.csv') else pd.read_excel(ev_file)
    charger_df = pd.read_csv(charger_file) if charger_file.name.endswith('.csv') else pd.read_excel(charger_file)

    # 전처리: 시군구 기준 병합
    merged = pd.merge(ev_df, charger_df, on="시군구")
    merged["충전기당차량수"] = merged["전기차등록수"] / merged["충전기수"]

    # 상관계수 출력
    corr = merged["전기차등록수"].corr(merged["충전기수"])
    st.metric("상관계수 (r)", round(corr, 3))

    # 산점도 시각화
    fig, ax = plt.subplots()
    sns.regplot(data=merged, x="충전기수", y="전기차등록수", ax=ax)
    st.pyplot(fig)
