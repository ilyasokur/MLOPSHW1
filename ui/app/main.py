import os, time
import pandas as pd
import psycopg2
import streamlit as st
import matplotlib.pyplot as plt

PG_DSN = os.getenv("PG_DSN", "postgresql://fraud:fraud@postgres:5432/fraud")

@st.cache_resource(show_spinner=False)
def get_conn():
    return psycopg2.connect(PG_DSN)

def fetch_frauds(limit=10):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""
            SELECT transaction_id, score, fraud_flag, ts
            FROM scores
            WHERE fraud_flag=1
            ORDER BY ts DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["transaction_id","score","fraud_flag","ts"])

def fetch_last_scores(limit=100):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""
            SELECT score, ts
            FROM scores
            ORDER BY ts DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["score","ts"])

st.set_page_config(page_title="Fraud Scoring", layout="centered")
st.title("Fraud Scoring — результаты")

c1, c2 = st.columns([1,1])
with c1:
    if st.button("Посмотреть результаты", type="primary", use_container_width=True):
        st.session_state["show"] = True
with c2:
    auto = st.toggle("Автообновление (5 сек)", value=False)

if st.session_state.get("show"):
    st.subheader("Последние 10 транзакций с fraud_flag == 1")
    df = fetch_frauds(10)
    if df.empty:
        st.info("Записей с fraud_flag=1 пока нет.")
    else:
        st.dataframe(df, use_container_width=True)

    st.subheader("Распределение скоров последних 100 транзакций")
    ds = fetch_last_scores(100)
    if ds.empty:
        st.info("Скорингов пока нет.")
    else:
        fig, ax = plt.subplots()
        ax.hist(ds["score"].astype(float), bins=30)
        ax.set_title("Score distribution (last 100)")
        ax.set_xlabel("score"); ax.set_ylabel("count")
        st.pyplot(fig, clear_figure=True)

    if auto:
        time.sleep(5)
        st.rerun()
else:
    st.info("Нажмите «Посмотреть результаты».")