import streamlit as st
import pandas as pd
import os

st.title("📊 NyayaSetu Analytics Dashboard")

feedback_file = "feedback.csv"

if not os.path.exists(feedback_file):
    st.warning("No feedback data yet. Interact with the chatbot first.")
else:
    df = pd.read_csv(feedback_file)

    st.subheader("✅ Summary")
    st.write(f"Total Queries: {len(df)}")
    st.write(f"Positive Feedback: {(df['feedback'] == 'positive').sum()}")
    st.write(f"Negative Feedback: {(df['feedback'] == 'negative').sum()}")

    st.subheader("📌 Feedback Table")
    st.dataframe(df)

    st.subheader("📈 Feedback Breakdown")
    st.bar_chart(df['feedback'].value_counts())

    st.subheader("🔥 Most Asked Queries")
    st.table(df['query'].value_counts().head(5))
