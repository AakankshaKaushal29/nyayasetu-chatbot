import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("AI legal assistant - data.xlsx")
    return df

df = load_data()

# Build vectorizer on Hindi queries
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["query_hindi"])

st.title("NyayaSetu - Legal Assistance Chatbot (Demo)")
st.write("Type your legal question in Hindi:")

# User input
user_query = st.text_input("👉 आपका सवाल (in Hindi):")

if user_query:
    # Convert user query into vector
    query_vec = vectorizer.transform([user_query])
    # Compute similarity
    similarity = cosine_similarity(query_vec, X).flatten()
    # Find best match
    idx = similarity.argmax()
    best_match = df.iloc[idx]

    st.subheader("📌 Result")
    st.write(f"**Matched Category:** {best_match['category']}")
    st.write(f"**Similar Question:** {best_match['query_hindi']}")
    st.write(f"**Answer (Hindi):** {best_match['answer_hindi']}")

    if st.checkbox("Show English Answer"):
        st.write(f"**Answer (English):** {best_match['answer_english']}")
