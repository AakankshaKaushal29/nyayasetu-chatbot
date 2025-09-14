import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# Load dataset
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("AI_legal_assistant_translated_full.xlsx")
    return df

df = load_data()

# ------------------------
# UI
# ------------------------
st.title("⚖️ NyayaSetu - Multilingual Legal Assistance Chatbot (Demo)")
st.write("Ask your legal question in your preferred language:")

# Language options mapping to column names
languages = {
    "Hindi": "hindi",
    "English": "english",
    "Bengali": "bengali",
    "Telugu": "telugu",
    "Marathi": "marathi",
    "Tamil": "tamil"
}

# Dropdown for language selection
selected_lang = st.selectbox("🌍 Choose language", list(languages.keys()))

# Get column names for query & answer
query_col = f"query_{languages[selected_lang]}"
answer_col = f"answer_{languages[selected_lang]}"

# ------------------------
# Build search index (based on selected language)
# ------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df[query_col].astype(str))

# ------------------------
# User input
# ------------------------
user_query = st.text_input(f"👉 Your Question ({selected_lang}):")

if user_query:
    # Convert user query into vector
    query_vec = vectorizer.transform([user_query])
    # Compute similarity
    similarity = cosine_similarity(query_vec, X).flatten()
    # Find best match
    idx = similarity.argmax()
    best_match = df.iloc[idx]

    # ------------------------
    # Show result
    # ------------------------
    st.subheader("📌 Result")
    st.write(f"**Matched Category:** {best_match['category']}")
    st.write(f"**Similar Question ({selected_lang}):** {best_match[query_col]}")
    st.write(f"**Answer ({selected_lang}):**\n\n{best_match[answer_col]}")

    # Optional: Show answers in other languages
    with st.expander("🌐 Show answers in other languages"):
        for lang, code in languages.items():
            if lang != selected_lang:
                st.write(f"**{lang}:**\n{best_match[f'answer_{code}']}")
