import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import os

# For voice input (optional mic recorder)
try:
    from streamlit_mic_recorder import mic_recorder
    mic_available = True
except:
    mic_available = False

# ------------------------
# Load dataset
# ------------------------
@st.cache_data
def load_data():
    return pd.read_excel("SIH_Dataset_Final.xlsx")

df = load_data()

# ------------------------
# Search Function
# ------------------------
def search_query(user_query, language="English"):
    lang_df = df[df["Language"] == language]

    if lang_df.empty:
        return None, None, None

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(lang_df["Query"].astype(str))
    user_vec = vectorizer.transform([user_query])

    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx = similarity.argmax()

    best_match = lang_df.iloc[best_idx]
    return best_match["Query"], best_match["Short Answer"], best_match["Detailed Answer"]

# ------------------------
# Text-to-Speech
# ------------------------
def speak_text(text, lang_code="en"):
    tts = gTTS(text=text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts.save(tmpfile.name)
        return tmpfile.name

# ------------------------
# UI Layout
# ------------------------
st.set_page_config(page_title="Nyayasetu - SIH Demo", layout="wide")

st.title("⚖️ Nyayasetu")
st.markdown(
    """
    **Nyayasetu** is an AI-based legal assistant that helps citizens get quick,  
    multi-language legal guidance in simple steps.  
    Ask your question and get structured answers based on Indian laws.
    """
)

# Language Selection
languages = df["Language"].unique().tolist()
selected_lang = st.selectbox("🌐 Select Language", languages, index=0)

# Sample Questions
with st.expander("💡 Try sample questions"):
    st.write(
        """
        - How can I file for divorce?  
        - What to do in case of cyber fraud?  
        - How to file an FIR?  
        """
    )

# Input
st.subheader("🔎 Ask Your Question")
user_query = st.text_input("Type your query here...", "")

# Optional Mic Input
if mic_available:
    st.write("🎤 Or use your voice:")
    audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop", key="recorder")
    if audio:
        st.audio(audio["bytes"])

# Process Query
if st.button("Get Answer"):
    if user_query.strip():
        q, short_ans, detailed_ans = search_query(user_query, selected_lang)

        if q:
            st.success(f"✅ Closest Match: **{q}**")

            st.write("### 📝 Short Answer")
            st.info(short_ans)

            st.write("### 📖 Detailed Answer")
            st.markdown(detailed_ans)

            # Text to Speech option
            if st.checkbox("🔊 Listen to Answer"):
                lang_map = {
                    "English": "en",
                    "Hindi": "hi",
                    "Marathi": "mr",
                    "Bengali": "bn",
                    "Tamil": "ta",
                    "Telugu": "te"
                }
                lang_code = lang_map.get(selected_lang, "en")
                audio_file = speak_text(detailed_ans, lang_code)
                st.audio(audio_file, format="audio/mp3")
        else:
            st.error("❌ Sorry, I couldn’t find an answer. Please try rephrasing.")
    else:
        st.warning("⚠️ Please enter a question.")

# Feedback Section
st.markdown("---")
st.subheader("💬 Feedback")
feedback = st.text_area("Tell us how we can improve:")
if st.button("Submit Feedback"):
    st.success("✅ Thank you for your feedback!")

