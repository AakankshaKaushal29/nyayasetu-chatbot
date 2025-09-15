import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import os
import base64
import speech_recognition as sr

# --------------------------
# Load dataset
# --------------------------
@st.cache_data
def load_data():
    return pd.read_excel("AI_legal_assistant_detailed_expanded_cleaned.xlsx")

df = load_data()

# --------------------------
# Language column mapping
# --------------------------
LANG_MAP = {
    "english": {
        "query": "query_english",
        "answer": "answer_english",
        "detailed": "detailed_answer_english"
    },
    "hindi": {
        "query": "query_hindi",
        "answer": "answer_hindi",
        "detailed": "detailed_answer_hindi"
    },
    "bengali": {
        "query": "query_bengali",
        "answer": "answer_bengali",
        "detailed": "detailed_answer_bengali"
    },
    "telugu": {
        "query": "query_telugu",
        "answer": "answer_telugu",
        "detailed": "detailed_answer_telugu"
    },
    "marathi": {
        "query": "query_marathi",
        "answer": "answer_marathi",
        "detailed": "detailed_answer_marathi"
    },
    "tamil": {
        "query": "query_tamil",
        "answer": "answer_tamil",
        "detailed": "detailed_answer_tamil"
    }
}

# gTTS language codes
TTS_LANG = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "telugu": "te",
    "marathi": "mr",
    "tamil": "ta"
}

# --------------------------
# Query → Answer
# --------------------------
def get_response(user_query, language):
    lang = language.lower()
    if lang not in LANG_MAP:
        return "", "⚠️ Language not supported."

    query_col = LANG_MAP[lang]["query"]
    detailed_col = LANG_MAP[lang]["detailed"]

    if query_col not in df.columns or detailed_col not in df.columns:
        return "", "⚠️ Columns missing in dataset."

    queries = df[query_col].fillna("").astype(str).tolist()
    answers = df[detailed_col].fillna("").astype(str).tolist()

    if not queries:
        return "", "⚠️ No data available."

    vectorizer = TfidfVectorizer().fit(queries)
    vecs = vectorizer.transform(queries)
    user_vec = vectorizer.transform([user_query])
    sim = cosine_similarity(user_vec, vecs)
    idx = sim.argmax()

    return queries[idx], answers[idx]

# --------------------------
# Clean numbered steps
# --------------------------
def clean_steps(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = []
    num = 1
    for line in lines:
        if line[0].isdigit() and "." in line[:3]:
            line = line.split(".", 1)[-1].strip()
        cleaned.append(f"{num}. {line}")
        num += 1
    return "\n".join(cleaned)

# --------------------------
# Speak text (TTS)
# --------------------------
def speak_text(text, language):
    lang_code = TTS_LANG.get(language.lower(), "en")
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            audio_file = tmp.name
        with open(audio_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        audio_html = f"""
        <audio autoplay controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        os.remove(audio_file)
    except Exception as e:
        st.error(f"TTS error: {e}")

# --------------------------
# Nyayasetu App UI
# --------------------------
st.set_page_config(page_title="⚖️ Nyayasetu - AI Legal Assistant", layout="wide")

st.title("⚖️ Nyayasetu - AI Legal Assistant")
st.markdown("Ask your legal questions in multiple languages.")

language = st.selectbox("🌐 Choose Language:", list(LANG_MAP.keys()))

input_method = st.radio("🎙️ Ask by Voice or Keyboard", ["Keyboard", "Microphone"])

user_query = ""
if input_method == "Keyboard":
    user_query = st.text_input("Enter your question:")
else:
    st.info("🎤 Speak now...")
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source, phrase_time_limit=5)
        try:
            user_query = recognizer.recognize_google(audio, language="en-IN")
            st.success(f"Recognized: {user_query}")
        except Exception as e:
            st.error(f"Voice recognition failed: {e}")

if user_query:
    query, answer = get_response(user_query, language)
    if answer:
        st.subheader("✅ Step-by-Step Guidance:")
        st.markdown(clean_steps(answer))

        if st.button("🔊 Hear the Answer"):
            speak_text(answer, language)

# --------------------------
# Feedback Section
# --------------------------
st.markdown("---")
st.subheader("🙋 Feedback")
feedback = st.radio("Was this answer helpful?", ["👍 Yes", "👎 No"], horizontal=True)
comment = st.text_area("Additional comments (optional):")
if st.button("Submit Feedback"):
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Lang: {language} | Query: {user_query} | Feedback: {feedback} | Comment: {comment}\n")
    st.success("✅ Thank you for your feedback!")

