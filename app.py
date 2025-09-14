import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import os
import speech_recognition as sr

# ==========================
# Load Data
# ==========================
@st.cache_data
def load_data():
    df = pd.read_excel("AI_legal_assistant_detailed_expanded_cleaned.xlsx")
    return df

df = load_data()

# ==========================
# Language Mapping
# ==========================
tts_lang_map = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te"
}

# ==========================
# Helper Functions
# ==========================
def clean_steps(text):
    """Ensure proper numbering/alignment of stepwise answers"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = []
    counter = 1
    for line in lines:
        if line[0].isdigit() and "." in line:
            # Already numbered → replace with proper counter
            line = f"{counter}. {line.split('.', 1)[1].strip()}"
        else:
            # Add numbering
            line = f"{counter}. {line}"
        cleaned.append(line)
        counter += 1
    return "\n".join(cleaned)

def get_response(user_query, language):
    lang_col = language.capitalize()
    queries = df[f"Query_{lang_col}"].fillna("").astype(str).tolist()
    answers = df[f"Detailed_Answer_{lang_col}"].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer().fit(queries)
    vecs = vectorizer.transform(queries)
    user_vec = vectorizer.transform([user_query])
    sim = cosine_similarity(user_vec, vecs)
    idx = sim.argmax()

    return queries[idx], answers[idx]

def speak_text(text, language):
    """Convert text to speech using gTTS"""
    if language.lower() not in tts_lang_map:
        st.warning(f"TTS not supported for {language}")
        return None
    tts = gTTS(text=text, lang=tts_lang_map[language.lower()])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def transcribe_audio(file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return None

# ==========================
# Streamlit App
# ==========================
st.set_page_config(page_title="AI Legal Assistant", layout="wide")

st.title("⚖️ AI Legal Assistant")
st.write("Ask your legal questions in multiple languages.")

# Language selection
language = st.selectbox("Choose Language:", ["english", "hindi", "bengali", "marathi", "tamil", "telugu"])

# Input options
st.subheader("🎙️ Ask by Voice or Keyboard")
input_mode = st.radio("Select Input Method", ["Keyboard", "Microphone"])

user_query = ""

if input_mode == "Keyboard":
    user_query = st.text_input("Enter your question:")
else:
    uploaded_audio = st.file_uploader("Upload your voice (wav format)", type=["wav"])
    if uploaded_audio:
        user_query = transcribe_audio(uploaded_audio)
        if user_query:
            st.success(f"Recognized: {user_query}")
        else:
            st.error("Could not recognize speech.")

# Get Answer
if st.button("🔍 Get Answer") and user_query:
    query, answer = get_response(user_query, language)
    
    st.markdown("### ❓ Closest Question:")
    st.write(query)

    st.markdown("### ✅ Step-by-Step Guidance:")
    st.text(clean_steps(answer))

    # Audio Answer
    audio_file = speak_text(answer, language)
    if audio_file:
        st.audio(audio_file, format="audio/mp3")



