import streamlit as st
import pandas as pd
from googletrans import Translator
from bs4 import BeautifulSoup
import requests
from googlesearch import search
from gtts import gTTS
import os
import tempfile
import speech_recognition as sr

# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(page_title="Nyayasetu - AI Legal Consultant", layout="centered")
st.title("⚖️ Nyayasetu - AI Legal Consultant")
st.markdown(
    """
Nyayasetu is an AI-based legal assistant that helps citizens get quick, multi-language legal guidance in simple steps.  
Ask your question and get structured answers based on Indian laws.
"""
)
st.write("---")

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_excel("SIH_Dataset_Final.xlsx")

df = load_data()

# ----------------------------
# Example questions
# ----------------------------
st.subheader("Try one of these example questions:")
examples = df["Query_English"].tolist()[:5]  # first 5 examples
for ex in examples:
    if st.button(ex):
        st.session_state['user_query'] = ex

# ----------------------------
# Language Selection
# ----------------------------
languages = df.columns[3::3]  # every third column starting from Query_English
lang_dict = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te"
}
selected_lang = st.selectbox("Select language:", list(lang_dict.keys()))

# ----------------------------
# User Input
# ----------------------------
if 'user_query' not in st.session_state:
    st.session_state['user_query'] = ""

user_input = st.text_input(
    "Enter your question:",
    value=st.session_state['user_query'],
    key="input_box",
    placeholder="Type your legal question here and press Enter or Get Answer"
)

# ----------------------------
# Speech to Text (Mic Input)
# ----------------------------
def record_and_transcribe():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio)
            return text
    except Exception as e:
        st.warning(f"Mic input error: {e}")
        return ""

if st.button("🎤 Speak your question"):
    spoken_text = record_and_transcribe()
    if spoken_text:
        user_input = spoken_text
        st.session_state['user_query'] = spoken_text
        st.success(f"You said: {spoken_text}")

# ----------------------------
# IndiaCode Integration
# ----------------------------
def get_legal_answer(query, lang_code='en'):
    # Fuzzy matching first
    matched = df[df.apply(lambda row: query.lower() in str(row["Query_English"]).lower(), axis=1)]
    if not matched.empty:
        short = matched.iloc[0][f"Short_{selected_lang}"]
        detailed = matched.iloc[0][f"Detailed_{selected_lang}"]
        return short, detailed

    # If dataset fails → Live IndiaCode search
    try:
        urls = list(search(f"{query} site:indiacode.nic.in", num_results=3))
        if not urls:
            return "❌ Sorry, we couldn't find an exact answer. Try rephrasing.", "❌ Sorry, we couldn't find an exact answer. Try rephrasing."
        response = requests.get(urls[0])
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('div', {'id': 'content'})
        text = main_content.get_text(separator="\n").strip() if main_content else "Content not found."
        translator = Translator()
        translated_text = translator.translate(text, dest=lang_code).text
        short_answer = translated_text.split('.')[0]  # first sentence
        detailed_answer = translated_text
        return short_answer, detailed_answer
    except Exception as e:
        return "❌ Could not fetch answer from IndiaCode.", f"Error details: {e}"

# ----------------------------
# Get Answer
# ----------------------------
def get_answer_action():
    if not user_input.strip():
        st.warning("⚠️ Please enter a question.")
        return
    lang_code = lang_dict[selected_lang]
    short, detailed = get_legal_answer(user_input, lang_code)
    st.subheader("Short Answer:")
    st.write(short)
    st.subheader("Detailed Answer:")
    st.write(detailed)

    # ----------------------------
    # Text to Speech
    # ----------------------------
    try:
        tts = gTTS(text=short, lang=lang_code)
        t_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(t_file.name)
        st.audio(t_file.name)
    except Exception as e:
        st.warning(f"TTS not available for {selected_lang}: {e}")

# ----------------------------
# Enter or Button Trigger
# ----------------------------
if st.button("Get Answer") or st.session_state.get('input_box_enter', False):
    get_answer_action()

# ----------------------------
# Feedback
# ----------------------------
st.write("---")
st.subheader("Feedback")
col1, col2 = st.columns(2)
if col1.button("👍"):
    st.success("Thanks for your feedback!")
if col2.button("👎"):
    st.info("Thanks for your feedback! We'll improve.")

# ----------------------------
# Consult Lawyer Note
# ----------------------------
st.write("---")
st.info("Consult a lawyer nearby feature coming soon! 🚀")

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
    .stTextInput>div>div>input { height: 45px; font-size:16px; }
    .stButton button { height:45px; font-size:16px; }
    .stAudio audio { width: 100%; margin-top:10px; }
    </style>
    """,
    unsafe_allow_html=True
)

