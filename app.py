import streamlit as st
import pandas as pd
from gtts import gTTS
from io import BytesIO
import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("SIH_Dataset_Final.xlsx")
    return df

df = load_data()

# App title & intro
st.title("⚖️ Nyayasetu - AI Legal Consultant")
st.markdown(
    """
Nyayasetu is an AI-based legal assistant that helps citizens get quick, multi-language legal guidance in simple steps.
Ask your question and get structured answers based on Indian laws.
"""
)

# Select language
language_map = {
    "English": "EN",
    "Hindi": "Hindi",
    "Bengali": "Bengali",
    "Marathi": "Marathi",
    "Tamil": "Tamil",
    "Telugu": "Telugu"
}

selected_lang = st.selectbox("Select language:", list(language_map.keys()))

# Example questions
example_queries = df[f'Query_{language_map[selected_lang]}'].dropna().tolist()[:3]
st.markdown("**Try one of these example questions:**")
cols = st.columns(len(example_queries))
for i, col in enumerate(cols):
    if col.button(example_queries[i]):
        st.session_state['query'] = example_queries[i]

# Question input
query_input = st.text_input("Enter your question:", value=st.session_state.get('query', ''))

def get_answer(query):
    # Match exact first
    col_name = f"Query_{language_map[selected_lang]}"
    short_col = f"Short_{language_map[selected_lang]}"
    detailed_col = f"Detailed_{language_map[selected_lang]}"

    match = df[df[col_name].str.lower() == query.lower()]
    if not match.empty:
        short_answer = match[short_col].values[0]
        detailed_answer = match[detailed_col].values[0]
        return short_answer, detailed_answer
    else:
        return "❌ Sorry, we couldn't find an exact answer. Try rephrasing.", \
               "❌ Sorry, we couldn't find an exact answer. Try rephrasing."

# Submit question (Enter or button)
if st.button("Get Answer") or st.session_state.get('submit', False):
    if query_input:
        st.session_state['submit'] = False
        short_answer, detailed_answer = get_answer(query_input)

        # Display answers
        st.subheader("Short Answer:")
        st.write(short_answer)
        st.subheader("Detailed Answer:")
        st.write(detailed_answer)

        # Text-to-speech for EN & Hindi only
        if selected_lang in ["English", "Hindi"]:
            tts = gTTS(text=short_answer, lang="en" if selected_lang=="English" else "hi")
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            st.audio(audio_bytes.getvalue(), format="audio/mp3")

        # Feedback buttons
        col1, col2 = st.columns(2)
        if col1.button("👍"):
            st.success("Thanks for your feedback!")
        if col2.button("👎"):
            st.warning("Thanks for your feedback! We will improve.")

# Consult lawyer message
st.markdown("*Consult a lawyer nearby feature coming soon!*")
