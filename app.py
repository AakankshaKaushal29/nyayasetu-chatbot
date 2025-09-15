# app.py
import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import os

# -----------------------
# Load Dataset
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_excel("SIH_Dataset_Final.xlsx")
    return df

df = load_data()

# -----------------------
# App Title
# -----------------------
st.set_page_config(page_title="Nyayasetu - AI Legal Consultant", page_icon="⚖️")
st.title("Nyayasetu - AI Legal Consultant")
st.markdown("""
Nyayasetu is an AI-based legal assistant that helps citizens get quick, multi-language legal guidance in simple steps.
Ask your question and get structured answers based on Indian laws.
""")

# -----------------------
# Example Questions
# -----------------------
example_questions = df['Query_English'].head(5).tolist()
st.markdown("**Try one of these example questions:**")
cols = st.columns(len(example_questions))
for i, col in enumerate(cols):
    if col.button(example_questions[i]):
        st.session_state['user_input'] = example_questions[i]

# -----------------------
# User Input
# -----------------------
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

user_input = st.text_input("Enter your question:", value=st.session_state['user_input'], key="input_box")

# Enter key functionality
if st.session_state['user_input'] != user_input:
    st.session_state['user_input'] = user_input

# -----------------------
# Voice Input (optional)
# -----------------------
try:
    from streamlit_mic_recorder import st_mic_recorder
    voice_available = True
except ImportError:
    st.info("🎤 Voice input is not available in this environment.")
    voice_available = False

if voice_available:
    st.markdown("🎤 Speak your question:")
    audio_bytes = st_mic_recorder(key="mic_recorder")
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            f.flush()
        # You can replace this with actual speech-to-text API like OpenAI Whisper
        st.session_state['user_input'] = "Transcribed text from voice"

# -----------------------
# Language Selection
# -----------------------
languages = ["English", "Hindi", "Marathi", "Bengali", "Tamil", "Telugu"]
selected_lang = st.selectbox("Select language:", languages)

# -----------------------
# Get Answer Button
# -----------------------
if st.button("Get Answer") or st.session_state['user_input']:
    query = st.session_state['user_input'].strip()
    
    if not query:
        st.warning("⚠️ Please enter a question.")
    else:
        # Search for closest match in selected language column
        lang_col_map = {
            "English": "Query_English",
            "Hindi": "Query_Hindi",
            "Marathi": "Query_Marathi",
            "Bengali": "Query_Bengali",
            "Tamil": "Query_Tamil",
            "Telugu": "Query_Telugu"
        }
        short_col_map = {
            "English": "Short_English",
            "Hindi": "Short_Hindi",
            "Marathi": "Short_Marathi",
            "Bengali": "Short_Bengali",
            "Tamil": "Short_Tamil",
            "Telugu": "Short_Telugu"
        }
        detailed_col_map = {
            "English": "Detailed_English",
            "Hindi": "Detailed_Hindi",
            "Marathi": "Detailed_Marathi",
            "Bengali": "Detailed_Bengali",
            "Tamil": "Detailed_Tamil",
            "Telugu": "Detailed_Telugu"
        }

        df_lang = df[[lang_col_map[selected_lang], short_col_map[selected_lang], detailed_col_map[selected_lang]]]
        df_lang.columns = ["Query", "Short", "Detailed"]

        # Simple matching
        matched = df_lang[df_lang['Query'].str.contains(query, case=False, na=False)]
        if not matched.empty:
            short_answer = matched.iloc[0]['Short']
            detailed_answer = matched.iloc[0]['Detailed']
        else:
            short_answer = "Sorry, I couldn't find an answer. Please try rephrasing or contact legal aid."
            detailed_answer = short_answer

        # -----------------------
        # Display Answers
        # -----------------------
        st.markdown("**Short Answer:**")
        st.write(short_answer)
        st.markdown("**Detailed Answer:**")
        st.write(detailed_answer)

        # -----------------------
        # Text-to-Speech
        # -----------------------
        tts_col1, tts_col2 = st.columns(2)
        with tts_col1:
            if st.button("🔊 Play Short Answer"):
                tts = gTTS(text=short_answer, lang=selected_lang[:2].lower())
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    tts.save(f.name)
                    st.audio(f.name, format='audio/mp3')
        with tts_col2:
            if st.button("🔊 Play Detailed Answer"):
                tts = gTTS(text=detailed_answer, lang=selected_lang[:2].lower())
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    tts.save(f.name)
                    st.audio(f.name, format='audio/mp3')

        # -----------------------
        # Feedback
        # -----------------------
        st.markdown("**Did this answer help you?**")
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            if st.button("👍 Yes"):
                st.success("Thanks for your feedback!")
        with feedback_col2:
            if st.button("👎 No"):
                st.info("Thanks! We'll try to improve.")

# -----------------------
# Footer / Placeholder for future
# -----------------------
st.markdown("---")
st.info("Consult a lawyer nearby feature coming soon 🚀")

