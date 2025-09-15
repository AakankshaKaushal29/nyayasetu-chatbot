import streamlit as st
import pandas as pd
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import st_mic_recorder
from playsound import playsound
import tempfile

# ------------------------
# Load dataset
# ------------------------
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

df = load_data("SIH_Dataset_Final.xlsx")

# ------------------------
# Page setup
# ------------------------
st.set_page_config(page_title="Nyayasetu - AI Legal Consultant", layout="wide")
st.title("Nyayasetu - AI Legal Consultant")
st.markdown("""
Nyayasetu is an AI-based legal assistant that helps citizens get quick,
multi-language legal guidance in simple steps.
Ask your question and get structured answers based on Indian laws.
""")
st.markdown("---")

# ------------------------
# Example questions
# ------------------------
st.markdown("### Try an example question:")
examples = df['Query_English'].head(3).tolist()
col_ex = st.columns(len(examples))
for i, ex in enumerate(examples):
    if col_ex[i].button(ex):
        st.session_state['user_input'] = ex

# ------------------------
# User input
# ------------------------
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

# Text input
user_input = st.text_input("Enter your question or press Enter:", st.session_state['user_input'], key="input_field")

# Voice input
st.markdown("🎤 Or use your voice:")
audio_bytes = st_mic_recorder(key="mic_recorder")
if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        f.flush()
        # Use transcription API / Whisper / any service
        # For now, we just simulate transcription
        st.session_state['user_input'] = "Transcribed text from voice"
        user_input = st.session_state['user_input']

# ------------------------
# Get answer function
# ------------------------
def get_answer(query):
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
        return
    # Find the row with matching query
    row = df[df['Query_English'].str.lower() == query.lower()]
    if row.empty:
        st.info("Sorry, could not find an answer. Try rephrasing or contact us later.")
        return
    # Show short and detailed answers (depending on selected language)
    lang = st.selectbox("Select Language:", ["English", "Hindi", "Marathi", "Bengali", "Tamil", "Telugu"])
    
    short_col = f"Short_{lang}"
    detailed_col = f"Detailed_{lang}"
    short_answer = row.iloc[0][short_col]
    detailed_answer = row.iloc[0][detailed_col]
    
    # Display answers
    st.markdown("**Short Answer:**")
    st.markdown(short_answer)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Detailed Answer:**")
    st.markdown(detailed_answer)
    
    # ------------------------
    # Text-to-speech
    # ------------------------
    tts_text = f"{short_answer} {detailed_answer}"
    tts = gTTS(text=tts_text, lang='en')  # Could map lang code dynamically
    tts_fp = BytesIO()
    tts.write_to_fp(tts_fp)
    tts_fp.seek(0)
    st.audio(tts_fp.read(), format="audio/mp3")

    # ------------------------
    # Feedback buttons with spacing
    # ------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.button("👍 Helpful")
    with col2:
        st.button("👎 Not Helpful")

# ------------------------
# Trigger answer
# ------------------------
if st.button("Get Answer") or (st.session_state['user_input'] != "" and st.session_state['input_field'] != ""):
    get_answer(st.session_state['user_input'])

# Footer
st.markdown("---")
st.markdown("⚖️ Detailed lawyer consultation coming soon! Stay tuned for updates.")

