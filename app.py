import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import speech_recognition as sr
from pydub import AudioSegment

# ------------------------
# Load Dataset
# ------------------------
df = pd.read_excel("SIH_Dataset_Final.xlsx")

# ------------------------
# Language Map
# ------------------------
lang_code_map = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te"
}

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Nyayasetu - AI Legal Consultant", page_icon="⚖️")
st.title("Nyayasetu - AI Legal Consultant")
st.markdown("""
Nyayasetu is an AI-based legal assistant that helps citizens get quick,
multi-language legal guidance in simple steps.  
Ask your question and get structured answers based on Indian laws.
""")

# ------------------------
# Language selection
# ------------------------
languages = list(lang_code_map.keys())
selected_lang = st.selectbox("🌐 Select Language", languages, index=0)

# ------------------------
# Example Questions (clickable)
# ------------------------
example_questions = [
    "How to file an FIR?",
    "What to do in case of cyber fraud?",
    "How can I register a marriage?"
]
st.markdown("**Example Questions:**")
for q in example_questions:
    if st.button(q):
        st.session_state.user_input = q
        st.experimental_rerun()

# ------------------------
# Live Mic / Speech-to-Text
# ------------------------
st.markdown("**🎤 Speak your question:**")
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def record_and_transcribe():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Recording... Speak now")
        audio_data = r.listen(source, phrase_time_limit=7)
    try:
        text = r.recognize_google(audio_data, language=lang_code_map[selected_lang])
        st.session_state.user_input = text
        st.success(f"You said: {text}")
    except:
        st.warning("Sorry, could not recognize your voice.")

if st.button("🎙️ Record"):
    record_and_transcribe()

# ------------------------
# Get Answer Function
# ------------------------
def get_answer():
    user_query = st.session_state.user_input
    if not user_query:
        st.warning("⚠️ Please enter a question.")
        return
    query_col = f"Query_{selected_lang}"
    short_col = f"Short_{selected_lang}"
    detailed_col = f"Detailed_{selected_lang}"

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[query_col].astype(str))
    user_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx = similarity.argmax()
    best_match = df.iloc[best_idx]

    short_ans = best_match[short_col]
    detailed_ans = best_match[detailed_col]

    # ------------------------
    # Display Answers
    # ------------------------
    st.markdown(f"**Short Answer:** {short_ans}")
    st.markdown(f"**Detailed Answer:** {detailed_ans}")

    # ------------------------
    # Text-to-Speech
    # ------------------------
    tts_text = f"{short_ans}. {detailed_ans}"
    tts = gTTS(text=tts_text, lang=lang_code_map[selected_lang])
    tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_file.name)
    st.audio(tts_file.name, format="audio/mp3")

# ------------------------
# Text Input Box (Enter key triggers)
# ------------------------
st.text_input("Ask your question:", key="user_input", on_change=get_answer)

# ------------------------
# Feedback Section
# ------------------------
st.markdown("**Feedback:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("👍"):
        st.success("Thanks for your feedback!")
with col2:
    if st.button("👎"):
        st.info("We’ll review this answer.")

# ------------------------
# Footer / Contact Info
# ------------------------
st.markdown("---")
st.markdown("⚖️ Detailed lawyer consultation coming soon! Stay tuned for updates.")
