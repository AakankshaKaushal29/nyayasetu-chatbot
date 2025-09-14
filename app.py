import streamlit as st
import pandas as pd
import re
import textwrap
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="NyayaSetu — Multilingual Legal Assistant", layout="wide")

# -------------------------
# HELPERS
# -------------------------
def load_data(path="AI_legal_assistant_detailed_expanded.xlsx"):
    return pd.read_excel(path)

def get_detailed_steps(text, max_steps=6):
    if not text:
        return []
    txt = str(text).strip()

    if "\n" in txt:
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
    elif ";" in txt:
        lines = [l.strip() for l in txt.split(";") if l.strip()]
    else:
        lines = re.split(r'(?<=[.?!])\s+', txt)

    if len(lines) < 5:
        chunks = textwrap.wrap(txt, width=100)
        lines = chunks

    return lines[:max_steps]

def steps_to_text(steps):
    return "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def cached_load():
    return load_data()

df = cached_load()

# -------------------------
# LANGUAGES
# -------------------------
languages = {
    "English": "english",
    "Hindi": "hindi",
    "Bengali": "bengali",
    "Telugu": "telugu",
    "Marathi": "marathi",
    "Tamil": "tamil"
}

tts_lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta"
}

# -------------------------
# UI
# -------------------------
st.title("⚖️ NyayaSetu - Multilingual Legal Assistant (SIH Demo)")
st.markdown("Helping citizens overcome language barriers with clear legal guidance.")

selected_lang = st.selectbox("🌍 Choose language", list(languages.keys()))
query_col = f"query_{languages[selected_lang]}"
answer_col = f"answer_{languages[selected_lang]}"

# TF-IDF setup
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df[query_col].astype(str))

# -------------------------
# INPUT SECTION
# -------------------------
st.subheader("🎤 Voice Input (Demo)")
voice_audio = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="🛑 Stop",
    just_once=True
)

st.subheader("⌨️ Text Input")
typed_query = st.text_input(f"👉 Your Question ({selected_lang}):")

# Handle mic
user_query = ""
if voice_audio:
    st.warning("⚠️ Voice transcription not enabled. Please use text input.")
else:
    user_query = typed_query.strip()

# -------------------------
# QUERY HANDLING
# -------------------------
if user_query:
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, X).flatten()
    idx = int(similarity.argmax())
    best_match = df.iloc[idx]

    st.subheader("📌 Matched Result")
    st.write(f"**Category:** {best_match.get('category', 'N/A')}")
    st.write(f"**Closest Question ({selected_lang}):** {best_match.get(query_col, '')}")

    steps = get_detailed_steps(best_match.get(answer_col, ""), max_steps=6)

    if not steps:
        st.warning("No detailed steps found in dataset for this query.")
    else:
        st.markdown("### ✅ Step-by-Step Guidance")
        for i, s in enumerate(steps):
            st.markdown(f"**{i+1}.** {s}")

    report_text = f"Query: {user_query}\n\nSteps:\n{steps_to_text(steps)}"
    st.download_button("📄 Download Report (TXT)", data=report_text, file_name="nyayasetu_report.txt")

    st.subheader("🔊 Hear the Answer")
    try:
        lang_code = tts_lang_map.get(selected_lang, "en")
        tts_text = " ".join(steps)
        tts = gTTS(tts_text, lang=lang_code)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")
    except Exception as e:
        st.error(f"TTS error: {e}")

else:
    st.info("👉 Please type your query or use the mic above.")
