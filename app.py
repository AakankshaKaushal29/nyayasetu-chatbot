import streamlit as st
import pandas as pd
import re
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import os
from streamlit_mic_recorder import mic_recorder
import io
import soundfile as sf
import speech_recognition as sr
from pydub import AudioSegment

# ------------------------
# Load dataset
# ------------------------
@st.cache_data
def load_data():
    return pd.read_excel("AI_legal_assistant_detailed_expanded.xlsx")

df = load_data()

# ------------------------
# UI
# ------------------------
st.title("⚖️ NyayaSetu - Multilingual Legal Assistance Chatbot")
st.write("Select category, language, and ask your legal query to get clear, step-by-step guidance.")

# Language options
languages = {
    "Hindi": "hindi",
    "English": "english",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta"
}
selected_lang = st.selectbox("🌍 Choose language", list(languages.keys()))

# Category filter
all_categories = df['category'].dropna().unique().tolist()
selected_category = st.selectbox("📂 Choose category", ["All"] + all_categories)
df_filtered = df if selected_category == "All" else df[df['category'] == selected_category]

query_col = f"query_{languages[selected_lang]}"
answer_col = f"answer_{languages[selected_lang]}"

# ------------------------
# Build vectorizer
# ------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_filtered[query_col].astype(str))

# ------------------------
# Voice Input
# ------------------------
st.subheader("🎤 Voice Input")
voice_data = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop", just_once=True)

user_query = None
if voice_data and "bytes" in voice_data:
    try:
        audio_bytes = io.BytesIO(voice_data["bytes"])
        audio = AudioSegment.from_file(audio_bytes, format="webm")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            user_query = recognizer.recognize_google(audio_data, language="en-IN")
            st.success(f"Recognized voice query: {user_query}")
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")

# ------------------------
# Text Input fallback
# ------------------------
st.subheader("⌨️ Text Input")
typed_query = st.text_input(f"👉 Your Question ({selected_lang}):")
if not user_query:
    user_query = typed_query

# ------------------------
# Step-by-step function
# ------------------------
def get_detailed_steps(text, desired=6):
    if not text:
        return []

    # Remove empty lines first
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

    # Split into sentences
    sentences = re.split(r'(?<=[\.\?\!।])\s+', text)
    expanded = []
    for s in sentences:
        s_clean = s.strip()
        if ";" in s_clean:
            expanded.extend([p.strip() for p in s_clean.split(";") if p.strip()])
        elif "," in s_clean and len(s_clean.split()) > 12:
            expanded.extend(textwrap.wrap(s_clean, width=len(s_clean)//2))
        else:
            expanded.append(s_clean)

    # Remove empty strings and numbers-only steps
    expanded = [s for s in expanded if s.strip() and not re.fullmatch(r'\d+', s.strip())]

    # Pad to desired length
    while len(expanded) < desired:
        expanded.append(expanded[-1] if expanded else "")

    return expanded[:desired]

# ------------------------
# Handle query
# ------------------------
if user_query and user_query.strip():
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, X).flatten()
    idx = similarity.argmax()
    best_match = df_filtered.iloc[idx]

    st.subheader("📌 Result")
    st.write(f"**Category:** {best_match['category']}")
    st.write(f"**Closest Question ({selected_lang}):** {best_match[query_col]}")

    # Step-by-step display with perfect alignment
    steps = get_detailed_steps(best_match[answer_col], desired=6)
    cleaned_steps = [re.sub(r'^\d+\.\s*', '', s).strip() for s in steps if s.strip()]

    st.markdown("### ✅ Step-by-Step Guidance:")
    for i, step in enumerate(cleaned_steps):
        st.markdown(f"**{i+1}.** {step}")  # numbering stays clean, text on same line

    # TTS Output
    st.subheader("🔊 Hear the Answer")
    try:
        lang_code = languages.get(selected_lang, "en")
        tts = gTTS(best_match[answer_col], lang=lang_code)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")
    except Exception as e:
        st.error(f"TTS not available for {selected_lang}. ({e})")

    # Feedback
    st.write("---")
    st.write("💬 Was this answer helpful?")
    col1, col2 = st.columns(2)
    feedback_file = "feedback.csv"
    if not os.path.exists(feedback_file):
        pd.DataFrame(columns=["query", "answer", "feedback"]).to_csv(feedback_file, index=False)

    if col1.button("👍 Yes"):
        fb = pd.DataFrame([[user_query, best_match[answer_col], "positive"]],
                          columns=["query", "answer", "feedback"])
        fb.to_csv(feedback_file, mode="a", header=False, index=False)
        st.success("Thanks for your feedback! ✅")

    if col2.button("👎 No"):
        fb = pd.DataFrame([[user_query, best_match[answer_col], "negative"]],
                          columns=["query", "answer", "feedback"])
        fb.to_csv(feedback_file, mode="a", header=False, index=False)
        st.warning("Thanks for your feedback! 🙏")

    # Analytics
    st.write("---")
    st.subheader("📊 Live Analytics")
    fb_df = pd.read_csv(feedback_file)
    st.write(f"Total Queries: {len(fb_df)}")
    st.write(f"👍 Positive: {(fb_df['feedback'] == 'positive').sum()}")
    st.write(f"👎 Negative: {(fb_df['feedback'] == 'negative').sum()}")
    st.write("🔥 Most Asked Queries")
    st.table(fb_df['query'].value_counts().head(5))

else:
    st.info("👉 Please type your query or use the mic above to get an answer.")




