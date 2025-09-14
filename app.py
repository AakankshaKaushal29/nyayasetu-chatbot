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
import speech_recognition as sr
import io
import soundfile as sf

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
    "Bengali": "bengali",
    "Telugu": "telugu",
    "Marathi": "marathi",
    "Tamil": "tamil"
}
selected_lang = st.selectbox("🌍 Choose language", list(languages.keys()))

# Category filter
all_categories = df['category'].dropna().unique().tolist()
selected_category = st.selectbox("📂 Choose category", ["All"] + all_categories)

# Filter dataframe by category
df_filtered = df if selected_category == "All" else df[df['category'] == selected_category]

query_col = f"query_{languages[selected_lang]}"
answer_col = f"answer_{languages[selected_lang]}"

# ------------------------
# Build vectorizer on filtered data
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
        # Convert recorded audio bytes to WAV
        audio_bytes = io.BytesIO(voice_data["bytes"])
        data, samplerate = sf.read(audio_bytes)
        wav_io = io.BytesIO()
        sf.write(wav_io, data, samplerate, format="WAV")
        wav_io.seek(0)

        # Recognize speech using Google Web Speech API
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            user_query = recognizer.recognize_google(audio_data, language="en-IN")
            st.success(f"Recognized voice query: {user_query}")

    except Exception as e:
        st.error(f"Voice recognition failed: {e}")

# ------------------------
# Text Input (Fallback)
# ------------------------
st.subheader("⌨️ Text Input")
typed_query = st.text_input(f"👉 Your Question ({selected_lang}):")
if not user_query:
    user_query = typed_query

# ------------------------
# Function to split into 5–6 full steps
# ------------------------
def get_detailed_steps(text, desired=6):
    if not text:
        return []
    txt = str(text).strip()

    # Split on sentence end markers
    sentences = re.split(r'(?<=[\.\?\!।])\s+', txt)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) >= desired:
        return sentences[:desired]

    # Try splitting further if too few
    expanded = []
    for s in sentences:
        if ";" in s:
            expanded.extend([p.strip() for p in s.split(";") if p.strip()])
        elif "," in s and len(s.split()) > 12:
            expanded.extend(textwrap.wrap(s, width=len(s)//2))
        else:
            expanded.append(s)

    expanded = [e for e in expanded if e]

    if len(expanded) >= desired:
        return expanded[:desired]
    else:
        while len(expanded) < desired:
            expanded.append(expanded[-1])
        return expanded

# ------------------------
# Handle query
# ------------------------
if user_query and user_query.strip():
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, X).flatten()
    idx = similarity.argmax()
    best_match = df_filtered.iloc[idx]

    # Show result
    st.subheader("📌 Result")
    st.write(f"**Category:** {best_match['category']}")
    st.write(f"**Closest Question ({selected_lang}):** {best_match[query_col]}")

    # Step-by-step
    steps = get_detailed_steps(best_match[answer_col], desired=6)
    st.markdown("### ✅ Step-by-Step Guidance:")

    # Show first 2 steps
    for i, step in enumerate(steps[:2]):
        st.markdown(f"**{i+1}.** {step.replace('\n', '<br>&emsp;')}", unsafe_allow_html=True)

    # Remaining inside expander
    if len(steps) > 2:
        with st.expander("Read More"):
            for j, step in enumerate(steps[2:], start=3):
                st.markdown(f"**{j}.** {step.replace('\n', '<br>&emsp;')}", unsafe_allow_html=True)

    # TTS Output
    st.subheader("🔊 Hear the Answer")
    try:
        lang_map = {
            "Hindi": "hi",
            "English": "en",
            "Bengali": "bn",
            "Telugu": "te",
            "Marathi": "mr",
            "Tamil": "ta"
        }
        lang_code = lang_map.get(selected_lang, "en")
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



