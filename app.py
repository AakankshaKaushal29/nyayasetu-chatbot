# app.py — full working version with mic, text fallback, TF-IDF matching,
# properly cleaned & aligned steps, multi-language TTS, feedback & analytics.

import os
import io
import re
import textwrap
import tempfile
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment

# ------------------------
# OPTIONAL: If you installed ffmpeg in C:\ffmpeg (Windows),
# uncomment the lines below so pydub finds ffmpeg without editing PATH.
# ------------------------
# if os.path.exists(r"C:\ffmpeg\bin\ffmpeg.exe"):
#     AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
#     AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"

# ------------------------
# Config & constants
# ------------------------
st.set_page_config(page_title="⚖️ NyayaSetu - Legal Assistant", layout="wide")
DATA_FILE = "AI_legal_assistant_detailed_expanded_cleaned.xlsx"  # your cleaned Excel
FEEDBACK_FILE = "feedback.csv"

# Map UI language name -> (column suffix, gTTS language code, SR language for recognition)
LANG_META = {
    "English": ("english", "en", "en-IN"),
    "Hindi": ("hindi", "hi", "hi-IN"),
    "Bengali": ("bengali", "bn", "bn-IN"),
    "Telugu": ("telugu", "te", "te-IN"),
    "Marathi": ("marathi", "mr", "mr-IN"),
    "Tamil": ("tamil", "ta", "ta-IN"),
}

# ------------------------
# Load dataset
# ------------------------
@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_excel(path)

try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(f"Could not load dataset `{DATA_FILE}`. Make sure the file exists. Error: {e}")
    st.stop()

# ------------------------
# Helper: clean & split detailed answers into steps
# ------------------------
def get_detailed_steps(text, desired=6):
    """
    Clean the long answer text and return a list of `desired` steps.
    Removes blank lines, stray numbering, and numeric-only lines.
    Attempts to split on "Step", language-specific markers, or sentence punctuation.
    """
    if not text or pd.isna(text):
        return []

    # Normalize to string and strip
    txt = str(text).strip()

    # 1) Remove blank lines and trim whitespace
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return []

    # 2) Join into one string (preserves sentences), but keep "Step" markers
    joined = " ".join(lines)

    # 3) Remove numbering like "1." or "1)" at start of fragments
    joined = re.sub(r'\b\d+\s*[\.\)]\s*', ' ', joined)

    # 4) Split into candidate parts:
    # split on Step markers in English or common local equivalents, else fallback to sentence punctuation
    parts = re.split(r'(?:Step\s*\d+\s*:?)|(?:चरण\s*\d+\s*:?)|(?:পদক্ষেপ\s*\d+\s*:?)|(?:দশ\s*\d+\s*:?)|(?<=[\.\?\!।])\s+', joined)
    parts = [p.strip() for p in parts if p and p.strip()]

    # 5) Further expand fragments with semicolons and long commas
    expanded = []
    for p in parts:
        p_clean = p.strip()
        if not p_clean:
            continue
        # Remove stray leading numbers again
        p_clean = re.sub(r'^\d+\.\s*', '', p_clean).strip()
        # Skip if only numeric
        if re.fullmatch(r'\d+', p_clean):
            continue
        if ";" in p_clean:
            expanded.extend([s.strip() for s in p_clean.split(";") if s.strip()])
        elif "," in p_clean and len(p_clean.split()) > 14:
            # break very long sentences roughly in half
            half = len(p_clean) // 2
            chunks = textwrap.wrap(p_clean, width=half)
            expanded.extend([c.strip() for c in chunks if c.strip()])
        else:
            expanded.append(p_clean)

    # Remove any empty strings and entries that are only numbers
    expanded = [e for e in expanded if e and not re.fullmatch(r'\d+', e)]

    if not expanded:
        return []

    # Pad/truncate to desired length
    while len(expanded) < desired:
        expanded.append(expanded[-1])
    return expanded[:desired]

# ------------------------
# UI: Top controls
# ------------------------
st.title("⚖️ NyayaSetu — Multilingual Legal Assistance")
st.write("Choose category & language, then type or speak your legal question. Steps will be shown cleanly; audio (TTS) available for all listed languages.")

# Language selection
selected_lang = st.selectbox("🌐 Choose language", list(LANG_META.keys()))
col_suffix, tts_lang_code, sr_lang_code = LANG_META[selected_lang]

# Category filter
all_categories = df['category'].dropna().unique().tolist()
selected_category = st.selectbox("📂 Choose category", ["All"] + all_categories)

# Filter dataset by category
df_filtered = df if selected_category == "All" else df[df['category'] == selected_category].copy()

# Query and answer column names
query_col = f"query_{col_suffix}"
# prefer detailed column if present, else fallback to less detailed
detailed_col_candidate = f"answer_{col_suffix}_detailed"
if detailed_col_candidate in df_filtered.columns:
    answer_detailed_col = detailed_col_candidate
else:
    # fallback to plain answer column
    answer_detailed_col = f"answer_{col_suffix}"

answer_plain_col = f"answer_{col_suffix}" if f"answer_{col_suffix}" in df_filtered.columns else answer_detailed_col

# ------------------------
# Build TF-IDF vectorizer for queries (for similarity matching)
# ------------------------
vectorizer = TfidfVectorizer()
# ensure no NaN
queries_for_vector = df_filtered[query_col].astype(str).fillna("")
X = vectorizer.fit_transform(queries_for_vector)

# ------------------------
# Voice Input (mic_recorder) — unchanged behavior
# ------------------------
st.subheader("🎤 Voice Input (optional)")
voice_data = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop", just_once=True, key="micrec")

user_query = None

if voice_data and isinstance(voice_data, dict) and "bytes" in voice_data:
    try:
        audio_bytes = io.BytesIO(voice_data["bytes"])
        # try explicit webm first (streamlit mic usually gives webm/opus)
        try:
            audio = AudioSegment.from_file(audio_bytes, format="webm")
        except Exception:
            audio_bytes.seek(0)
            audio = AudioSegment.from_file(audio_bytes)  # let pydub detect
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_rec = recognizer.record(source)
            # use sr_lang_code (like 'hi-IN', 'en-IN' etc.)
            try:
                user_query = recognizer.recognize_google(audio_rec, language=sr_lang_code)
                st.success(f"Recognized voice query: {user_query}")
            except sr.UnknownValueError:
                st.error("Could not understand audio.")
            except sr.RequestError as e:
                st.error(f"Speech recognition service error: {e}")
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")

# ------------------------
# Text input fallback
# ------------------------
st.subheader("⌨️ Text Input")
typed_query = st.text_input(f"👉 Your Question ({selected_lang}):")
if not user_query:
    user_query = typed_query

# ------------------------
# Decide which input to use (preserve previous behavior)
# ------------------------
# (user_query already set above by mic or typed input)

# ------------------------
# Helper: ensure feedback file exists
# ------------------------
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["timestamp", "language", "query", "closest_question", "answer", "feedback"]).to_csv(FEEDBACK_FILE, index=False)

# ------------------------
# Handle query: similarity match + display
# ------------------------
if user_query and str(user_query).strip():
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, X).flatten()
    idx = similarity.argmax()
    try:
        best_match = df_filtered.iloc[idx]
    except Exception:
        st.error("No matching question found.")
        best_match = None

    if best_match is not None:
        st.subheader("📌 Result")
        # Category
        st.write(f"**Category:** {best_match.get('category', '')}")
        # Closest question
        closest_q = best_match.get(query_col, "")
        st.write(f"**Closest Question ({selected_lang}):** {closest_q}")

        # Prepare detailed steps (cleaned)
        raw_answer_detailed = best_match.get(answer_detailed_col, "") or ""
        steps = get_detailed_steps(raw_answer_detailed, desired=6)
        # Remove any residual numbering at start and drop empty
        cleaned_steps = [re.sub(r'^\d+\.\s*', '', s).strip() for s in steps if s and s.strip()]

        # Show step-by-step: first 2 visible, rest in expander
        st.markdown("### ✅ Step-by-Step Guidance:")
        # If no steps found, show raw answer
        if not cleaned_steps:
            st.write(raw_answer_detailed)
        else:
            # show first 2
            for i, step in enumerate(cleaned_steps[:2], start=1):
                st.markdown(f"**{i}.** {step}")
            if len(cleaned_steps) > 2:
                with st.expander("Read More"):
                    for j, step in enumerate(cleaned_steps[2:], start=3):
                        st.markdown(f"**{j}.** {step}")

        # TTS — generate audio for the detailed answer and play
        st.subheader("🔊 Hear the Answer")
        tts_code = tts_lang_code  # from LANG_META, guaranteed to be correct code (en, hi, bn,...)

        try:
            # Use the joined cleaned steps as text for TTS (keeps short and clean)
            tts_text = " ".join(cleaned_steps) if cleaned_steps else str(raw_answer_detailed)
            if not tts_text.strip():
                st.warning("No textual answer available for TTS.")
            else:
                # Create TTS file and play
                tts = gTTS(text=tts_text, lang=tts_code)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tmp.close()
                tts.save(tmp.name)
                st.audio(tmp.name, format="audio/mp3")
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
        except Exception as e:
            st.error(f"TTS not available for {selected_lang}. ({e})")

        # ------------------------
        # Feedback buttons & logging
        # ------------------------
        st.write("---")
        st.write("💬 Was this answer helpful?")
        col1, col2 = st.columns([1, 1])
        if col1.button("👍 Yes"):
            entry = {
                "timestamp": pd.Timestamp.now(),
                "language": selected_lang,
                "query": user_query,
                "closest_question": closest_q,
                "answer": raw_answer_detailed,
                "feedback": "positive"
            }
            pd.DataFrame([entry]).to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
            st.success("Thanks for your feedback! ✅")
        if col2.button("👎 No"):
            entry = {
                "timestamp": pd.Timestamp.now(),
                "language": selected_lang,
                "query": user_query,
                "closest_question": closest_q,
                "answer": raw_answer_detailed,
                "feedback": "negative"
            }
            pd.DataFrame([entry]).to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
            st.warning("Thanks for your feedback! We will improve this answer. 🙏")

# if no query yet
else:
    st.info("👉 Please type your query or use the mic above to get an answer.")

# ------------------------
# Analytics & Most Asked
# ------------------------
st.markdown("---")
st.subheader("📊 Live Analytics / Most Asked Queries")

try:
    fb_df = pd.read_csv(FEEDBACK_FILE)
    total = len(fb_df)
    pos = (fb_df['feedback'] == 'positive').sum()
    neg = (fb_df['feedback'] == 'negative').sum()
    st.write(f"Total feedback entries: {total}")
    st.write(f"👍 Positive: {pos}")
    st.write(f"👎 Negative: {neg}")

    if total > 0:
        st.write("🔥 Most asked queries (from feedback):")
        top = fb_df['query'].value_counts().head(10)
        st.table(top.rename_axis("query").reset_index(name="count"))
    else:
        st.info("No feedback yet — top queries from dataset:")
        sample_qs = df_filtered[query_col].dropna().head(10).tolist()
        st.write("\n".join([f"- {q}" for q in sample_qs]))
except Exception:
    # fallback: show top queries from dataset
    sample_qs = df_filtered[query_col].dropna().head(10).tolist()
    st.write("\n".join([f"- {q}" for q in sample_qs]))


