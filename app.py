import streamlit as st
import pandas as pd
import re
import textwrap
import tempfile
import os
import math
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

def split_into_sentences(text):
    """Split text into sentences; include Devanagari danda '।' for Hindi."""
    # keep punctuation marks as sentence boundaries
    sents = re.split(r'(?<=[।\.\?\!])\s+', text)
    sents_clean = [s.strip() for s in sents if s.strip()]
    return sents_clean

def split_text_into_n_parts(text, n):
    """Split text into n approximately-equal parts on word boundaries."""
    words = text.split()
    if not words:
        return []
    per = math.ceil(len(words) / n)
    parts = [' '.join(words[i:i+per]).strip() for i in range(0, len(words), per)]
    # If we produced more than n parts (rare), merge tail
    if len(parts) > n:
        merged = parts[:n-1] + [' '.join(parts[n-1:])]
        parts = merged
    # If fewer, return as is (caller will adjust)
    return parts

def normalize_step_text(s):
    s = s.strip()
    # remove leading numbering like "1." "1)" "Step 1:" etc.
    s = re.sub(r'^\s*(?:Step\s*)?\d+[\.\):\-]*\s*', '', s, flags=re.IGNORECASE)
    return s

def get_detailed_steps(text, desired=6):
    """
    Return a list of exactly up to `desired` step strings.
    Strategy:
      1) Prefer explicit newline-separated lines.
      2) Else use sentence-splitting.
      3) If still fewer than desired, split long text into `desired` word-based parts.
      4) Clean/normalize each step.
    """
    if text is None:
        return []
    txt = str(text).strip()
    if not txt:
        return []

    # 1) Try explicit lines
    lines = [l.strip() for l in re.split(r'\r?\n', txt) if l.strip()]
    # If lines look like a single long line that contains semicolons, split on semicolon
    if len(lines) == 1 and ';' in lines[0]:
        lines = [l.strip() for l in lines[0].split(';') if l.strip()]

    # 2) If not enough lines, try sentence splitting per paragraph
    if len(lines) < desired:
        sents = []
        for para in lines:
            sents.extend(split_into_sentences(para))
        # if we had no newline-separated lines originally, try splitting whole text into sentences
        if not lines:
            sents = split_into_sentences(txt)
        # use sents if it gives more items
        if len(sents) >= desired:
            chosen = sents[:desired]
        else:
            # 3) If still fewer, try sentence list but possibly shorter than desired
            if sents:
                chosen = sents
            else:
                chosen = []
    else:
        chosen = lines

    # Normalize chosen steps
    chosen = [normalize_step_text(s) for s in chosen if s.strip()]

    # If we have more than desired, truncate
    if len(chosen) >= desired:
        return chosen[:desired]

    # If fewer than desired, create additional parts by splitting the whole text into desired parts
    if len(chosen) < desired:
        parts = split_text_into_n_parts(txt, desired)
        # Normalize parts and return exactly desired parts
        parts = [normalize_step_text(p) for p in parts if p.strip()]
        if len(parts) >= desired:
            return parts[:desired]
        # Fallback: if both methods gave something, merge them and then pad by slicing words
        merged = chosen[:]
        remaining_needed = desired - len(merged)
        if remaining_needed > 0:
            # try to add from sentence list if available
            whole_sents = split_into_sentences(txt)
            for s in whole_sents:
                ns = normalize_step_text(s)
                if ns and ns not in merged:
                    merged.append(ns)
                    if len(merged) >= desired:
                        break
        # If still not enough, final fallback split
        if len(merged) < desired:
            final_parts = split_text_into_n_parts(txt, desired)
            merged = [normalize_step_text(p) for p in final_parts if p.strip()]
        return merged[:desired]

    return chosen[:desired]

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
# UI LAYOUT
# -------------------------
st.title("⚖️ NyayaSetu - Multilingual Legal Assistant (Demo)")
st.markdown("I will show **5–7 clear steps** on the page (no downloads). Choose how many steps you want:")

left, right = st.columns([2, 1])

with left:
    # language & step-count choice
    selected_lang = st.selectbox("🌍 Choose language", list(languages.keys()))
    desired_steps = st.selectbox("How many steps to show?", [5, 6, 7], index=1)  # default 6
    query_col = f"query_{languages[selected_lang]}"
    answer_col = f"answer_{languages[selected_lang]}"

    # TF-IDF setup (use chosen language queries)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[query_col].astype(str))

    # Input
    st.subheader("⌨️ Text Input")
    typed_query = st.text_input(f"👉 Your Question ({selected_lang}):")

    st.subheader("🎤 Voice Input (optional)")
    voice_audio = mic_recorder(start_prompt="Start", stop_prompt="Stop", just_once=True)
    # mic_recorder in some installations returns dict or audio bytes; we keep text input for accuracy
    if voice_audio:
        st.info("Voice recording captured. For demo reliability please paste the text into the input box (voice-to-text not enabled).")

    user_query = typed_query.strip()

    if user_query:
        # find best match
        query_vec = vectorizer.transform([user_query])
        similarity = cosine_similarity(query_vec, X).flatten()
        idx = int(similarity.argmax())
        best_match = df.iloc[idx]

        st.subheader("📌 Matched Result")
        st.write(f"**Category:** {best_match.get('category', 'N/A')}")
        st.write(f"**Closest Question ({selected_lang}):** {best_match.get(query_col, '')}")

        # produce desired number of steps (5-7)
        steps = get_detailed_steps(best_match.get(answer_col, ""), desired=desired_steps)

        if not steps:
            st.warning("No steps found in dataset for this question.")
        else:
            st.markdown("### ✅ Step-by-Step Guidance")
            for i, s in enumerate(steps):
                st.markdown(f"**{i+1}.** {s}")

        # TTS playback of the steps (joined)
        st.subheader("🔊 Hear these steps")
        try:
            lang_code = tts_lang_map.get(selected_lang, "en")
            tts_text = " ".join(steps)
            if tts_text.strip():
                tts = gTTS(tts_text, lang=lang_code)
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(tmp_file.name)
                st.audio(tmp_file.name, format="audio/mp3")
            else:
                st.info("No text available for TTS.")
        except Exception as e:
            st.error(f"TTS error: {e}")

        # Feedback UI (keeps analytics)
        st.write("---")
        st.write("💬 Was this answer helpful?")
        col1, col2 = st.columns(2)
        feedback_file = "feedback.csv"
        if not os.path.exists(feedback_file):
            pd.DataFrame(columns=["query", "answer", "feedback"]).to_csv(feedback_file, index=False)

        if col1.button("👍 Yes"):
            fb = pd.DataFrame([[user_query, steps_to_text(steps), "positive"]], columns=["query", "answer", "feedback"])
            fb.to_csv(feedback_file, mode="a", header=False, index=False)
            st.success("Thanks — noted as positive ✅")

        if col2.button("👎 No"):
            fb = pd.DataFrame([[user_query, steps_to_text(steps), "negative"]], columns=["query", "answer", "feedback"])
            fb.to_csv(feedback_file, mode="a", header=False, index=False)
            st.warning("Thanks — noted as negative. We'll review. 🙏")

    else:
        st.info("👉 Please type your query (voice-to-text not enabled in demo).")

with right:
    st.markdown("## 📊 Quick Analytics")
    feedback_file = "feedback.csv"
    show_most = st.checkbox("Show Most Asked Queries", value=False)

    if os.path.exists(feedback_file):
        df_fb = pd.read_csv(feedback_file)
        st.write(f"**Total interactions:** {len(df_fb)}")
        st.write(f"👍 {(df_fb['feedback'] == 'positive').sum()}  |  👎 {(df_fb['feedback'] == 'negative').sum()}")

        if show_most:
            st.subheader("🔥 Most Asked Queries (Top 5)")
            topq = df_fb['query'].value_counts().reset_index()
            topq.columns = ['query', 'count']
            st.table(topq.head(5))
    else:
        st.info("No feedback yet — vote on answers to populate analytics.")

