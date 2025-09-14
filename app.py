import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import os
import re
import textwrap
import plotly.express as px

# Voice input
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="NyayaSetu — Multilingual Legal Assistant", layout="wide")

# ------------------------
# Helpers
# ------------------------
def load_data(path="AI_legal_assistant_detailed.xlsx"):
    return pd.read_excel(path)

def get_detailed_steps(text, max_steps=6):
    """
    Try to produce a list of up to `max_steps` readable step lines from `text`.
    Methods (in order):
      1. Split on explicit newlines.
      2. Sentence-split using punctuation . ? !
      3. Fallback: chunk text into wrap-width segments.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    txt = str(text).strip()
    if not txt:
        return []

    # 1) Lines
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if lines:
        # If lines include "Step" prefixes, keep them, otherwise we'll return them as-is
        return lines[:max_steps]

    # 2) Sentence split (works for many languages that use .?!)
    sents = re.split(r'(?<=[\.\?\!])\s+', txt)
    sents = [s.strip() for s in sents if s.strip()]
    if sents:
        return sents[:max_steps]

    # 3) Fallback chunking
    chunks = textwrap.wrap(txt, width=100)
    return chunks[:max_steps]

def steps_to_text(steps):
    return "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])

# ------------------------
# Load dataset
# ------------------------
@st.cache_data
def cached_load():
    return load_data()

df = cached_load()

# ------------------------
# UI - Layout
# ------------------------
st.title("⚖️ NyayaSetu - Multilingual Legal Assistant (Demo)")
st.write("Type or speak your legal query and get a full step-by-step process (5-6 points).")

left_col, right_col = st.columns([2, 1])

with left_col:
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
    query_col = f"query_{languages[selected_lang]}"
    answer_col = f"answer_{languages[selected_lang]}"

    # Build vectorizer on chosen language column
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[query_col].astype(str))

    # Input section
    st.subheader("🎤 Voice Input")
    voice_text = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop", just_once=True)

    st.subheader("⌨️ Text Input")
    typed_query = st.text_input(f"👉 Your Question ({selected_lang}):")

    # Decide input source
    user_query = ""
    if voice_text:
        if isinstance(voice_text, dict):
            user_query = voice_text.get("text", "") or ""
        else:
            user_query = str(voice_text)
    else:
        user_query = typed_query or ""

    # Query handling
    if user_query and user_query.strip():
        query_vec = vectorizer.transform([user_query])
        similarity = cosine_similarity(query_vec, X).flatten()
        idx = int(similarity.argmax())
        best_match = df.iloc[idx]

        st.subheader("📌 Matched Result")
        st.write(f"**Category:** {best_match.get('category', 'N/A')}")
        st.write(f"**Closest Question ({selected_lang}):** {best_match.get(query_col, '')}")

        # Get steps (up to 6)
        full_steps = get_detailed_steps(best_match.get(answer_col, ""), max_steps=6)
        if not full_steps:
            st.warning("No detailed steps found in dataset for this question.")
        else:
            st.markdown("### ✅ Full process (detailed steps)")
            for i, s in enumerate(full_steps):
                st.markdown(f"**{i+1}.** {s}")

            # Optional: show more if dataset actually had more lines
            # (Assumes original answer text may contain more content)
            original_text = str(best_match.get(answer_col, ""))
            if "\n" in original_text and len(original_text.splitlines()) > len(full_steps):
                with st.expander("Show full original text"):
                    st.write(original_text)

        # Downloadable report for the user
        report_text = f"Query: {user_query}\n\nSteps:\n{steps_to_text(full_steps)}\n\nSource Question: {best_match.get(query_col,'')}\nCategory: {best_match.get('category','')}"
        st.download_button("📄 Download Full Report (TXT)", data=report_text, file_name="nyayasetu_report.txt")

        # TTS
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
            tts_text = " ".join(full_steps)
            tts = gTTS(tts_text, lang=lang_code)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
        except Exception as e:
            st.error(f"TTS error or language not supported: {e}")

        # Feedback
        st.write("---")
        st.write("💬 Were these steps helpful?")
        col_y, col_n = st.columns(2)
        feedback_file = "feedback.csv"
        if not os.path.exists(feedback_file):
            pd.DataFrame(columns=["query", "answer", "feedback"]).to_csv(feedback_file, index=False)

        if col_y.button("👍 Yes"):
            fb = pd.DataFrame([[user_query, steps_to_text(full_steps), "positive"]],
                              columns=["query", "answer", "feedback"])
            fb.to_csv(feedback_file, mode="a", header=False, index=False)
            st.success("Thanks for your feedback! ✅")

        if col_n.button("👎 No"):
            fb = pd.DataFrame([[user_query, steps_to_text(full_steps), "negative"]],
                              columns=["query", "answer", "feedback"])
            fb.to_csv(feedback_file, mode="a", header=False, index=False)
            st.warning("Thanks for your feedback. We'll review it. 🙏")

    else:
        st.info("👉 Please type your query or use the mic above to get an answer.")

with right_col:
    st.markdown("## 📊 Quick Analytics")
    feedback_file = "feedback.csv"
    show_most = st.checkbox("Show Most Asked Queries", value=False)

    if os.path.exists(feedback_file):
        df_fb = pd.read_csv(feedback_file)
        total = len(df_fb)
        pos = int((df_fb['feedback'] == 'positive').sum())
        neg = int((df_fb['feedback'] == 'negative').sum())
        st.write(f"**Total interactions:** {total}")
        st.write(f"👍 {pos}  |  👎 {neg}")

        # Plotly bar (compact, labeled)
        counts = df_fb['feedback'].value_counts().rename_axis('feedback').reset_index(name='count')
        # ensure ordering
        counts['feedback'] = counts['feedback'].astype(str)
        color_map = {"positive": "#2ca02c", "negative": "#d62728"}
        # provide a safe color map for unknown labels:
        counts['color'] = counts['feedback'].map(color_map).fillna("#1f77b4")
        fig = px.bar(counts, x='feedback', y='count', color='feedback',
                     color_discrete_map=color_map, text='count', height=260)
        fig.update_traces(textposition='outside', showlegend=False)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if show_most:
            st.subheader("🔥 Most Asked Queries (Top 5)")
            topq = df_fb['query'].value_counts().reset_index()
            topq.columns = ['query', 'count']
            st.table(topq.head(5))
    else:
        st.info("No feedback yet — any interaction (👍/👎) will appear here in realtime.")
