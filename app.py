import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import os

# Voice input
from streamlit_mic_recorder import mic_recorder

# ------------------------
# Load dataset
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("AI_legal_assistant_detailed.xlsx")
    return df

df = load_data()

# ------------------------
# UI
# ------------------------
st.title("⚖️ NyayaSetu - Multilingual Legal Assistant")
st.write("Type or speak your legal query and get **step-by-step guidance** in your preferred language.")

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

# ------------------------
# Build vectorizer
# ------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df[query_col].astype(str))

# ------------------------
# Input Section
# ------------------------
st.subheader("🎤 Voice Input")
voice_text = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop", just_once=True)

st.subheader("⌨️ Text Input")
typed_query = st.text_input(f"👉 Your Question ({selected_lang}):")

# Decide which input to use (voice has priority)
user_query = None
if voice_text:
    if isinstance(voice_text, dict):  # sometimes mic_recorder returns dict
        user_query = voice_text.get("text", "")
    else:
        user_query = voice_text
else:
    user_query = typed_query

# ------------------------
# Answer Section
# ------------------------
if user_query and str(user_query).strip():
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, X).flatten()
    idx = similarity.argmax()
    best_match = df.iloc[idx]

    st.subheader("📌 Result")
    st.write(f"**Matched Category:** {best_match['category']}")
    st.write(f"**Similar Question ({selected_lang}):** {best_match[query_col]}")

    # Step-by-step guidance
    steps = best_match[answer_col].split("\n")
    st.markdown("### ✅ Step-by-Step Guidance:")
    for i, step in enumerate(steps):
        if i < 2:  # show first two
            st.markdown(f"- {step.strip()}")
        else:
            with st.expander("Read More Steps"):
                for extra_step in steps[2:]:
                    st.markdown(f"- {extra_step.strip()}")
                break

    # ------------------------
    # TTS Output
    # ------------------------
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

    # ------------------------
    # Feedback
    # ------------------------
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

# ------------------------
# Mini Analytics Dashboard
# ------------------------
st.write("## 📊 Quick Analytics (Live)")
feedback_file = "feedback.csv"
if os.path.exists(feedback_file):
    df_fb = pd.read_csv(feedback_file)
    st.write(f"**Total Queries:** {len(df_fb)}")
    st.write(f"👍 Positive: {(df_fb['feedback'] == 'positive').sum()} | 👎 Negative: {(df_fb['feedback'] == 'negative').sum()}")

    st.subheader("📈 Feedback Breakdown")
    st.bar_chart(df_fb['feedback'].value_counts())

    st.subheader("🔥 Most Asked Queries")
    st.table(df_fb['query'].value_counts().head(5))
else:
    st.info("No feedback yet. Interact with the chatbot above to see analytics.")
