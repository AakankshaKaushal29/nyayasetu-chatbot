import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import os

# Load Excel dataset
df = pd.read_excel("SIH_Dataset_Final.xlsx")

st.set_page_config(page_title="Nyayasetu - AI Legal Consultant", page_icon="⚖️", layout="wide")

st.markdown("## ⚖️ Nyayasetu - AI Legal Consultant")
st.markdown("""
Nyayasetu is an AI-based legal assistant that helps citizens get quick, multi-language legal guidance in simple steps.
Ask your question and get structured answers based on Indian laws.
""")

# Example questions
st.markdown("Try one of these example questions:")
example_questions = df['Query_English'].head(3).tolist()
for q in example_questions:
    if st.button(q):
        st.session_state['question'] = q

# Language selection
languages = ['English', 'Hindi', 'Marathi', 'Bengali', 'Tamil', 'Telugu']
selected_lang = st.selectbox("Select language:", languages)

# User input
if 'question' not in st.session_state:
    st.session_state['question'] = ""

user_question = st.text_input("Enter your question:", value=st.session_state['question'], key="question_input")

# Function to play text-to-speech
def play_audio(text, lang):
    try:
        tts = gTTS(text=text, lang=lang[:2].lower())
        with tempfile.NamedTemporaryFile(delete=True) as fp:
            tts.save(f"{fp.name}.mp3")
            st.audio(f"{fp.name}.mp3")
    except:
        st.warning(f"Text-to-speech not available for {lang}")

# Retrieve answer
if st.button("Get Answer") or st.session_state.get('enter_pressed', False):
    if not user_question.strip():
        st.error("⚠️ Please enter a question.")
    else:
        # Match question in selected language column
        col_map = {
            'English': 'Query_English', 'Hindi': 'Query_Hindi', 'Marathi': 'Query_Marathi',
            'Bengali': 'Query_Bengali', 'Tamil': 'Query_Tamil', 'Telugu': 'Query_Telugu'
        }
        lang_col = col_map[selected_lang]
        row = df[df[lang_col].str.lower() == user_question.lower()]
        
        if not row.empty:
            short_ans = row[f"Short_{selected_lang}"].values[0]
            detailed_ans = row[f"Detailed_{selected_lang}"].values[0]

            st.markdown("### Short Answer:")
            st.info(short_ans)
            play_audio(short_ans, selected_lang)

            st.markdown("### Detailed Answer:")
            st.write(detailed_ans)
            st.write("⚖️ **Consult a lawyer nearby – coming soon!**")  # USP line
            play_audio(detailed_ans, selected_lang)
        else:
            st.error("❌ Sorry, we couldn't find an exact answer. Try rephrasing or select another language.")

# Feedback buttons
st.markdown("### Feedback:")
col1, col2 = st.columns(2)
with col1:
    if st.button("👍"):
        st.success("Thanks for your feedback!")
with col2:
    if st.button("👎"):
        st.info("Thanks, we will try to improve!")

# Optional: detect Enter key
st.session_state['enter_pressed'] = False
st.markdown("""
*Press Enter to submit your question (Streamlit may require clicking the input box first).*
""")
