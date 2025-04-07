import streamlit as st
import requests
import os
from io import BytesIO
from PyPDF2 import PdfReader
import google.generativeai as genai

# -------------------------------
# Load and process the CV PDF
# -------------------------------
@st.cache_data(show_spinner=False)
def extract_cv_text(pdf_url):
    response = requests.get(pdf_url)
    with BytesIO(response.content) as pdf_file:
        reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in reader.pages)

    sections = {}
    lines = text.split('\n')
    current_section = None

    for line in lines:
        if line.isupper():
            current_section = line.strip()
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line.strip())

    for section in sections:
        sections[section] = " ".join(sections[section])

    return "\n".join(f"{sec}:\n{content}" for sec, content in sections.items())

cv_text = extract_cv_text("https://rafaelabastidas.github.io/files/CV.pdf")

# -------------------------------
# Configure Gemini API
# -------------------------------
api_key = os.getenv("API_KEY")
if not api_key:
    st.error("API_KEY not found. Please set it as an environment variable in Streamlit Cloud settings.")
    st.stop()

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 1000,
    },
)

cv_prompt = f"""
You are a CV assistant. You will answer questions based on the CV provided below that belongs to Rafaela Bastidas Ripalda.

{cv_text}
"""

def query_cv_chatbot(question):
    full_prompt = f"{cv_prompt}\n\nQuestion: {question}\nAnswer:"
    response = model.generate_content(full_prompt)
    return response.candidates[0].content.parts[0].text.strip()


# -------------------------------
# Streamlit App UI
# -------------------------------
st.markdown("""
<h1 style='text-align: center; font-size: 2.5rem;'>📄 Rafaela Bastidas Ripalda</h1>
<h2 style='text-align: center; color: gray;'>CV Chatbot Assistant</h2>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 1.1rem;'>
👋 Welcome! I'm a chatbot powered by Gemini, trained to answer questions about Rafaela's CV.<br>
You can explore her profile by asking your own question, or choose a sample one below.<br><br>
📄 You can also view her CV directly <a href="https://rafaelabastidas.github.io/files/CV.pdf" target="_blank">here</a>.
</div>
""", unsafe_allow_html=True)

st.divider()

# Example Questions
with st.expander("💡 Example Questions (click to expand)"):
    example_question = st.selectbox(
        "Choose one to get started:",
        [
            "Select an example question...",
            "What is Rafaela's educational background?",
            "What are Rafaela's key skills in data science?",
            "What programming languages does Rafaela know?",
        ]
    )

    if example_question != "Select an example question...":
        st.markdown("**Answer:**")
        st.success(query_cv_chatbot(example_question))

# Custom Question
st.markdown("### ✏️ Ask your own question:")
user_input = st.text_input("Type your question here (you can ask in English or Spanish):")

if user_input:
    st.markdown("**Answer:**")
    st.success(query_cv_chatbot(user_input))
