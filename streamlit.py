import streamlit as st
import requests
from io import BytesIO
from PyPDF2 import PdfReader
import google.generativeai as genai
from google.api_core import exceptions as gexc

import streamlit as st
import requests
import google.generativeai as genai
from google.api_core import exceptions as gexc

st.write({"google-generativeai": getattr(genai, "__version__", "unknown")})

API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("Falta GEMINI_API_KEY en st.secrets")
    st.stop()

# 1) Chequeo rápido del formato del key (AI Studio keys suelen empezar con 'AIza')
st.write({"api_key_prefix": API_KEY[:4]})

# 2) Ping REST a /models (no pasa por el SDK)
REST_URL = "https://generativelanguage.googleapis.com/v1beta/models"
r = requests.get(REST_URL, params={"key": API_KEY}, timeout=20)
st.write({"rest_models_status": r.status_code})
st.code(r.text[:1000])  # muestra las primeras ~1000 chars

# 3) Si REST funciona, listar modelos via SDK
try:
    genai.configure(api_key=API_KEY)
    avail = [
        m.name.split("/")[-1]
        for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    ]
    st.write({"available_models_via_sdk": avail})
except Exception as e:
    st.error(f"SDK list_models error: {type(e).__name__}: {e}")



# -------------------------------
# Load and process the CV PDF
# -------------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def extract_cv_text(pdf_url: str) -> str:
    try:
        resp = requests.get(pdf_url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        return f"[ERROR] Could not download CV: {e}"

    with BytesIO(resp.content) as pdf_file:
        reader = PdfReader(pdf_file)
        # PyPDF2 puede devolver None en algunas páginas
        pages_text = []
        for p in reader.pages:
            t = p.extract_text() or ""
            pages_text.append(t)
        text = "\n".join(pages_text)

    # Parse muy simple por secciones (líneas en mayúsculas)
    sections = {}
    lines = text.split("\n")
    current = None
    for line in lines:
        if line.strip() and line.strip().isupper():
            current = line.strip()
            sections[current] = []
        elif current:
            sections[current].append(line.strip())

    for sec in list(sections):
        sections[sec] = " ".join([s for s in sections[sec] if s])

    # Si no detectó secciones, devuelve el texto plano
    if not sections:
        return text

    return "\n".join(f"{sec}:\n{content}" for sec, content in sections.items())

CV_URL = "https://rafaelabastidas.github.io/files/CV.pdf"
cv_text = extract_cv_text(CV_URL)

# -------------------------------
# Configure Gemini API
# -------------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("Falta GEMINI_API_KEY en st.secrets. Agrega en Secrets:\nGEMINI_API_KEY = \"TU_API_KEY\"")
    st.stop()

genai.configure(api_key=API_KEY)

# Permite cambiar el modelo desde secrets
MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-1.5-flash")
GEN_CFG = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1000,
}

def get_model(model_name: str):
    try:
        return genai.GenerativeModel(model_name, generation_config=GEN_CFG)
    except gexc.NotFound:
        # Fallback por si el alias cambió
        return genai.GenerativeModel("gemini-1.5-flash", generation_config=GEN_CFG)

model = get_model(MODEL_NAME)

# Prompt base
cv_prompt = f"""
You are a CV assistant. Answer questions based ONLY on the CV below of Rafaela Bastidas Ripalda.
If the answer is not present, say you don't have that information in the CV.

CV:
{cv_text}
"""

def query_cv_chatbot(question: str) -> str:
    full_prompt = f"{cv_prompt}\n\nQuestion: {question}\nAnswer:"
    try:
        resp = model.generate_content(full_prompt)
        # En versiones recientes puedes usar resp.text directamente
        return (resp.text or "").strip()
    except gexc.NotFound:
        # Si justo el modelo no existe para tu clave, haz fallback duro
        fallback = genai.GenerativeModel("gemini-1.5-flash", generation_config=GEN_CFG)
        resp = fallback.generate_content(full_prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

# -------------------------------
# Streamlit App UI
# -------------------------------
st.markdown("""
<h1 style='text-align: center; font-size: 2.5rem;'>Rafaela Bastidas Ripalda</h1>
<h2 style='text-align: center; color: gray;'>CV Chatbot Assistant</h2>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='text-align: center; font-size: 1.1rem;'>
Welcome! I'm a chatbot powered by Gemini, trained to answer questions about Rafaela's CV.<br>
You can explore her profile by asking your own question, or choose a sample one below.<br><br>
📄 You can also view her CV directly <a href="{CV_URL}" target="_blank">here</a>.
</div>
""", unsafe_allow_html=True)

st.divider()

# Example Questions
with st.expander("Example Questions (click to expand)"):
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
st.markdown("### Ask your own question:")
user_input = st.text_input("Type your question here (you can ask in English or Spanish):")
if user_input:
    st.markdown("**Answer:**")
    st.success(query_cv_chatbot(user_input))
