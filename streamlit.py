# streamlit.py
import streamlit as st
import requests
from io import BytesIO
from PyPDF2 import PdfReader

# ===============================
# Configuración
# ===============================
st.set_page_config(page_title="CV Chatbot - Rafaela Bastidas Ripalda", page_icon="📄", layout="centered")

API_KEY = st.secrets.get("GEMINI_API_KEY", "").strip()
if not API_KEY:
    st.error('Falta GEMINI_API_KEY en Secrets. Agrega: GEMINI_API_KEY = "AIza..."')
    st.stop()

MODEL_NAME = (st.secrets.get("GEMINI_MODEL") or "gemini-2.5-flash-lite").strip()
GEN_CFG = {"temperature": 0, "top_p": 1, "top_k": 1, "max_output_tokens": 512}
CV_URL = "https://rafaelabastidas.github.io/files/CV.pdf"

# ===============================
# Utilidades
# ===============================
@st.cache_data(show_spinner=False, ttl=60*60)
def extract_cv_text(pdf_url: str) -> str:
    try:
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return f"[ERROR] No se pudo descargar el CV: {e}"

    with BytesIO(resp.content) as f:
        reader = PdfReader(f)
        text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()

    if not text:
        return "[ADVERTENCIA] No se pudo extraer texto del PDF."

    # Heurística simple de secciones: líneas en MAYÚSCULAS
    sections, current = {}, None
    for line in text.split("\n"):
        s = line.strip()
        if s and s.isupper():
            current = s
            sections[current] = []
        elif current:
            sections[current].append(s)

    if sections:
        for sec in list(sections):
            sections[sec] = " ".join([s for s in sections[sec] if s])
        return "\n".join(f"{sec}:\n{content}" for sec, content in sections.items())
    return text

def gemini_generate(prompt: str, model_name: str = None) -> str:
    """Llamada REST sencilla con mensajes de error precisos."""
    model = (model_name or MODEL_NAME).strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": GEN_CFG}

    try:
        r = requests.post(url, params={"key": API_KEY}, json=payload, timeout=60)
    except requests.RequestException as e:
        return f"[ERROR DE RED] {e}"

    if not r.ok:
        # Intento extraer un error legible
        try:
            err = r.json().get("error", {})
            code, status, msg = err.get("code"), err.get("status"), err.get("message", "")
        except Exception:
            return f"[ERROR HTTP {r.status_code}] {r.text[:400]}"

        if code == 404:
            return (f"[ERROR 404] El modelo '{model}' no está disponible para tu clave. "
                    "Cambia GEMINI_MODEL (p.ej., gemini-2.5-flash-lite o gemini-2.5-flash).")
        if code == 429:
            return ("[CUOTA EXCEDIDA / 429] Tu API key no tiene free tier activo o se agotó. "
                    "Usa otra clave de AI Studio o activa billing en GCP.")
        return f"[ERROR {code} {status}] {msg}"

    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return str(data)[:1000]

# ===============================
# Carga CV y prompt
# ===============================
cv_text_full = extract_cv_text(CV_URL)
MAX_CHARS = 12000  # recorta para evitar 429 por tokens
cv_text = cv_text_full[:MAX_CHARS]

cv_prompt = f"""
You are a CV assistant. Answer strictly using the CV below of Rafaela Bastidas Ripalda.
If the answer is not present in the CV, say you don't have that information in the CV.

CV:
{cv_text}
""".strip()

def query_cv_chatbot(question: str) -> str:
    full_prompt = f"{cv_prompt}\n\nQuestion: {question}\nAnswer:"
    return gemini_generate(full_prompt)

# ===============================
# UI mínima
# ===============================
st.markdown(
    """
<h1 style='text-align: center; font-size: 2.2rem;'>Rafaela Bastidas Ripalda</h1>
<h2 style='text-align: center; color: gray;'>CV Chatbot Assistant</h2>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div style='text-align: center;'>
Ask about Rafaela's CV or open it directly
<a href="{CV_URL}" target="_blank">here</a>.
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

with st.expander("Example Questions"):
    example = st.selectbox(
        "Choose one:",
        [
            "Select an example…",
            "What is Rafaela's educational background?",
            "What are Rafaela's key skills in data science?",
            "What programming languages does Rafaela know?",
        ],
    )
    if example != "Select an example…":
        st.markdown("**Answer:**")
        st.success(query_cv_chatbot(example))

st.markdown("### Ask your own question")
user_q = st.text_input("Type your question (English or Spanish):")
if user_q:
    st.markdown("**Answer:**")
    st.success(query_cv_chatbot(user_q))
