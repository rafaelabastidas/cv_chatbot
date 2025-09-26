# streamlit.py
import streamlit as st
import requests
from io import BytesIO
from PyPDF2 import PdfReader

# SDK se usa solo para listar modelos (no para generar)
import google.generativeai as genai

# ===============================
# Config y constantes
# ===============================
st.set_page_config(page_title="CV Chatbot - Rafaela Bastidas Ripalda", page_icon="📄", layout="centered")

API_KEY = st.secrets.get("GEMINI_API_KEY", "").strip()

# Modelos con buena probabilidad de free tier activo (orden de preferencia)
CANDIDATE_MODELS = [
    (st.secrets.get("GEMINI_MODEL") or "").strip() or "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# Config de generación (conservadora para ahorrar cuota)
GEN_CFG = {"temperature": 0, "top_p": 1, "top_k": 1, "max_output_tokens": 512}

# CV fuente
CV_URL = "https://rafaelabastidas.github.io/files/CV.pdf"

# ===============================
# Utilidades: CV
# ===============================
@st.cache_data(show_spinner=False, ttl=60 * 60)
def extract_cv_text(pdf_url: str) -> str:
    """Descarga y extrae texto del CV (PDF). Si no detecta secciones, devuelve texto plano."""
    try:
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return f"[ERROR] No se pudo descargar el CV: {e}"

    with BytesIO(resp.content) as pdf_file:
        reader = PdfReader(pdf_file)
        pages_text = []
        for p in reader.pages:
            t = p.extract_text() or ""
            pages_text.append(t)
        text = "\n".join(pages_text).strip()

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
        for sec in list(sections.keys()):
            sections[sec] = " ".join([s for s in sections[sec] if s])
        return "\n".join(f"{sec}:\n{content}" for sec, content in sections.items())
    return text

cv_text_full = extract_cv_text(CV_URL)

# Reducimos tamaño para ahorrar tokens (ajusta si necesitas más contexto)
MAX_CHARS = 12000
cv_text = cv_text_full[:MAX_CHARS]

cv_prompt = f"""
You are a CV assistant. Answer strictly using the CV below of Rafaela Bastidas Ripalda.
If the answer is not present in the CV, say you don't have that information in the CV.

CV:
{cv_text}
""".strip()

# ===============================
# Utilidades: Gemini (REST)
# ===============================
def _probe_rest(model_name: str):
    """Hace una llamada mínima (ping) por REST para verificar acceso/cuota."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {"contents": [{"parts": [{"text": "ping"}]}], "generationConfig": GEN_CFG}
    r = requests.post(url, params={"key": API_KEY}, json=payload, timeout=20)
    ok = r.ok
    code = r.status_code
    try:
        detail = r.json().get("error", {}).get("message", "")[:200]
    except Exception:
        detail = r.text[:200]
    return ok, code, detail

def pick_working_model():
    """Intenta listar por SDK para filtrar y luego hace ping REST a candidatos; devuelve tabla diag y el elegido."""
    try:
        genai.configure(api_key=API_KEY)
        available = {
            m.name.split("/")[-1]
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        }
    except Exception:
        available = set()

    diag = []
    chosen = None
    for m in CANDIDATE_MODELS:
        if available and (m not in available):
            diag.append({"model": m, "status": "skip (no aparece en list_models)", "http": "-", "detail": "-"})
            continue
        ok, code, detail = _probe_rest(m)
        diag.append({"model": m, "status": "OK" if ok else code, "http": code, "detail": detail})
        if ok and chosen is None:
            chosen = m
    return diag, chosen

with st.expander("Diagnóstico modelos (free tier)"):
    st.caption("Probando modelos con una llamada mínima para detectar uno utilizable con tu API key…")
    diag, WORKING_MODEL = pick_working_model()
    st.dataframe(diag, use_container_width=True)
    if not WORKING_MODEL:
        st.error(
            "Ningún modelo respondió OK. Tu API key no tiene cuota (429) o acceso (404).\n"
            "Soluciones: crea una nueva API key en Google AI Studio o activa billing en GCP."
        )
        st.stop()
    else:
        st.success(f"Usando modelo: {WORKING_MODEL}")

def gemini_generate(prompt: str, model_name: str = None) -> str:
    """Generación por REST con manejo claro de errores/cuota."""
    model_name = model_name or WORKING_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": GEN_CFG}
    r = requests.post(url, params={"key": API_KEY}, json=payload, timeout=60)
    if not r.ok:
        try:
            err = r.json().get("error", {})
            code = err.get("code")
            status = err.get("status")
            msg = err.get("message")
            if code == 429:
                return (
                    "[CUOTA EXCEDIDA / 429]\n"
                    "Tu API key no tiene free tier activo o se agotó.\n"
                    "Usa otra clave de AI Studio o activa billing en GCP.\n"
                    f"Detalle: {status}: {msg}"
                )
            if code == 404:
                return (
                    f"[ERROR 404] Modelo '{model_name}' no disponible para tu clave.\n"
                    "Cambia GEMINI_MODEL en Secrets a uno de los mostrados como OK en el diagnóstico."
                )
            return f"[ERROR REST {model_name}] {code} {status}: {msg}"
        except Exception:
            return f"[ERROR REST {model_name}] {r.status_code}: {r.text[:400]}"
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return str(data)[:1000]

# ===============================
# Lógica de preguntas/respuestas
# ===============================
def query_cv_chatbot(question: str) -> str:
    full_prompt = f"{cv_prompt}\n\nQuestion: {question}\nAnswer:"
    return gemini_generate(full_prompt)

# ===============================
# UI
# ===============================
st.markdown(
    """
<h1 style='text-align: center; font-size: 2.5rem;'>Rafaela Bastidas Ripalda</h1>
<h2 style='text-align: center; color: gray;'>CV Chatbot Assistant</h2>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div style='text-align: center; font-size: 1.1rem;'>
Welcome! I'm a chatbot powered by Gemini, trained to answer questions about Rafaela's CV.<br>
You can explore her profile by asking your own question, or choose a sample one below.<br><br>
📄 You can also view her CV directly <a href="{CV_URL}" target="_blank">here</a>.
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

with st.expander("Example Questions (click to expand)"):
    example_question = st.selectbox(
        "Choose one to get started:",
        [
            "Select an example question...",
            "What is Rafaela's educational background?",
            "What are Rafaela's key skills in data science?",
            "What programming languages does Rafaela know?",
        ],
    )
    if example_question != "Select an example question...":
        st.markdown("**Answer:**")
        st.success(query_cv_chatbot(example_question))

st.markdown("### Ask your own question:")
user_input = st.text_input("Type your question here (you can ask in English or Spanish):")
if user_input:
    st.markdown("**Answer:**")
    st.success(query_cv_chatbot(user_input))
