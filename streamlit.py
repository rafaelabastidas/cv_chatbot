# streamlit.py
import streamlit as st
import requests
from io import BytesIO
from PyPDF2 import PdfReader

# Gemini (SDK) + manejo de errores
import google.generativeai as genai
from google.api_core import exceptions as gexc

# =========================================
# Configuración básica
# =========================================

st.set_page_config(page_title="CV Chatbot - Rafaela Bastidas Ripalda", page_icon="📄", layout="centered")

# Clave desde Streamlit Secrets
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error(
        "Falta GEMINI_API_KEY en Streamlit Secrets.\n\n"
        "Ve a Manage App → Settings → Secrets y agrega:\n\n"
        'GEMINI_API_KEY = "AIza..."\nGEMINI_MODEL = "gemini-1.5-flash"\n'
    )
    st.stop()

# Configura SDK
genai.configure(api_key=API_KEY)

# Modelos “seguros” (evitamos sufijos tipo -002 que a veces no están habilitados)
SAFE_MODELS = [
    (st.secrets.get("GEMINI_MODEL") or "").strip() or "gemini-1.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
]

GEN_CFG = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1000,
}

CV_URL = "https://rafaelabastidas.github.io/files/CV.pdf"


# =========================================
# Utilidades
# =========================================

@st.cache_data(show_spinner=False, ttl=60 * 60)
def extract_cv_text(pdf_url: str) -> str:
    """Descarga y extrae texto del CV en PDF. Si no detecta secciones, devuelve texto plano."""
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

    # Heurística simple para secciones: líneas en MAYÚSCULAS
    sections = {}
    lines = text.split("\n")
    current = None
    for line in lines:
        if line.strip() and line.strip().isupper():
            current = line.strip()
            sections[current] = []
        elif current:
            sections[current].append(line.strip())

    # Si detectó secciones, las arma; si no, devuelve texto plano
    if sections:
        for sec in list(sections.keys()):
            sections[sec] = " ".join([s for s in sections[sec] if s])
        return "\n".join(f"{sec}:\n{content}" for sec, content in sections.items())
    else:
        return text


def _pick_available_model() -> str:
    """Intenta elegir el primer modelo de SAFE_MODELS que esté disponible vía SDK."""
    try:
        avail = [
            m.name.split("/")[-1]
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        for m in SAFE_MODELS:
            if m in avail:
                return m
    except Exception:
        pass
    return "gemini-1.5-flash"


# Modelo inicial vía SDK
_MODEL_NAME = _pick_available_model()
_model = genai.GenerativeModel(_MODEL_NAME, generation_config=GEN_CFG)


def ask_gemini(prompt: str) -> str:
    """
    1) Intenta con SDK y el modelo elegido.
    2) Si NotFound, rota por los otros SAFE_MODELS (SDK).
    3) Si todo falla, usa REST con modelos seguros.
    """
    # 1) Intento con el modelo seleccionado
    try:
        r = _model.generate_content(prompt)
        return (getattr(r, "text", "") or "").strip()
    except gexc.NotFound:
        # 2) Rotar otros modelos con SDK
        for alt in SAFE_MODELS:
            if alt == _MODEL_NAME:
                continue
            try:
                r = genai.GenerativeModel(alt, generation_config=GEN_CFG).generate_content(prompt)
                return (getattr(r, "text", "") or "").strip()
            except gexc.NotFound:
                continue
            except Exception as e:
                return f"[ERROR SDK {alt}] {type(e).__name__}: {e}"
        # 3) Último recurso: REST
        last_err = None
        for alt in ["gemini-1.5-flash", "gemini-1.5-pro-latest", "gemini-1.5-pro"]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{alt}:generateContent"
            payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": GEN_CFG}
            rr = requests.post(url, params={"key": API_KEY}, json=payload, timeout=60)
            if rr.ok:
                data = rr.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except Exception:
                    return str(data)[:1000]
            else:
                if rr.status_code == 404:
                    last_err = f"{alt}: 404 NOT_FOUND"
                    continue
                return f"[ERROR REST {alt}] {rr.status_code}: {rr.text[:300]}"
        return f"[ERROR] No working model (SDK y REST). Último error: {last_err or 'desconocido'}"
    except Exception as e:
        return f"[ERROR SDK {_MODEL_NAME}] {type(e).__name__}: {e}"


# =========================================
# Carga CV y prompt base
# =========================================

cv_text = extract_cv_text(CV_URL)

cv_prompt = f"""
You are a CV assistant. Answer questions based ONLY on the CV below of Rafaela Bastidas Ripalda.
If the answer is not present, say you don't have that information in the CV.

CV:
{cv_text}
""".strip()


def query_cv_chatbot(question: str) -> str:
    full_prompt = f"{cv_prompt}\n\nQuestion: {question}\nAnswer:"
    return ask_gemini(full_prompt)


# =========================================
# UI
# =========================================

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
