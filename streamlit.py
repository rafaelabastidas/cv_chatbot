
#cd "C:/Users/RAFAELAB/OneDrive - Inter-American Development Bank Group/Documents/PBLs/Depth/Code"
#streamlit run .\streamlit.py
import streamlit as st
import requests
import os
from io import BytesIO
from PyPDF2 import PdfReader
import google.generativeai as genai

# Step 1: Download and extract the CV text
url = "https://rafaelabastidas.github.io/files/CV.pdf"
response = requests.get(url)

with BytesIO(response.content) as pdf_file:
    reader = PdfReader(pdf_file)
    cv_text = ""
    for page in reader.pages:
        cv_text += page.extract_text()

# Structure the CV text into sections
sections = {}
lines = cv_text.split('\n')

current_section = None
for line in lines:
    if line.isupper():  # Assuming section headers are in uppercase
        current_section = line
        sections[current_section] = []
    elif current_section:
        sections[current_section].append(line)

# Convert sections to a readable format
for section, content in sections.items():
    sections[section] = " ".join(content)

cv_doc = "\n".join(
    f"{section}:\n{content}" for section, content in sections.items()
)

# Step 2: Configure the generative AI model
api_key = os.getenv('API_KEY')
if api_key is None:
    raise ValueError("API_KEY not found. Please set the environment variable.")
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1000,
}

model = genai.GenerativeModel(model_name='gemini-1.5-flash', generation_config=generation_config)

cv_prompt = f"""
You are a CV assistant. You will answer questions based on the CV provided below.

{cv_doc}
"""

def query_cv_chatbot(question):
    prompt = f"Here is a CV. Answer the following question based on the CV:\n\n{cv_prompt}\n\nQuestion: {question}\nAnswer:"
    response = model.generate_content(prompt)  # Request a response from the model
    generated_answer = response.candidates[0].content.parts[0].text.strip()
    return generated_answer

# Step 3: Create the Streamlit app
st.title("CV Chatbot: Rafaela Bastidas Ripalda 🌟")
st.markdown("""
Hello! 👋 I'm the personal chatbot of **Rafaela Bastidas Ripalda**, an economist and data scientist passionate about analyzing and solving complex problems.  
📄 Feel free to ask me anything about her experience, skills, or professional background, and I'll be happy to assist you.  
If you'd like to take a closer look at her CV, you can find it [here](https://rafaelabastidas.github.io/files/CV.pdf).  
Let's get started!
""")


# Input box for user questions
question = st.text_input("Ask a question about the CV:")

# Display the answer if a question is provided
if question:
    with st.spinner("Generating answer..."):
        answer = query_cv_chatbot(question)
    st.write("Answer:")
    st.write(answer)
