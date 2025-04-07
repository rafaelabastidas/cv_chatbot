
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
You are a CV assistant. You will answer questions based on the CV provided below that belongs to Rafaela Bastidas Ripalda.

{cv_doc}
"""

def query_cv_chatbot(question):
    prompt = f"Here is a CV. Answer the following question based on the CV:\n\n{cv_prompt}\n\nQuestion: {question}\nAnswer:"
    response = model.generate_content(prompt)  # Request a response from the model
    generated_answer = response.candidates[0].content.parts[0].text.strip()
    return generated_answer

# Step 3: Create the Streamlit app
st.title("CV Chatbot: Rafaela Bastidas Ripalda")
st.markdown("""
Hello! 👋 I'm the personal chatbot of **Rafaela Bastidas Ripalda**.
Feel free to ask me anything about her experience, skills, or professional background, and I'll be happy to assist you.  
If you'd like to take a closer look at her CV, you can find it [here](https://rafaelabastidas.github.io/files/CV.pdf).  
""")


# Dropdown menu for example questions
#st.markdown("### Example Questions:")
example_question = st.selectbox(
    #"Select a question to see the answer:",
    "",
    [
        "Select an example question...",
        "What is Rafaela's educational background?",
        "What are Rafaela's key skills in data science?",
        "What programming languages does Rafaela know?",
    ]
)

# Display the answer to the selected question
if example_question != "Select an example question...":
    answer = query_cv_chatbot(example_question)
    st.markdown(f"**Answer:** {answer}")

# Input box for custom user questions
st.markdown("### Ask your own question:")
custom_question = st.text_input("Ask a question about Rafaela's CV (if you prefer to ask a question in spanish, please feel free to do so):")

# Process the user question
if custom_question:
    custom_answer = query_cv_chatbot(custom_question)
    st.markdown(f"**Answer:** {custom_answer}")



