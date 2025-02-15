import streamlit as st
import fitz  # PyMuPDF for PDF parsing
import docx
import os
import pandas as pd
import zipfile
import tempfile
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text("text") + "\n" for page in doc]).strip()

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

# Function to extract resumes from a ZIP file
def extract_resumes_from_zip(zip_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    
    resume_texts = {}
    for root, _, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = file.split(".")[-1].lower()
            
            with open(file_path, "rb") as f:
                if file_extension == "pdf":
                    resume_texts[file] = extract_text_from_pdf(f)
                elif file_extension == "docx":
                    resume_texts[file] = extract_text_from_docx(f)
    
    return resume_texts

# Function to calculate similarity
def calculate_similarity(resume_texts, jd_text):
    if not jd_text or not resume_texts:
        return {}

    embeddings = OpenAIEmbeddings()
    jd_vector = embeddings.embed_query(jd_text)

    scores = {
        filename: round(cosine_similarity([embeddings.embed_query(text)], [jd_vector])[0][0] * 100, 2)
        for filename, text in resume_texts.items()
    }
    return scores

# Streamlit UI
st.title("📂 AI Resume Scanner with Folder Upload Support")
st.write("Upload multiple resumes or a ZIP folder and enter a Job Description to find the best matches.")

# Upload multiple resumes manually
uploaded_files = st.file_uploader("📤 Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

# Upload a ZIP folder containing multiple resumes
uploaded_zip = st.file_uploader("📦 Upload a ZIP folder containing resumes", type=["zip"])

job_description = st.text_area("📌 Paste the Job Description (JD)")

# Process uploaded files
resume_texts = {}

if uploaded_files:
    with st.spinner("Processing uploaded resumes..."):
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_extension = file_name.split(".")[-1].lower()

            if file_extension == "pdf":
                resume_texts[file_name] = extract_text_from_pdf(uploaded_file)
            elif file_extension == "docx":
                resume_texts[file_name] = extract_text_from_docx(uploaded_file)

elif uploaded_zip:
    with st.spinner("Extracting resumes from ZIP file..."):
        resume_texts = extract_resumes_from_zip(uploaded_zip)

# Compute similarity
if resume_texts and job_description:
    with st.spinner("Matching resumes..."):
        match_scores = calculate_similarity(resume_texts, job_description)

        if match_scores:
            results_df = pd.DataFrame(match_scores.items(), columns=["Resume", "Match Score"]).sort_values(by="Match Score", ascending=False)
            st.success("✅ Processing Complete!")
            st.dataframe(results_df)

            csv_data = results_df.to_csv(index=False)
            st.download_button(label="📥 Download Shortlist", data=csv_data, file_name="shortlisted_resumes.csv", mime="text/csv")
        else:
            st.warning("⚠️ No valid resumes found!")

st.sidebar.write("📌 Upload resumes in **PDF or DOCX format** or as a **ZIP folder** containing multiple resumes.")

