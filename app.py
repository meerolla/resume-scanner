import streamlit as st
import fitz  # PyMuPDF for PDF parsing
import docx
import os
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text("text") + "\n" for page in doc])
    return text.strip()

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

# Function to process resumes in a folder
def process_resumes(folder_path):
    resume_texts = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(".pdf"):
            resume_texts[filename] = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(".docx"):
            resume_texts[filename] = extract_text_from_docx(file_path)
    return resume_texts

# Function to compute similarity scores
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
st.title("📂 Bulk Resume Scanner with LangChain & AI")
st.write("Upload resumes in a folder and enter a Job Description to find the best matches.")

folder_path = st.text_input("Enter Resume Folder Path", value="resumes/")
job_description = st.text_area("Paste the Job Description (JD)")

if st.button("Process Resumes"):
    if not os.path.exists(folder_path):
        st.error("❌ Invalid folder path! Please check and try again.")
    else:
        with st.spinner("Processing resumes..."):
            resume_texts = process_resumes(folder_path)
            match_scores = calculate_similarity(resume_texts, job_description)

            if match_scores:
                results_df = pd.DataFrame(match_scores.items(), columns=["Resume", "Match Score"]).sort_values(by="Match Score", ascending=False)
                st.success("✅ Processing Complete!")
                import ace_tools as tools; tools.display_dataframe_to_user(name="Resume Match Results", dataframe=results_df)

                csv_data = results_df.to_csv(index=False)
                st.download_button(label="📥 Download Shortlist", data=csv_data, file_name="shortlisted_resumes.csv", mime="text/csv")
            else:
                st.warning("⚠️ No valid resumes found in the folder!")

st.sidebar.write("📌 Ensure your resumes are in PDF or DOCX format inside the folder.")

