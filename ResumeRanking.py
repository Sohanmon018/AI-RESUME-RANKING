import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip() if text else "No readable text found."

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  # Combine job description with resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    # Scale scores to a max of 10
    scaled_scores = (cosine_similarities / cosine_similarities.max()) * 10
    return scaled_scores

# Streamlit app
st.set_page_config(page_title="AI Resume Screening", page_icon="📄")

# Set background image
def set_bg():
    page_bg_img = '''
    <style>
    .stApp {
        background: url("https://www.google.com/imgres?q=background%20image%20&imgurl=https%3A%2F%2Fplus.unsplash.com%2Fpremium_photo-1701534008693-0eee0632d47a%3Ffm%3Djpg%26q%3D60%26w%3D3000%26ixlib%3Drb-4.1.0%26ixid%3DM3wxMjA3fDB8MHxzZWFyY2h8MXx8bGlnaHQlMjBjb2xvciUyMGJhY2tncm91bmR8ZW58MHx8MHx8fDA%253D&imgrefurl=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Flight-color-background&docid=Sjq70HRaSyrPmM&tbnid=Ep05NyLWfLZ0yM&vet=12ahUKEwi_mMi0zJmOAxUsb_UHHQ_HK2YQM3oECG8QAA..i&w=3000&h=1688&hcb=2&ved=2ahUKEwi_mMi0zJmOAxUsb_UHHQ_HK2YQM3oECG8QAA") no-repeat center fixed;
        background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg()

st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description", key="job_desc")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="resume_upload")

if uploaded_files and job_description:
    st.subheader("Processing Resumes...")
    
    # Extract text from uploaded resumes
    resume_texts = [extract_text_from_pdf(file) for file in uploaded_files]
    resume_names = [file.name for file in uploaded_files]

    # Rank resumes
    similarity_scores = rank_resumes(job_description, resume_texts)

    # Create DataFrame for results
    results_df = pd.DataFrame({
        "Resume Name": resume_names,
        "Score (Out of 10)": similarity_scores
    }).sort_values(by="Score (Out of 10)", ascending=False)

    # Display ranked resumes
    st.subheader("Ranked Resumes")
    st.write(results_df)

    # Plot results
    st.subheader("Ranking Visualization")
    fig, ax = plt.subplots()
    ax.barh(results_df["Resume Name"], results_df["Score (Out of 10)"], color="skyblue")
    ax.set_xlabel("Similarity Score (Out of 10)")
    ax.set_ylabel("Resume Name")
    ax.set_title("Resume Ranking Based on Job Description")
    ax.invert_yaxis()
    st.pyplot(fig)
