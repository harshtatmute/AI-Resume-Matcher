# Step 1: Import Libraries
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Step 2: Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = " "

    for page in reader.pages:
        text += page.extract_text()
    
    return text


# Step 3: Function to Read Job Description (.txt)
def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()
    
# step 4: Similarity Function
def calculate_similarity(resume_text, job_description_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text,job_description_text])
    similarity = cosine_similarity(vectors[0:1],vectors[1:2])
    return similarity[0][0]

# Step 5: Main Block Execution
if __name__ == "__main__":

    resume_path = "resumes/DataBuisnessAnalyst.pdf"
    jd_path = "jobdesc/software_engineer.txt"

    resume_text = extract_text_from_pdf(resume_path)
    job_description_text = extract_text_from_txt(jd_path)

    score = calculate_similarity(resume_text, job_description_text)

    print(f"\nMatch Score :{round(score * 100, 2)}%")