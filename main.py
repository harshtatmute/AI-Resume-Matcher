# Master Skill List
SKILLS = [
    "python", "c++", "sql", "mysql", "postgresql", "mongodb",
    "aws", "azure", "terraform", "ci/cd", "jenkins",
    "machine learning", "deep learning", "pytorch", "scikit-learn",
    "etl", "spark", "hadoop", "data analysis",
    "agile", "scrum", "jira", "api", "rest",
    "docker", "statistics"
]

def extract_skills(text):
    text = text.lower()
    found_skills = []
   
    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)
    return found_skills


# Step 1: Import Libraries
import re
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
    
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# step 4: Similarity Function
def calculate_similarity(resume_text, job_description_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text,job_description_text])
    similarity = cosine_similarity(vectors[0:1],vectors[1:2])
    return similarity[0][0]

# Step 5: Main Block Execution
if __name__ == "__main__":

    resume_path = "resumes/MachineLearning.pdf"
    jd_path = "jobdesc/ml_engineer.txt"

    resume_text = preprocess_text(extract_text_from_pdf(resume_path))
    job_description_text = preprocess_text(extract_text_from_txt(jd_path))

    score = calculate_similarity(resume_text, job_description_text)

    print(f"\nMatch Score :{round(score * 100, 2)}%")

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_description_text)

    matched_skills = sorted(list(set(resume_skills) & set(jd_skills)))
    missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

    print("\nMatched Skills:")
    print(", ".join(matched_skills) if matched_skills else "None")

    print("\nMissing Skills:")
    print(", ".join(missing_skills) if missing_skills else "None")