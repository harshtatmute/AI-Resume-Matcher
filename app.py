import streamlit as st
from main import (
    extract_text_from_pdf,
    extract_text_from_txt,
    preprocess_text,
    calculate_similarity,
    extract_skills
)

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.title("📄 AI Resume Matcher")
st.write("Upload your resume and paste a job description to analyze the match.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description Here")

if st.button("Analyze"):
    if resume_file and job_description:
        
        # Save uploaded resume temporarily
        with open("temp_resume.pdf", "wb") as f:
            f.write(resume_file.read())
        
        resume_text = preprocess_text(extract_text_from_pdf("temp_resume.pdf"))
        jd_text = preprocess_text(job_description)

        score = calculate_similarity(resume_text, jd_text)

        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)

        matched_skills = sorted(list(set(resume_skills) & set(jd_skills)))
        missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

        st.subheader("📊 Match Score")
        st.write(f"**{round(score * 100, 2)}%**")

        st.subheader("✅ Matched Skills")
        st.write(", ".join(matched_skills) if matched_skills else "None")

        st.subheader("❌ Missing Skills")
        st.write(", ".join(missing_skills) if missing_skills else "None")

    else:
        st.warning("Please upload a resume and paste a job description.")