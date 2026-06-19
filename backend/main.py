from fastapi import FastAPI, UploadFile, File, Form
import pdfplumber
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@app.get("/")
def home():
    return {"message": "Backend is running!"}


def extract_text_from_pdf(file_bytes):
    pdf_file = io.BytesIO(file_bytes)
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
    return extracted_text


SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "react", "react.js",
    "node", "node.js", "express", "express.js", "django", "flask", "fastapi",
    "mongodb", "mysql", "postgresql", "sql", "nosql", "redis", "firebase",
    "html", "html5", "css", "css3", "tailwind", "bootstrap", "sass",
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "git", "github",
    "rest", "rest api", "graphql", "api", "microservices",
    "tensorflow", "keras", "pytorch", "scikit-learn", "pandas", "numpy",
    "opencv", "nlp", "machine learning", "deep learning", "cnn", "llm",
    "langchain", "rag", "vector database", "embeddings",
    "agile", "scrum", "jira", "postman", "vs code", "linux",
    "jwt", "oauth", "authentication", "websocket", "redux", "next.js",
    "vue", "angular", "spring boot", "data structures", "algorithms",
    "dbms", "oop", "system design", "testing", "unit testing"
]


def get_keywords(text):
    text_lower = text.lower()
    found = set()
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            found.add(skill)
    return found


def get_llm_suggestions(resume_text, job_description, missing_keywords):
    keywords_str = ", ".join(missing_keywords) if missing_keywords else "none"

    prompt = f"""You are an expert resume reviewer helping a student improve their resume for ATS systems.

RESUME TEXT:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:1500]}

MISSING KEYWORDS (skills mentioned in JD but not in resume): {keywords_str}

Give exactly 4 short, specific, actionable suggestions to improve this resume for this job.
Each suggestion should be 1-2 sentences. Focus on: missing skills, weak bullet points, and ATS optimization.
Format as a numbered list, nothing else before or after."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content


@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    contents = await file.read()
    resume_text = extract_text_from_pdf(contents)

    resume_embedding = model.encode([resume_text])
    jd_embedding = model.encode([job_description])

    similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
    match_score = round(float(similarity) * 100, 2)

    resume_keywords = get_keywords(resume_text)
    jd_keywords = get_keywords(job_description)
    missing_keywords = list(jd_keywords - resume_keywords)

    suggestions = get_llm_suggestions(resume_text, job_description, missing_keywords)

    return {
        "match_score": match_score,
        "missing_keywords": missing_keywords[:20],
        "suggestions": suggestions,
        "resume_text_length": len(resume_text)
    }