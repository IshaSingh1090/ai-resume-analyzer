from fastapi import FastAPI, UploadFile, File, Form
import pdfplumber
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI()

# Load the AI model once when server starts (takes a few seconds first time)
model = SentenceTransformer('all-MiniLM-L6-v2')


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

# A list of real technical skills/tools to check for
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

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    contents = await file.read()
    resume_text = extract_text_from_pdf(contents)

    # Generate embeddings for both texts
    resume_embedding = model.encode([resume_text])
    jd_embedding = model.encode([job_description])

    # Calculate similarity score (0 to 1, we convert to percentage)
    similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
    match_score = round(float(similarity) * 100, 2)

    # Find missing keywords
    resume_keywords = get_keywords(resume_text)
    jd_keywords = get_keywords(job_description)
    missing_keywords = list(jd_keywords - resume_keywords)

    return {
        "match_score": match_score,
        "missing_keywords": missing_keywords[:20],  # limit to top 20
        "resume_text_length": len(resume_text)
    }