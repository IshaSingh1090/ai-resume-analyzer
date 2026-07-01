from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI(title="AI Resume Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


@app.get("/")
def home():
    return {"message": "Backend is running!"}


def extract_text_from_pdf(file_bytes: bytes) -> str:
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


def get_keywords(text: str) -> set:
    text_lower = text.lower()
    return {skill for skill in SKILL_KEYWORDS if skill in text_lower}


def get_llm_suggestions(resume_text, job_description, missing_keywords):
    if groq_client is None:
        return "AI suggestions unavailable: GROQ_API_KEY not set."

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

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI suggestions unavailable: {str(e)}"


@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    contents = await file.read()
    resume_text = extract_text_from_pdf(contents)

    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF. Try a different file.")

    # TF-IDF similarity (captures general content/phrasing overlap)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Keyword overlap (captures actual skill matching)
    resume_keywords = get_keywords(resume_text)
    jd_keywords = get_keywords(job_description)
    missing_keywords = list(jd_keywords - resume_keywords)

    if jd_keywords:
        keyword_overlap = len(jd_keywords & resume_keywords) / len(jd_keywords)
    else:
        keyword_overlap = tfidf_similarity  # fallback if JD has no recognized skill keywords

    # Blended score: keyword match weighted higher than raw text similarity
    final_score = (0.4 * tfidf_similarity) + (0.6 * keyword_overlap)
    match_score = round(float(final_score) * 100, 2)

    suggestions = get_llm_suggestions(resume_text, job_description, missing_keywords)

    return {
        "match_score": match_score,
        "missing_keywords": missing_keywords[:20],
        "suggestions": suggestions,
        "resume_text_length": len(resume_text)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)