# AI Resume Analyzer

An AI-powered web app that analyzes how well a resume matches a job description. Upload a resume (PDF) and paste a job description to get a match score, a list of missing skill keywords, and AI-generated suggestions to improve the resume for that specific role.

**Live Demo:** [ai-resume-analyzer-yv2n.vercel.app](https://ai-resume-analyzer-yv2n.vercel.app)

> Note: The backend is hosted on Render's free tier, so the first request after a period of inactivity may take 30-60 seconds to respond while the server wakes up.

---

## Features

- **PDF text extraction** — parses uploaded resumes using `pdfplumber`
- **Match score** — a blended score combining:
  - TF-IDF + cosine similarity (overall content/phrasing overlap between resume and job description)
  - Keyword overlap (percentage of required skills from the job description that are present in the resume)
- **Missing keyword detection** — flags relevant technical skills mentioned in the job description but missing from the resume
- **AI-generated suggestions** — uses Groq's LLaMA 3.3 70B model to generate specific, actionable feedback for improving the resume for the target role

## Why a blended match score?

A naive TF-IDF similarity score alone tends to produce misleadingly low numbers (e.g. 10-20%) even for well-matched resumes, because resumes and job descriptions are structurally different documents with a lot of non-overlapping filler text (names, dates, formatting, unrelated sections). To make the score more meaningful, it's calculated as:

```
Final Score = (40% × TF-IDF similarity) + (60% × keyword overlap %)
```

Keyword overlap is weighted higher because it more directly reflects whether the resume covers the actual skills the job requires.

## Tech Stack

**Frontend**
- React (Vite)
- Deployed on Vercel

**Backend**
- FastAPI (Python)
- `pdfplumber` for PDF text extraction
- `scikit-learn` for TF-IDF vectorization and cosine similarity
- Groq API (LLaMA 3.3 70B) for AI-generated suggestions
- Deployed on Render

## How It Works

1. User uploads a resume PDF and pastes a job description
2. Backend extracts text from the PDF
3. Backend computes:
   - TF-IDF cosine similarity between resume and job description text
   - Keyword overlap using a predefined list of 70+ technical skill keywords
4. A blended match score is calculated and returned along with missing keywords
5. The resume text, job description, and missing keywords are sent to Groq's LLaMA 3.3 70B model to generate 4 specific improvement suggestions
6. Results are displayed on the frontend

## Running Locally

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` folder:

```
GROQ_API_KEY=your_groq_api_key_here
```

Run the server:

```bash
uvicorn main:app --reload
```

The backend will run at `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Create a `.env` file in the `frontend/` folder (optional for local dev, required for production):

```
VITE_API_URL=http://127.0.0.1:8000
```

The frontend will run at `http://localhost:5173`.

## API

### `POST /analyze`

**Request:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `file` | File | Resume PDF |
| `job_description` | String | Job description text |

**Response:**
```json
{
  "match_score": 51.95,
  "missing_keywords": ["vue", "testing", "angular"],
  "suggestions": "1. ...\n2. ...\n3. ...\n4. ...",
  "resume_text_length": 1834
}
```

## Future Improvements

- Replace TF-IDF with embedding-based semantic similarity for more accurate matching
- Support multiple resume formats (DOCX)
- Allow comparing one resume against multiple job descriptions
- Add a downloadable PDF report of the analysis

## Author

Isha Singh
