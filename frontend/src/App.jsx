import { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [jobDescription, setJobDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const handleSubmit = async () => {
    if (!file) {
      setError('Please upload your resume PDF.')
      return
    }
    if (!jobDescription.trim()) {
      setError('Please paste a job description.')
      return
    }

    setError('')
    setLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('job_description', jobDescription)

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Server error. Please try again.')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError('Something went wrong. Make sure the backend server is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>AI Resume Analyzer</h1>
      <p className="subtitle">Upload your resume and a job description to get your match score and improvement suggestions.</p>

      <div className="upload-section">
        <label className="label">Resume (PDF)</label>
        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <label className="label">Job Description</label>
        <textarea
          rows={8}
          placeholder="Paste the job description here..."
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
        />

        <button onClick={handleSubmit} disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Resume'}
        </button>

        {error && <p className="error">{error}</p>}
      </div>

      {result && (
        <div className="result-section">
          <h2>Match Score: {result.match_score}%</h2>

          <h3>Missing Keywords</h3>
          {result.missing_keywords.length > 0 ? (
            <div className="keyword-tags">
              {result.missing_keywords.map((kw, i) => (
                <span key={i} className="tag">{kw}</span>
              ))}
            </div>
          ) : (
            <p>No major keywords missing. Great job!</p>
          )}

          <h3>Suggestions</h3>
          <div className="suggestions">
            {result.suggestions}
          </div>
        </div>
      )}
    </div>
  )
}

export default App