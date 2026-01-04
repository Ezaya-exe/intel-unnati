#!/usr/bin/env python3
"""
NCERT Doubt Solver - FastAPI REST API
Provides HTTP endpoints for mobile apps and external integrations.

Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import sys
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.doubt_solver import NCERTDoubtSolver, detect_language
from core.feedback import save_feedback, get_feedback_stats

# Initialize FastAPI app
app = FastAPI(
    title="NCERT Doubt Solver API",
    description="Multilingual RAG-based doubt solver for NCERT textbooks (Grades 5-10)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global solver instance
solver: Optional[NCERTDoubtSolver] = None


# Request/Response Models
class QuestionRequest(BaseModel):
    """Request body for asking a question"""
    question: str = Field(..., description="The student's question", min_length=3, max_length=1000)
    grade: Optional[int] = Field(None, description="Grade level (5-10)", ge=5, le=10)
    subject: Optional[str] = Field(None, description="Subject filter (Mathematics, Science, etc.)")
    language: Optional[str] = Field(None, description="Response language (Hindi, English, or auto)")
    n_results: int = Field(5, description="Number of context chunks to retrieve", ge=1, le=10)


class Citation(BaseModel):
    """Citation information"""
    index: int
    text: str
    grade: int
    subject: str
    language: str
    page: Optional[str] = None


class AnswerResponse(BaseModel):
    """Response body with the answer"""
    question_id: str
    question: str
    answer: str
    language: str
    citations: List[Citation]
    in_scope: bool
    latency_ms: float
    timestamp: str


class FeedbackRequest(BaseModel):
    """Request body for submitting feedback"""
    question_id: str = Field(..., description="ID of the question being rated")
    rating: str = Field(..., description="Rating: 'thumbs_up' or 'thumbs_down'")
    comment: Optional[str] = Field(None, description="Optional feedback comment")


class FeedbackResponse(BaseModel):
    """Response body for feedback submission"""
    success: bool
    message: str


class StatsResponse(BaseModel):
    """Response body for statistics"""
    total_chunks: int
    grades: Dict[str, int]
    subjects: Dict[str, int]
    languages: Dict[str, int]
    feedback_stats: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    vector_db_ready: bool
    model_loaded: bool


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the solver on startup"""
    global solver
    print("🚀 Initializing NCERT Doubt Solver API...")
    try:
        solver = NCERTDoubtSolver(use_wsl_server=False)
        print("✅ API Ready!")
    except Exception as e:
        print(f"❌ Failed to initialize solver: {e}")


# Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_db_ready=solver is not None,
        model_loaded=solver is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if solver else "initializing",
        version="1.0.0",
        vector_db_ready=solver is not None,
        model_loaded=solver is not None
    )


@app.post("/api/chat", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer from NCERT textbooks.
    
    Returns the answer with citations from the relevant textbook sections.
    """
    global solver
    
    if solver is None:
        raise HTTPException(status_code=503, detail="Solver not initialized")
    
    question_id = str(uuid.uuid4())[:12]
    start_time = time.time()
    
    try:
        # Process the question
        result = solver.ask(
            question=request.question,
            grade=request.grade,
            subject=request.subject,
            language=request.language,
            n_context_chunks=request.n_results
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Format citations
        citations = []
        for cite in result.get('citations', []):
            citations.append(Citation(
                index=cite.get('index', 0),
                text=cite.get('text', ''),
                grade=cite.get('grade', 0),
                subject=cite.get('subject', 'Unknown'),
                language=cite.get('language', 'Unknown'),
                page=str(cite.get('page', ''))
            ))
        
        return AnswerResponse(
            question_id=question_id,
            question=request.question,
            answer=result.get('answer', ''),
            language=result.get('language', 'en'),
            citations=citations,
            in_scope=result.get('in_scope', True),
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for an answer.
    
    Helps improve the system by tracking answer quality.
    """
    try:
        is_helpful = request.rating == "thumbs_up"
        
        save_feedback(
            question_id=request.question_id,
            question="",  # We don't store the question for privacy
            answer="",
            is_helpful=is_helpful,
            latency=0.0
        )
        
        return FeedbackResponse(
            success=True,
            message=f"Thank you for your feedback! ({'👍' if is_helpful else '👎'})"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def get_statistics():
    """
    Get system statistics including database info and feedback metrics.
    """
    global solver
    
    if solver is None:
        raise HTTPException(status_code=503, detail="Solver not initialized")
    
    try:
        db_stats = solver.get_stats()
        feedback_stats = get_feedback_stats()
        
        return StatsResponse(
            total_chunks=db_stats.get('total_chunks', 0),
            grades={str(k): v for k, v in db_stats.get('grades', {}).items()},
            subjects=db_stats.get('subjects', {}),
            languages=db_stats.get('languages', {}),
            feedback_stats=feedback_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/subjects")
async def get_subjects():
    """Get list of available subjects"""
    return {
        "subjects": [
            "Mathematics",
            "Science", 
            "Social Science",
            "Hindi",
            "English",
            "Sanskrit"
        ]
    }


@app.get("/api/grades")
async def get_grades():
    """Get list of available grades"""
    return {
        "grades": [5, 6, 7, 8, 9, 10]
    }


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting NCERT Doubt Solver API Server")
    print("="*60)
    print("\n📖 API Documentation: http://localhost:8000/docs")
    print("📖 ReDoc: http://localhost:8000/redoc")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
