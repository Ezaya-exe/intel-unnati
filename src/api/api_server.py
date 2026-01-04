from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from generation.llm_generator import NCERTDoubtSolver
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NCERT Doubt Solver API",
    description="Multilingual RAG-based doubt solver for NCERT textbooks (Grades 5-10)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize doubt solver (will be done on startup)
doubt_solver: Optional[NCERTDoubtSolver] = None


# Pydantic models for request/response
class DoubtRequest(BaseModel):
    question: str = Field(..., description="Student's question or doubt")
    grade: Optional[int] = Field(None, ge=5, le=10, description="Grade level (5-10)")
    subject: Optional[str] = Field(None, description="Subject (Mathematics, Science, SST, Hindi, etc.)")
    language: Optional[str] = Field('english', description="Preferred language (english, hindi, marathi, sanskrit)")
    n_context_chunks: Optional[int] = Field(5, ge=1, le=10, description="Number of context chunks to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is photosynthesis?",
                "grade": 6,
                "subject": "Science",
                "language": "english",
                "n_context_chunks": 5
            }
        }


class Source(BaseModel):
    text: str
    source: str
    grade: int
    subject: str


class DoubtResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    message: str
    vector_store_stats: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the doubt solver on startup"""
    global doubt_solver
    logger.info("Initializing NCERT Doubt Solver...")
    
    try:
        # Use HuggingFace for free tier, can switch to OpenAI with API key
        doubt_solver = NCERTDoubtSolver(
            model_provider="huggingface",  # Change to "openai" if you have API key
            model_name="google/flan-t5-base"
        )
        logger.info("✓ Doubt solver initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize doubt solver: {e}")
        # Continue anyway, will error on requests


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    if doubt_solver is None:
        raise HTTPException(status_code=503, detail="Doubt solver not initialized")
    
    # Get vector store statistics
    stats = doubt_solver.retriever.vector_store.get_collection_stats()
    
    return {
        "status": "healthy",
        "message": "NCERT Doubt Solver API is running",
        "vector_store_stats": stats
    }


@app.post("/solve", response_model=DoubtResponse)
async def solve_doubt(request: DoubtRequest):
    """
    Solve a student's doubt using RAG
    
    - **question**: The student's question or doubt
    - **grade**: Optional grade filter (5-10)
    - **subject**: Optional subject filter
    - **language**: Preferred language for the answer
    - **n_context_chunks**: Number of relevant text chunks to retrieve
    """
    if doubt_solver is None:
        raise HTTPException(status_code=503, detail="Doubt solver not initialized")
    
    try:
        logger.info(f"Received question: {request.question[:100]}")
        
        result = doubt_solver.solve_doubt(
            question=request.question,
            grade=request.grade,
            subject=request.subject,
            language=request.language,
            n_context_chunks=request.n_context_chunks
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error solving doubt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_textbooks(
    query: str = Query(..., description="Search query"),
    grade: Optional[int] = Query(None, ge=5, le=10, description="Grade filter"),
    subject: Optional[str] = Query(None, description="Subject filter"),
    language: Optional[str] = Query(None, description="Language filter"),
    n_results: int = Query(5, ge=1, le=20, description="Number of results")
):
    """
    Search textbooks without generating an answer
    Returns relevant text chunks from NCERT textbooks
    """
    if doubt_solver is None:
        raise HTTPException(status_code=503, detail="Doubt solver not initialized")
    
    try:
        result = doubt_solver.retriever.retrieve(
            query=query,
            grade=grade,
            subject=subject,
            language=language,
            n_results=n_results
        )
        return result
    
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get statistics about the vector database"""
    if doubt_solver is None:
        raise HTTPException(status_code=503, detail="Doubt solver not initialized")
    
    stats = doubt_solver.retriever.vector_store.get_collection_stats()
    return stats


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
