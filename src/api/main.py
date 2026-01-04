from fastapi import FastAPI
from pydantic import BaseModel

from src.retrieval.retrieve_chunks import search_chunks, rerank_chunks
from src.generation.phi_model import generate_answer


app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    grade: int | None = None
    subject: str | None = None
    language: str | None = None

class QueryResponse(BaseModel):
    answer: str
    contexts: list[dict]

@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    # 1. Retrieve chunks from Chroma
    raw_chunks = search_chunks(
        req.question,
        grade=req.grade,
        subject=req.subject,
        language=req.language,
        top_k=8,
    )
    chunks = rerank_chunks(req.question, raw_chunks)
    top_chunks = chunks[:5]

    # 2. Build context string
    context_parts = []
    for c in top_chunks:
        md = c["metadata"]
        ref = f"[Grade {md.get('grade')}, {md.get('subject')}, page {md.get('page_number', '?')}]"
        context_parts.append(ref + "\n" + c["text"])
    context_str = "\n\n---\n\n".join(context_parts)

    # 3. Call Gemma
    answer = generate_answer(context_str, req.question)

    return QueryResponse(
        answer=answer,
        contexts=top_chunks,
    )
