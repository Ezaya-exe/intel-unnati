# NCERT Multilingual Doubt-Solver - Design Document

## Intel Unnati Program 2024-25

---

## 1. Executive Summary

The NCERT Multilingual Doubt-Solver is an AI-powered educational assistant that helps students in Grades 5-10 get answers to their academic questions using NCERT textbooks as the sole knowledge source. Built on a Retrieval-Augmented Generation (RAG) architecture, the system supports Hindi and English, provides accurate citations, and maintains conversation context.

### Key Features
- **Multilingual Support**: Hindi, English with automatic language detection
- **Grade-Specific Filtering**: Filter responses by Grade (5-10) and Subject
- **Accurate Citations**: Every answer includes source references
- **Conversation Memory**: Maintains context across 5 conversation turns
- **"I Don't Know" Fallback**: Graceful handling of out-of-scope queries
- **Feedback System**: Students can rate answer quality

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   Gradio Web UI     │  │   FastAPI Endpoint  │  │   Mobile PWA        │  │
│  │   (Port 7860)       │  │   (Port 8000)       │  │   (Responsive)      │  │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘  │
└─────────────┼────────────────────────┼────────────────────────┼─────────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Language   │  │   Query     │  │   Filter    │  │   Conversation      │ │
│  │  Detection  │──▶  Expansion  │──▶   Builder   │──▶   History Manager   │ │
│  │  (langdetect)│  │  (LLM+Static)│ │  (Grade/Subj)│  │   (5-turn context) │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐
│   RETRIEVAL LAYER   │  │   GENERATION LAYER  │  │    FEEDBACK LAYER       │
│  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────────────┐  │
│  │ Hybrid Search │  │  │  │  Qwen3-4B     │  │  │  │  SQLite Database  │  │
│  │ (BM25+Semantic)│  │  │  │  GGUF Model   │  │  │  │  (feedback.db)    │  │
│  ├───────────────┤  │  │  ├───────────────┤  │  │  ├───────────────────┤  │
│  │  Cross-Encoder │  │  │  │ ChatML Prompt │  │  │  │ Thumbs Up/Down    │  │
│  │  Reranker     │  │  │  │  Template     │  │  │  │ Quality Metrics   │  │
│  ├───────────────┤  │  │  ├───────────────┤  │  │  └───────────────────┘  │
│  │  ChromaDB     │  │  │  │  Citation     │  │  │                         │
│  │  Vector Store │  │  │  │  Formatter    │  │  │                         │
│  └───────────────┘  │  └───────────────────┘  │  └─────────────────────────┘
└─────────────────────┘  └─────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Raw PDFs   │  │  Extracted  │  │  Chunked    │  │  Embeddings         │ │
│  │  (465 files)│──▶   Text     │──▶   Documents │──▶  (768-dim vectors)  │ │
│  │  by Grade   │  │  (PyMuPDF)  │  │  (768 tokens)│  │  (Multilingual)    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 RAG Pipeline Flow

```
User Query
    │
    ▼
┌──────────────────┐
│ Language Detect  │ ──▶ Detected: "Hindi" or "English"
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Query Expansion  │ ──▶ "photosynthesis" → ["chlorophyll", "sunlight", "glucose"]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Hybrid Search   │ ──▶ BM25 (keyword) + Semantic (embedding)
│  (Top 20 docs)   │     Score fusion via Reciprocal Rank Fusion
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Cross-Encoder   │ ──▶ Rerank top 20 → Top 5 most relevant
│  Reranking       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Context Builder │ ──▶ Combine: [Source 1] + [Source 2] + ... + History
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM Generation  │ ──▶ Qwen3-4B with ChatML prompt
│  (GGUF on GPU)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Citation Format │ ──▶ Add source references to response
└────────┬─────────┘
         │
         ▼
    Final Answer + Citations
```

---

## 3. Technology Stack

### 3.1 Core Components

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **LLM** | Qwen3-4B-GGUF | Q4_K_M | Answer generation, 4-bit quantized |
| **LLM Runtime** | llama-cpp-python | 0.3+ | CUDA-accelerated inference |
| **Embeddings** | paraphrase-multilingual-mpnet-base-v2 | - | 768-dim multilingual embeddings |
| **Vector DB** | ChromaDB | 0.4+ | Persistent vector storage |
| **Keyword Search** | rank-bm25 | 0.2+ | BM25 algorithm implementation |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | - | Cross-encoder reranking |
| **PDF Parsing** | PyMuPDF (fitz) | 1.23+ | Text extraction from PDFs |
| **OCR** | Tesseract | 5.0+ | Image-based text extraction |
| **Web UI** | Gradio | 4.0+ | Chat interface |
| **API** | FastAPI | 0.100+ | REST endpoints |

### 3.2 Infrastructure

| Component | Specification |
|-----------|--------------|
| **OS** | WSL2 (Ubuntu 22.04) on Windows 11 |
| **GPU** | NVIDIA RTX (CUDA 12.1) |
| **Python** | 3.10 (Conda environment) |
| **Memory** | 16GB+ RAM recommended |
| **VRAM** | 4GB+ (for Qwen3-4B Q4) |

---

## 4. Data Pipeline

### 4.1 PDF Ingestion Pipeline

```
NCERT Website
      │
      ▼ download_ncert.py
┌─────────────────┐
│  Raw PDFs       │ ──▶ data/raw_pdfs/grade_{5-10}/
│  (465 files)    │     ~2GB total
└────────┬────────┘
         │
         ▼ ingest_pdfs.py
┌─────────────────┐
│  Text Extraction│ ──▶ PyMuPDF extracts text per page
│  (PyMuPDF)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chapter Detect │ ──▶ Regex patterns for "Chapter X", "अध्याय X"
│  (Headers)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chunking       │ ──▶ 768 tokens per chunk, 150 token overlap
│  (LangChain)    │     RecursiveCharacterTextSplitter
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Metadata       │ ──▶ {grade, subject, language, chapter_title, source_file}
│  Enrichment     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding      │ ──▶ SentenceTransformer on CUDA
│  Generation     │     768-dimensional vectors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ChromaDB       │ ──▶ data/vector_db/
│  Storage        │     7,594 chunks indexed
└─────────────────┘
```

### 4.2 Chunk Metadata Schema

```json
{
  "chunk_id": "g10_Mat_jemh101_c5_3847",
  "text": "[Mathematics - Chapter 1: Real Numbers]\n\nThe Fundamental Theorem...",
  "metadata": {
    "grade": 10,
    "subject": "Mathematics",
    "language": "English",
    "textbook": "Mathematics",
    "chapter": "jemh101",
    "chapter_title": "Chapter 1 - Real Numbers",
    "source_file": "Mathematics_Mathematics_jemh101.pdf",
    "chunk_index": 5
  }
}
```

---

## 5. API Specification

### 5.1 Chat Endpoint

```http
POST /api/chat
Content-Type: application/json

{
  "question": "What is photosynthesis?",
  "grade": 10,           // Optional: 5-10
  "subject": "Science",  // Optional
  "language": null       // Optional: "Hindi" or "English"
}
```

**Response:**
```json
{
  "answer": "Photosynthesis is the process by which plants convert...",
  "language": "English",
  "citations": [
    {
      "index": 1,
      "grade": 10,
      "subject": "Science",
      "text": "Photosynthesis occurs in the chloroplasts...",
      "page": "45"
    }
  ],
  "latency_ms": 2340,
  "in_scope": true
}
```

### 5.2 Feedback Endpoint

```http
POST /api/feedback
Content-Type: application/json

{
  "question_id": "uuid-1234",
  "rating": "thumbs_up",  // or "thumbs_down"
  "comment": "Very helpful explanation"
}
```

---

## 6. Performance Specifications

### 6.1 Targets

| Metric | Target | Current |
|--------|--------|---------|
| **End-to-end Latency** | ≤ 3-5 seconds | ~2-4 seconds |
| **Citation Accuracy** | ≥ 85% | TBD (needs eval) |
| **GPU Memory** | ≤ 4GB VRAM | ~3.5GB |
| **Throughput** | 10+ queries/min | ~15 queries/min |

### 6.2 Optimizations Applied

1. **4-bit Quantization**: Qwen3-4B-Q4_K_M reduces model from 8GB → 2.4GB
2. **Hybrid Search**: BM25 + semantic improves retrieval precision
3. **Reranking**: Cross-encoder filters irrelevant chunks
4. **GPU Acceleration**: All models run on CUDA
5. **Larger Chunks**: 768 tokens provides better context

---

## 7. Security Considerations

- **No PII Storage**: User queries are not persisted
- **Feedback Anonymized**: No user identifiers with feedback
- **Local Deployment**: All models run locally, no external API calls
- **Input Sanitization**: Queries are sanitized before processing

---

## 8. Future Enhancements

### 8.1 Planned (Stretch Goals)
- [ ] Voice input/output using Whisper + TTS
- [ ] Image-based question input (OCR + Vision)
- [ ] Adaptive explanations based on grade level

### 8.2 Potential Improvements
- [ ] LLM finetuning on Hindi Q&A pairs
- [ ] Support for Urdu and regional languages
- [ ] Offline mobile app (ONNX runtime)

---

## 9. File Structure

```
intel-unnati/
├── app.py                     # Gradio web application
├── ingest_pdfs.py             # PDF ingestion pipeline
├── download_ncert.py          # NCERT textbook downloader
├── evaluate.py                # Evaluation/benchmark script
│
├── src/
│   ├── core/
│   │   ├── doubt_solver.py    # Main RAG orchestrator
│   │   └── feedback.py        # Feedback collection
│   │
│   ├── retrieval/
│   │   ├── vector_store.py    # ChromaDB + advanced search
│   │   ├── hybrid_search.py   # BM25 + semantic fusion
│   │   ├── reranker.py        # Cross-encoder reranking
│   │   └── query_expansion.py # Query term expansion
│   │
│   ├── generation/
│   │   ├── qwen_gguf.py       # GGUF model inference
│   │   └── llm_generator.py   # LLM wrapper
│   │
│   └── ocr/
│       └── extract_text.py    # Tesseract OCR
│
├── data/
│   ├── raw_pdfs/              # Downloaded NCERT PDFs
│   ├── vector_db/             # ChromaDB storage
│   └── evaluation/            # Benchmark datasets
│
├── docs/
│   └── DESIGN.md              # This document
│
├── environment.yml            # Conda environment
├── requirements.txt           # Pip dependencies
└── README.md                  # Quick start guide
```

---

## 10. References

- [NCERT Textbooks Portal](https://ncert.nic.in/textbook.php)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen2.5-4B)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Text Splitters](https://python.langchain.com/docs/how_to/recursive_text_splitter/)
- [OPEA GenAI Microservices](https://github.com/opea-project)

---

*Document Version: 1.0 | Last Updated: January 2026*
