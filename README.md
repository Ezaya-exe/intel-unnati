# NCERT Multilingual Doubt Solver 📚

> **Intel Unnati Grand Challenge 2024–25** | AI-powered study companion for the NCERT curriculum

An intelligent doubt-solving assistant for students in **Grades 5–10** that uses **NCERT textbooks as the sole knowledge source**. Built on a Retrieval-Augmented Generation (RAG) pipeline with hybrid search, cross-encoder reranking, streaming answers, and source citations — with automatic fallback handling for out-of-scope questions.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Gradio](https://img.shields.io/badge/UI-Gradio_4-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![LLM](https://img.shields.io/badge/LLM-Groq%20%2F%20Qwen3--4B-green)
![Vector DB](https://img.shields.io/badge/VectorDB-ChromaDB-darkgreen)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📖 **NCERT-Only Knowledge** | Answers strictly grounded in NCERT textbook content (Grades 5–10) |
| 🔍 **Hybrid Search** | BM25 keyword + semantic vector search with score fusion |
| 📈 **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` reranks retrieved chunks for relevance |
| 🔄 **Query Expansion** | Term expansion + multi-query retrieval for better recall |
| ⚡ **Streaming Answers** | Token-by-token streaming in the web UI with live citations |
| 🎯 **Scope Detection** | Calibrated relevance threshold → graceful "I don't know" fallbacks |
| 💬 **Conversation Memory** | Last 5 turns retained; recent history folded into context |
| 📊 **Grade & Subject Filters** | Metadata-filtered retrieval (Grade 5–10, Mathematics, Science, etc.) |
| 📚 **Citations** | Every answer ships with grade / subject / page / source-file references |
| 👍 **Feedback Loop** | Thumbs up/down captured in SQLite, with aggregate stats |
| 🖥️ **Modern UI** | Premium Gradio interface — responsive, mobile-friendly, animated |
| 🔌 **REST API** | FastAPI server for mobile apps and external integrations (CORS-enabled) |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10** (via Conda recommended)
- **NCERT textbook PDFs** (auto-downloadable)
- **An LLM API key** — Groq (recommended) or Gemini — *or* a local GPU for GGUF inference

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Ezaya-exe/intel-unnati.git
cd intel-unnati

# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate ncert_rag

# Option B: pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the repo root:

```env
# Generation provider: "groq" (default) | "gemini" | "gguf"
GENERATION_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Vector store
VECTOR_DB_PATH=data/vector_db
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

# Retrieval switches
HYBRID_SEARCH=true
RERANKING=true
QUERY_EXPANSION=true
```

> On Windows, just run **`run_app.bat`** — it prompts for your API key, saves it to `.env`, and launches the app.

### 3. Download NCERT Textbooks

```bash
python download_ncert.py
```

Downloads chapter PDFs for Grades 5–10 (organized by grade under `data/raw_pdfs/`).

### 4. Build the Vector Database

```bash
python ingest_pdfs.py
```

Extracts text from PDFs → chunks → embeds → stores in ChromaDB (`data/vector_db/`).

Alternatively, run the full modular pipeline step-by-step:

```bash
python src/main_pipeline.py            # all steps
python src/main_pipeline.py ocr        # individual steps: ocr | chunk | vector | test
```

### 5. Launch

```bash
# Web UI (Gradio)          → http://localhost:7860
python app.py

# REST API (FastAPI)       → http://localhost:8000/docs
python api.py
# or: uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 📁 Project Structure

```
intel-unnati/
├── app.py                       # 🌐 Gradio web app (streaming chat + citations)
├── api.py                       # 🔌 FastAPI REST endpoints
├── run_app.bat                  # 🪟 Windows one-click launcher (auto-configures .env)
├── setup.py                     # 🛠️ Directory setup / dependency check
│
├── download_ncert.py            # ⬇️ NCERT textbook downloader (Classes 5–10)
├── ingest_pdfs.py               # 📥 PDF → chunk → embed → ChromaDB pipeline
│
├── evaluate.py                  # 📊 Benchmarking (latency, citation accuracy)
├── evaluate_production.py       # 🛡️ Adversarial refusal tests + RAGAS metrics
├── eval_ragas.py                # 📏 RAGAS 0.2.x evaluation (faithfulness, relevancy…)
├── extract_ragas.py             # 📤 Extract/summarize RAGAS scores from CSV
├── generate_gold.py             # 🪙 Generate gold answers for the eval dataset
├── profile_latency.py           # ⏱️ Latency profiling across sample questions
├── calibrate_scope_threshold.py # 🎯 Tune the in/out-of-scope reranker threshold
│
├── src/
│   ├── main_pipeline.py         # 🏭 End-to-end pipeline (OCR→chunk→embed→test)
│   ├── core/
│   │   ├── doubt_solver.py      # 🧠 Main RAG orchestrator + scope fallback
│   │   └── feedback.py          # 👍 SQLite feedback storage & stats
│   ├── retrieval/
│   │   ├── vector_store.py      # 💾 ChromaDB wrapper (add/query/stats)
│   │   ├── hybrid_search.py     # 🔍 BM25 + semantic score fusion
│   │   ├── reranker.py          # 📈 Cross-encoder reranking (+ caching)
│   │   ├── query_expansion.py   # 🔄 Query expansion & multi-query retrieval
│   │   ├── retrieve_chunks.py   # 🔎 Retriever helpers (filters, rerank)
│   │   └── setup_vectordb.py    # 🧱 Collection bootstrap
│   ├── embedding/
│   │   ├── chunk_text.py        # ✂️ Text chunking (512 tokens / 128 overlap)
│   │   └── create_embeddings.py # 🧮 Embedding generation
│   ├── generation/
│   │   ├── groq_generator.py    # ☁️ Groq LLM (LangChain, streaming-capable)
│   │   └── qwen_gguf.py         # 🤖 Local Qwen3-4B GGUF inference (llama.cpp)
│   ├── ocr/
│   │   ├── extract_text.py      # 📷 PaddleOCR text extraction
│   │   ├── pdf_to_images.py     # 🖼️ PDF → image rasterization
│   │   ├── download_ncert.py    # ⬇️ Grade-wise PDF download helper
│   │   └── scrape_ncert_pdfs.py # 🕸️ NCERT site scraper (with fallback URLs)
│   ├── api/
│   │   ├── api_server.py        # 🔧 Internal API (solve / search / stats)
│   │   └── main.py              # 🔧 Minimal query endpoint
│   └── utils/
│       └── timer.py             # ⏲️ Latency timing utility
│
├── configs/
│   └── pipeline_config.json     # ⚙️ Pipeline, embedding & LLM settings
│
├── data/
│   ├── raw_pdfs/                # 📚 Downloaded NCERT PDFs
│   ├── processed/               # 🗂️ OCR output + chunk JSONs
│   ├── vector_db/               # 🗄️ ChromaDB persist directory
│   ├── evaluation/              # 📋 Eval datasets + RAGAS score CSVs
│   └── feedback.db              # 🗃️ SQLite feedback database
│
├── docs/
│   └── DESIGN.md                # 📐 Architecture & design documentation
│
├── environment.yml              # 📦 Conda environment (ncert_rag)
└── requirements.txt             # 📦 Pip dependencies
```

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` and `/health` | Health check |
| `POST` | `/api/chat` | Ask a doubt (returns answer + citations + latency) |
| `POST` | `/api/feedback` | Submit thumbs up/down feedback |
| `GET` | `/api/stats` | Vector DB + feedback statistics |
| `GET` | `/api/subjects` | Available subject filters |
| `GET` | `/api/grades` | Available grade filters |

### Chat Endpoint

```http
POST /api/chat
```

**Request:**
```json
{
  "question": "What is photosynthesis?",
  "grade": 10,
  "subject": "Science",
  "language": null,
  "n_results": 5
}
```

**Response:**
```json
{
  "question_id": "abc123",
  "question": "What is photosynthesis?",
  "answer": "Photosynthesis is the process by which...",
  "language": "English",
  "citations": [
    { "index": 1, "text": "...", "grade": 10, "subject": "Science", "page": "42" }
  ],
  "in_scope": true,
  "latency_ms": 2340,
  "timestamp": "2026-10-16T10:30:00"
}
```

📚 Full interactive docs: open **http://localhost:8000/docs** (Swagger) or `/redoc` after starting the API server.

---

## 🎯 Performance & Evaluation

| Metric | Target | Status |
|--------|--------|--------|
| End-to-end latency | ≤ 3–5 s | ✅ ~2–4 s (see `profile_output.txt`) |
| Citation accuracy | ≥ 85% | Run `python evaluate.py` |
| Scope threshold | calibrated | Run `python calibrate_scope_threshold.py` |

### Run Evaluation

```bash
python evaluate.py                 # benchmark: latency, citation accuracy, coverage
python eval_ragas.py               # RAGAS metrics on the full eval set
python eval_ragas.py -n 5          # quick RAGAS test (5 questions)
python evaluate_production.py      # adversarial refusal + RAGAS harness
python profile_latency.py          # per-question latency breakdown
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **LLM (cloud)** | Groq — `llama-3.1-8b-instant` (default, via LangChain) |
| **LLM (local)** | Qwen3-4B GGUF (Q4_K_M) via llama-cpp-python |
| **Embeddings** | `paraphrase-multilingual-mpnet-base-v2` |
| **Vector DB** | ChromaDB |
| **Keyword Search** | BM25 (`rank-bm25` style index) |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` cross-encoder |
| **Web UI** | Gradio 4+ |
| **API** | FastAPI + Uvicorn |
| **PDF/OCR** | PyMuPDF, PaddleOCR |
| **Feedback DB** | SQLite |

---

## 📊 Architecture

```
                     ┌──────────────────────┐
   Student Query ──► │  Query Expansion /   │
                     │  Multi-Query         │
                     └──────────┬───────────┘
                                ▼
   ┌────────────────────────────────────────────────┐
   │  Hybrid Retrieval: BM25 + Semantic (ChromaDB)  │
   │              + metadata filters (grade/subject)│
   └────────────────────────┬───────────────────────┘
                            ▼
                ┌──────────────────────────┐
                │ Cross-Encoder Reranking  │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │  Scope Check (threshold) │── out of scope ──► "I don't know"
                └────────────┬─────────────┘                     fallback
                             ▼
                ┌──────────────────────────┐
                │  Context + Conversation  │
                │  History → LLM (Groq /   │
                │  Qwen GGUF) → stream     │
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ Answer + Citations +     │
                │ Latency → UI / API       │──► 👍👎 feedback → SQLite
                └──────────────────────────┘
```

For the detailed design, see [docs/DESIGN.md](docs/DESIGN.md).

---

## 🔧 Configuration Reference

All settings live in `.env` (repo root) or `configs/pipeline_config.json`.

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATION_PROVIDER` | `groq` | `groq` \| `gemini` \| `gguf` |
| `GROQ_API_KEY` | — | Required for the Groq provider |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model id |
| `GGUF_MODEL_PATH` | `models/Qwen3-4B-Q4_K_M.gguf` | Local GGUF model path |
| `N_GPU_LAYERS` / `N_CTX` | `35` / `4096` | llama.cpp GPU offload / context size |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | Sentence-transformers model |
| `VECTOR_DB_PATH` | `data/vector_db` | ChromaDB persist directory |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `128` | Chunking parameters |
| `HYBRID_SEARCH` | `true` | Enable BM25 + semantic fusion |
| `RERANKING` | `true` | Enable cross-encoder reranker |
| `QUERY_EXPANSION` | `true` | Enable query expansion |

---

## 📖 Sample Questions

| Grade | Question |
|-------|----------|
| 10 | What is the difference between evaporation and boiling? |
| 10 | Explain the Fundamental Theorem of Arithmetic |
| 9 | What were the causes of the French Revolution? |
| 8 | Describe the structure of an atom |
| 7 | What is the area of a circle? |
| 6 | Explain the water cycle |
| 5 | What is a rational number? |

---

## 🧪 Optional: Local GGUF Inference (WSL)

If you prefer fully-local, API-free generation:

```bash
# 1. Download the model
mkdir -p models
wget https://huggingface.co/lmstudio-community/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf -P models/

# 2. Start the WSL GGUF server (Flask + llama.cpp)
python wsl_gguf_server.py

# 3. Point the app at it
GENERATION_PROVIDER=gguf
```

Requires WSL2 + NVIDIA GPU with CUDA 12.1+ (4 GB+ VRAM).


---

## 🙏 Acknowledgments

- **Intel Unnati Program** for the problem statement
- **NCERT** for open-access textbooks
- **Groq**, **Qwen Team**, and the broader open-source ML community

---

*Built with ❤️ for students across India*
