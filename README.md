# NCERT Multilingual Doubt Solver 📚

> **Intel Unnati Grand Challenge 2024-25** | AI-powered educational assistant for NCERT curriculum

An intelligent doubt-solver for students in Grades 5-10 that uses NCERT textbooks as the sole knowledge source. Built with a Retrieval-Augmented Generation (RAG) pipeline, supporting Hindi and English with accurate citations.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange)
![LLM](https://img.shields.io/badge/LLM-Qwen3--4B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌍 **Multilingual** | Hindi & English with auto language detection |
| 📊 **Grade Filtering** | Filter responses by Grade (5-10) and Subject |
| 📖 **Citations** | Every answer includes source references |
| 💬 **Conversation** | 5-turn conversation memory |
| 🤖 **Smart Fallback** | "I don't know" for out-of-scope queries |
| 👍 **Feedback** | Rate answers with thumbs up/down |
| 📱 **Mobile Ready** | Responsive UI for web and mobile |
| 🔍 **Hybrid Search** | BM25 + Semantic search with reranking |

---

## 🚀 Quick Start

### Prerequisites
- **WSL2** with Ubuntu (Windows)
- **Conda** (Miniconda/Anaconda)
- **NVIDIA GPU** with CUDA 12.1+ (4GB+ VRAM)

### 1. Clone & Setup Environment

```bash
# Clone the repository
cd /mnt/d/study/python
git clone <repo-url> intel-unnati
cd intel-unnati

# Create conda environment
conda env create -f environment.yml
conda activate ncert_rag
```

### 2. Download NCERT Textbooks

```bash
python download_ncert.py
```
This downloads 465+ PDF chapters for Grades 5-10 (~2GB).

### 3. Download the LLM Model

```bash
# Download Qwen3-4B-Q4 GGUF model
mkdir -p models
cd models
wget https://huggingface.co/lmstudio-community/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf
cd ..
```

### 4. Ingest PDFs into Vector Database

```bash
python ingest_pdfs.py
```
Creates vector embeddings for all textbook content.

### 5. Run the Application

```bash
# Web UI (Gradio)
python app.py
# Open http://localhost:7860

# REST API (FastAPI)
python api.py
# Open http://localhost:8000/docs
```

---

## 📁 Project Structure

```
intel-unnati/
├── app.py                  # 🌐 Gradio web interface
├── api.py                  # 🔌 FastAPI REST endpoints
├── evaluate.py             # 📊 Benchmarking script
├── ingest_pdfs.py          # 📥 PDF ingestion pipeline
├── download_ncert.py       # ⬇️ NCERT textbook downloader
│
├── src/
│   ├── core/
│   │   ├── doubt_solver.py # 🧠 Main RAG orchestrator
│   │   └── feedback.py     # 👍 Feedback collection
│   │
│   ├── retrieval/
│   │   ├── vector_store.py # 💾 ChromaDB + advanced search
│   │   ├── hybrid_search.py# 🔍 BM25 + semantic fusion
│   │   ├── reranker.py     # 📈 Cross-encoder reranking
│   │   └── query_expansion.py # 🔄 Query term expansion
│   │
│   ├── generation/
│   │   ├── qwen_gguf.py    # 🤖 GGUF model inference
│   │   └── llm_generator.py# 📝 LLM wrapper
│   │
│   └── ocr/
│       └── extract_text.py # 📷 Tesseract OCR
│
├── data/
│   ├── raw_pdfs/           # 📚 Downloaded NCERT PDFs
│   ├── vector_db/          # 🗄️ ChromaDB storage
│   └── evaluation/         # 📋 Benchmark datasets
│
├── docs/
│   └── DESIGN.md           # 📐 Architecture documentation
│
├── environment.yml         # 📦 Conda environment
└── requirements.txt        # 📦 Pip dependencies
```

---

## 🎯 Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| End-to-end Latency | ≤ 3-5 seconds | ✅ ~2-4s |
| Citation Accuracy | ≥ 85% | Run `python evaluate.py` |
| GPU Memory | ≤ 4GB VRAM | ✅ ~3.5GB |

### Run Benchmarks

```bash
# Run full evaluation (50 questions)
python evaluate.py

# Run quick test (10 questions)
python evaluate.py -n 10
```

---

## 🔌 API Reference

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
  "language": null
}
```

**Response:**
```json
{
  "question_id": "abc123",
  "answer": "Photosynthesis is the process by which...",
  "language": "English",
  "citations": [...],
  "latency_ms": 2340,
  "in_scope": true
}
```

### Feedback Endpoint
```http
POST /api/feedback
```

### Full API Docs
Open http://localhost:8000/docs after starting the API server.

---

## 📖 Sample Questions

Try these questions to test the system:

| Grade | Subject | Question |
|-------|---------|----------|
| 10 | Science | What is the difference between evaporation and boiling? |
| 10 | Maths | Explain the Fundamental Theorem of Arithmetic |
| 9 | Social | What were the causes of the French Revolution? |
| 8 | Science | Describe the structure of an atom |
| 7 | Maths | What is the area of a circle? |
| 9 | Hindi | प्रकाश संश्लेषण क्या है? |

---

## 🛠️ Technology Stack

- **LLM**: Qwen3-4B-GGUF (Q4_K_M quantized)
- **Embeddings**: paraphrase-multilingual-mpnet-base-v2
- **Vector DB**: ChromaDB
- **Keyword Search**: BM25 (rank-bm25)
- **Reranker**: ms-marco-MiniLM-L-6-v2
- **Web UI**: Gradio
- **API**: FastAPI
- **PDF Parser**: PyMuPDF

---

## 📊 Architecture

```
Query → Language Detection → Query Expansion → Hybrid Search (BM25+Semantic)
                                                        ↓
                                               Cross-Encoder Rerank
                                                        ↓
Answer ← Citation Formatter ← LLM Generation ← Context Builder
```

For detailed architecture, see [docs/DESIGN.md](docs/DESIGN.md).

---

## 🔧 Configuration

Create a `.env` file:

```env
# Model Configuration
GGUF_MODEL_PATH=models/Qwen3-4B-Q4_K_M.gguf
N_GPU_LAYERS=35
N_CTX=4096

# Vector Store
VECTOR_DB_PATH=data/vector_db
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2

# Search Settings
HYBRID_SEARCH=true
RERANKING=true
QUERY_EXPANSION=true
```

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Intel Unnati Program** for the problem statement
- **NCERT** for providing open-access textbooks
- **Qwen Team** for the multilingual LLM

---

*Built with ❤️ for students across India*
