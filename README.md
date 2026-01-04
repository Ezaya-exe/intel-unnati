# 📚 NCERT Multilingual Doubt Solver

An AI-powered Retrieval-Augmented Generation (RAG) system for answering NCERT textbook queries for students in Grades 5-10. Built for the **Intel Unnati Program**.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-green)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🌟 Features

- **RAG Pipeline**: Retrieves relevant context from NCERT textbooks before generating answers
- **Multilingual Support**: Works with English, Hindi, and other Indian languages
- **GPU Accelerated**: Uses CUDA for fast inference with Qwen3-4B GGUF model
- **11,400+ Chunks**: Indexed content from Classes 5-10 across all subjects
- **Citation System**: Shows source textbook and chapter for each answer
- **Gradio UI**: Clean web interface for easy interaction

---

## 📁 Project Structure

```
intel-unnati/
├── app.py                    # 🌐 Main Gradio web application
├── ingest_pdfs.py            # 📄 PDF text extraction & embedding ingestion
├── download_ncert.py         # ⬇️ NCERT textbook downloader script
│
├── src/                      # 📦 Core modules
│   ├── core/
│   │   ├── doubt_solver.py   # Main RAG orchestrator
│   │   └── feedback.py       # User feedback system
│   │
│   ├── generation/
│   │   ├── qwen_gguf.py      # Qwen3-4B GGUF model inference
│   │   ├── llm_generator.py  # Generic LLM wrapper
│   │   └── phi_model.py      # (Legacy) Phi-3 model support
│   │
│   ├── retrieval/
│   │   └── vector_store.py   # ChromaDB vector store
│   │
│   ├── embedding/
│   │   └── chunk_text.py     # Text chunking with LangChain
│   │
│   └── ocr/
│       └── extract_text.py   # OCR for scanned PDFs (Tesseract)
│
├── data/                     # 📊 Data directory (gitignored)
│   ├── raw_pdfs/             # Downloaded NCERT PDFs by grade
│   │   ├── grade_5/
│   │   ├── grade_6/
│   │   └── ...
│   ├── vector_db/            # ChromaDB persistent storage
│   └── processed/            # Intermediate processing files
│
├── models/                   # 🤖 Model files (gitignored)
│   └── Qwen3-4B-Q4_K_M.gguf  # Quantized LLM model
│
├── configs/                  # ⚙️ Configuration files
│   └── settings.yaml
│
├── environment.yml           # 📦 Conda environment specification
├── requirements.txt          # 📦 Pip requirements
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Operating System**: WSL2 (Ubuntu) on Windows, or native Linux
- **GPU**: NVIDIA GPU with CUDA 12.1+ support
- **Conda**: Miniconda or Anaconda installed

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ncert-doubt-solver.git
cd ncert-doubt-solver
```

### 2. Create Conda Environment

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate ncert_rag
```

### 3. Download NCERT Textbooks

```bash
python download_ncert.py
```

This downloads all NCERT textbooks (Classes 5-10) and organizes them in `data/raw_pdfs/`.

### 4. Download the LLM Model

Download the Qwen3-4B GGUF model:

```bash
mkdir -p models
# Download from HuggingFace
wget https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf -P models/
```

### 5. Ingest PDFs into Vector Database

```bash
python ingest_pdfs.py
```

This extracts text from all PDFs, creates embeddings, and stores them in ChromaDB.

### 6. Run the Application

```bash
python app.py
```

Open your browser and navigate to: **http://localhost:7860**

---

## 🧪 Sample Questions to Try

After launching the app, try these questions:

| Grade | Question |
|-------|----------|
| 10 | "Explain the Fundamental Theorem of Arithmetic with an example" |
| 10 | "What is the difference between metallic and electrolytic conductors?" |
| 9 | "Describe the nitrogen cycle and its importance in agriculture" |
| 9 | "How did the Treaty of Versailles lead to World War II?" |
| 8 | "What are rational numbers? Give examples." |

---

## ⚙️ Configuration

### Environment Variables (.env)

Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_PATH=/path/to/models/Qwen3-4B-Q4_K_M.gguf
N_GPU_LAYERS=-1  # -1 for all layers on GPU

# Vector Store
VECTOR_DB_PATH=data/vector_db
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
```

---

## 📊 Technical Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Qwen3-4B (GGUF, 4-bit quantized) |
| **Embeddings** | `paraphrase-multilingual-mpnet-base-v2` |
| **Vector DB** | ChromaDB |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter |
| **Web UI** | Gradio |
| **Inference** | llama-cpp-python (CUDA) |

---

## 🛠️ Development

### Running in Development Mode

```bash
# With hot-reload
gradio app.py
```

### Rebuilding Vector Store

To re-ingest all PDFs (e.g., after adding new textbooks):

```bash
rm -rf data/vector_db/*
python ingest_pdfs.py
```

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Intel Unnati Program** for project sponsorship
- **NCERT** for educational content
- **Unsloth** for optimized GGUF models
- **HuggingFace** for transformers and sentence-transformers

---

## 📧 Contact

For questions or feedback, reach out via GitHub Issues.
