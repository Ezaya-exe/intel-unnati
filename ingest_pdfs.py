#!/usr/bin/env python3
"""
NCERT PDF Ingestion Pipeline
Extracts text from PDF chapters, chunks them, and ingests into ChromaDB vector store.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import logging
from tqdm import tqdm

# PyMuPDF (fitz) for PDF text extraction
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Installing PyMuPDF...")
    os.system("pip install pymupdf")
    import fitz

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).parent
RAW_PDFS_DIR = BASE_DIR / "data" / "raw_pdfs"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"

# Text splitter for chunking
encoding = tiktoken.get_encoding("cl100k_base")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    length_function=lambda x: len(encoding.encode(x)),
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def parse_filename(filename: str) -> Dict[str, str]:
    """Parse metadata from filename like: Mathematics_Mathematics_jemh101.pdf"""
    parts = filename.replace(".pdf", "").split("_")
    
    # Common subject mappings
    subject_map = {
        "Mathematics": "Mathematics",
        "Math": "Mathematics",
        "Science": "Science",
        "Social": "Social Science",
        "English": "English",
        "Hindi": "Hindi",
        "Sanskrit": "Sanskrit",
        "EVS": "EVS"
    }
    
    subject = parts[0] if parts else "Unknown"
    subject = subject_map.get(subject, subject)
    
    return {
        "subject": subject,
        "title": "_".join(parts[1:-1]) if len(parts) > 2 else parts[0],
        "chapter_code": parts[-1] if parts else "unknown"
    }


def detect_language(text: str) -> str:
    """Simple language detection based on script"""
    # Check for Devanagari (Hindi/Sanskrit)
    if re.search(r'[\u0900-\u097F]', text[:500]):
        return "Hindi"
    return "English"


def process_grade_folder(grade_dir: Path, grade: int) -> List[Dict]:
    """Process all PDFs in a grade folder"""
    chunks = []
    pdf_files = list(grade_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {grade_dir}")
        return chunks
    
    for pdf_path in tqdm(pdf_files, desc=f"Grade {grade}"):
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        if not text or len(text) < 100:
            logger.warning(f"Skipping {pdf_path.name} - insufficient text")
            continue
        
        # Parse metadata from filename
        file_meta = parse_filename(pdf_path.name)
        language = detect_language(text)
        
        # Split into chunks
        text_chunks = splitter.split_text(text)
        
        # Create chunk documents
        for idx, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < 20:
                continue
            
            # Use hash of text to ensure unique IDs
            text_hash = hash(chunk_text) % 10000
            chunk_id = f"g{grade}_{file_meta['subject'][:3]}_{file_meta['chapter_code']}_c{idx}_{text_hash}"
            
            chunk = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "grade": grade,
                    "subject": file_meta["subject"],
                    "language": language,
                    "textbook": file_meta["title"],
                    "chapter": file_meta["chapter_code"],
                    "source_file": pdf_path.name
                }
            }
            chunks.append(chunk)
    
    return chunks


def ingest_to_vector_store(chunks: List[Dict]):
    """Ingest chunks into ChromaDB vector store"""
    from retrieval.vector_store import NCERTVectorStore
    
    logger.info(f"\n📦 Initializing Vector Store...")
    
    # Delete old data and create fresh
    vector_store = NCERTVectorStore(persist_directory=str(VECTOR_DB_DIR))
    
    # Delete existing collection
    try:
        vector_store.delete_collection()
        logger.info("Deleted old collection")
    except:
        pass
    
    # Reinitialize
    vector_store = NCERTVectorStore(persist_directory=str(VECTOR_DB_DIR))
    
    # Batch the chunks (ChromaDB has a limit)
    batch_size = 500
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.info(f"Ingesting batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        vector_store.add_chunks(batch)
    
    # Print stats
    stats = vector_store.get_collection_stats()
    logger.info("\n" + "="*60)
    logger.info("📊 Vector Store Statistics:")
    logger.info(f"   Total chunks: {stats['total_chunks']}")
    logger.info(f"   Grades: {stats.get('grades', {})}")
    logger.info(f"   Subjects: {stats.get('subjects', {})}")
    logger.info(f"   Languages: {stats.get('languages', {})}")
    logger.info("="*60)
    
    return stats


def main():
    print("="*60)
    print("🏫 NCERT PDF Ingestion Pipeline")
    print("="*60)
    print(f"\n📁 Source: {RAW_PDFS_DIR}")
    print(f"📦 Target: {VECTOR_DB_DIR}\n")
    
    all_chunks = []
    
    # Process each grade folder
    for grade in range(5, 11):
        grade_dir = RAW_PDFS_DIR / f"grade_{grade}"
        
        if not grade_dir.exists():
            logger.warning(f"Grade {grade} folder not found: {grade_dir}")
            continue
        
        print(f"\n📚 Processing Grade {grade}...")
        chunks = process_grade_folder(grade_dir, grade)
        all_chunks.extend(chunks)
        print(f"   ✓ Extracted {len(chunks)} chunks from Grade {grade}")
    
    print(f"\n📊 Total chunks extracted: {len(all_chunks)}")
    
    if not all_chunks:
        logger.error("No chunks extracted! Check PDF files.")
        return
    
    # Save chunks to JSON for debugging
    chunks_file = BASE_DIR / "data" / "processed" / "all_chunks.json"
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved chunks to: {chunks_file}")
    
    # Ingest to vector store
    print("\n🔄 Ingesting to Vector Store...")
    stats = ingest_to_vector_store(all_chunks)
    
    print("\n✅ Ingestion Complete!")
    print(f"   Total documents in DB: {stats['total_chunks']}")


if __name__ == "__main__":
    main()
