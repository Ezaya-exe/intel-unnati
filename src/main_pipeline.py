"""
NCERT Multilingual Doubt Solver - Main Pipeline
Complete RAG pipeline: Data Collection → OCR → Chunking → Embedding → Vector Store
"""

import sys
from pathlib import Path
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from ocr.extract_text import NCERTOCRProcessor
from embedding.chunk_text import NCERTChunker
from retrieval.vector_store import NCERTVectorStore
from generation.llm_generator import NCERTDoubtSolver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NCERTPipeline:
    """Complete pipeline for NCERT doubt solver"""
    
    def __init__(self):
        self.ocr_processor = None
        self.chunker = None
        self.vector_store = None
        self.doubt_solver = None
        
    def step1_ocr_processing(self, pdf_dir="data/raw_pdfs", output_dir="data/processed/ocr_output"):
        """Step 1: Extract text from PDFs using PaddleOCR"""
        logger.info("="*80)
        logger.info("STEP 1: OCR PROCESSING")
        logger.info("="*80)
        
        self.ocr_processor = NCERTOCRProcessor()
        
        pdf_path = Path(pdf_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Define textbooks to process
        textbooks = [
            # Grade 5
            {"pdf": "grade_5_math_en.pdf", "metadata": {"grade": 5, "subject": "Mathematics", "language": "english", "title": "Math Class 5"}},
            {"pdf": "grade_5_science_en.pdf", "metadata": {"grade": 5, "subject": "Science", "language": "english", "title": "Science Class 5"}},
            {"pdf": "grade_5_hindi.pdf", "metadata": {"grade": 5, "subject": "Hindi", "language": "hindi", "title": "Hindi Class 5"}},
            
            # Grade 6
            {"pdf": "grade_6_math_en.pdf", "metadata": {"grade": 6, "subject": "Mathematics", "language": "english", "title": "Math Class 6"}},
            {"pdf": "grade_6_science_en.pdf", "metadata": {"grade": 6, "subject": "Science", "language": "english", "title": "Science Class 6"}},
            {"pdf": "grade_6_sst_en.pdf", "metadata": {"grade": 6, "subject": "Social Science", "language": "english", "title": "SST Class 6"}},
            
            # Add more grades 7-10 as needed
        ]
        
        processed_count = 0
        for item in textbooks:
            pdf_file = pdf_path / item['pdf']
            if not pdf_file.exists():
                logger.warning(f"PDF not found: {pdf_file}, skipping...")
                continue
            
            logger.info(f"\nProcessing: {item['metadata']['title']}")
            result = self.ocr_processor.process_pdf_textbook(str(pdf_file), item['metadata'])
            
            # Save OCR output
            output_file = output_path / f"{pdf_file.stem}_ocr.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ Saved: {output_file}")
            processed_count += 1
        
        logger.info(f"\n✓ OCR processing complete: {processed_count} textbooks processed")
        return processed_count
    
    def step2_chunk_text(self, ocr_dir="data/processed/ocr_output", output_dir="data/processed/chunks"):
        """Step 2: Chunk the extracted text"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: TEXT CHUNKING")
        logger.info("="*80)
        
        self.chunker = NCERTChunker(chunk_size=512, chunk_overlap=128)
        
        ocr_path = Path(ocr_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        if not ocr_path.exists():
            logger.error(f"OCR directory not found: {ocr_path}")
            return 0
        
        processed_count = 0
        total_chunks = 0
        
        for ocr_file in ocr_path.glob("*_ocr.json"):
            logger.info(f"\nChunking: {ocr_file.name}")
            
            chunks = self.chunker.create_chunks(str(ocr_file))
            
            # Save chunks
            output_file = output_path / ocr_file.name.replace('_ocr.json', '_chunks.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ Created {len(chunks)} chunks → {output_file}")
            processed_count += 1
            total_chunks += len(chunks)
        
        logger.info(f"\n✓ Chunking complete: {total_chunks} total chunks from {processed_count} textbooks")
        return total_chunks
    
    def step3_create_vector_store(self, chunks_dir="data/processed/chunks", vector_db_path="data/vector_db"):
        """Step 3: Create embeddings and populate vector store"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: VECTOR STORE CREATION")
        logger.info("="*80)
        
        self.vector_store = NCERTVectorStore(persist_directory=vector_db_path)
        
        chunks_path = Path(chunks_dir)
        
        if not chunks_path.exists():
            logger.error(f"Chunks directory not found: {chunks_path}")
            return 0
        
        total_chunks = 0
        
        for chunks_file in chunks_path.glob("*_chunks.json"):
            logger.info(f"\nProcessing: {chunks_file.name}")
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            self.vector_store.add_chunks(chunks)
            total_chunks += len(chunks)
        
        # Get statistics
        stats = self.vector_store.get_collection_stats()
        logger.info(f"\n✓ Vector store created successfully")
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Grades: {stats.get('grades', {})}")
        logger.info(f"Subjects: {stats.get('subjects', {})}")
        logger.info(f"Languages: {stats.get('languages', {})}")
        
        return total_chunks
    
    def step4_test_rag(self, vector_db_path="data/vector_db"):
        """Step 4: Test the complete RAG pipeline"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: TESTING RAG PIPELINE")
        logger.info("="*80)
        
        self.doubt_solver = NCERTDoubtSolver(
            vector_store_path=vector_db_path,
            model_provider="huggingface"  # Use free model for testing
        )
        
        # Test questions
        test_questions = [
            {
                'question': 'What is photosynthesis?',
                'grade': 6,
                'subject': 'Science',
                'language': 'english'
            },
            {
                'question': 'Explain the water cycle',
                'grade': 5,
                'subject': 'Science',
                'language': 'english'
            }
        ]
        
        for i, test in enumerate(test_questions, 1):
            logger.info(f"\n{'─'*80}")
            logger.info(f"Test Question {i}: {test['question']}")
            logger.info(f"Filters: Grade {test['grade']}, {test['subject']}, {test['language']}")
            
            result = self.doubt_solver.solve_doubt(**test)
            
            logger.info(f"\nAnswer:\n{result['answer']}")
            logger.info(f"\nSources ({result['metadata']['num_sources']}):")
            for j, source in enumerate(result['sources'], 1):
                logger.info(f"  {j}. {source['source']}")
        
        logger.info("\n✓ RAG pipeline test complete")
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        logger.info("="*80)
        logger.info("NCERT MULTILINGUAL DOUBT SOLVER - COMPLETE PIPELINE")
        logger.info("="*80)
        
        try:
            # Step 1: OCR
            self.step1_ocr_processing()
            
            # Step 2: Chunking
            self.step2_chunk_text()
            
            # Step 3: Vector Store
            self.step3_create_vector_store()
            
            # Step 4: Test RAG
            self.step4_test_rag()
            
            logger.info("\n" + "="*80)
            logger.info("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY")
            logger.info("="*80)
            logger.info("\nNext steps:")
            logger.info("1. Start the API server: python src/api/api_server.py")
            logger.info("2. Or use the doubt solver programmatically")
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)


def main():
    pipeline = NCERTPipeline()
    
    # You can run individual steps or the full pipeline
    import sys
    
    if len(sys.argv) > 1:
        step = sys.argv[1]
        if step == "ocr":
            pipeline.step1_ocr_processing()
        elif step == "chunk":
            pipeline.step2_chunk_text()
        elif step == "vector":
            pipeline.step3_create_vector_store()
        elif step == "test":
            pipeline.step4_test_rag()
        else:
            logger.error(f"Unknown step: {step}")
            logger.info("Available steps: ocr, chunk, vector, test")
    else:
        # Run full pipeline
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
