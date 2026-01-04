from typing import Dict, Any, Optional
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.vector_store import NCERTVectorStore


class NCERTRetriever:
    """Simple retriever wrapper around NCERTVectorStore"""
    
    def __init__(self, vector_store_path="data/vector_db"):
        self.vector_store = NCERTVectorStore(persist_directory=vector_store_path)
    
    def retrieve(self, query, grade=None, subject=None, language=None, n_results=5):
        """Retrieve relevant chunks from the vector store"""
        filters = {}
        if grade is not None:
            filters['grade'] = grade
        if subject is not None:
            filters['subject'] = subject
        if language is not None:
            filters['language'] = language
        
        results = self.vector_store.search(
            query=query, 
            n_results=n_results, 
            filters=filters if filters else None
        )
        
        return {
            'query': query,
            'context': [
                {
                    'text': r['text'],
                    'source': f"Grade {r['metadata'].get('grade', '?')} - {r['metadata'].get('subject', '?')}",
                    'grade': r['metadata'].get('grade', 0),
                    'subject': r['metadata'].get('subject', ''),
                    'language': r['metadata'].get('language', '')
                }
                for r in results.get('results', [])
            ]
        }
    
    def get_context_string(self, retrieval_result):
        """Format retrieved context as a string"""
        contexts = []
        for i, chunk in enumerate(retrieval_result.get('context', []), 1):
            contexts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
        return "\n\n".join(contexts)
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NCERTDoubtSolver:
    """
    LLM-based answer generation component for NCERT doubt solver
    Uses RAG pattern: Retrieve relevant context → Generate answer
    """
    
    def __init__(
        self, 
        vector_store_path="data/vector_db",
        model_provider="huggingface",  # Options: "openai", "huggingface", "ollama"
        model_name=None
    ):
        """
        Initialize doubt solver
        
        Args:
            vector_store_path: Path to vector database
            model_provider: LLM provider to use
            model_name: Specific model name
        """
        # Initialize retriever
        self.retriever = NCERTRetriever(vector_store_path=vector_store_path)
        
        # Initialize LLM based on provider
        self.model_provider = model_provider
        self._initialize_llm(model_name)
        
        logger.info(f"Doubt solver initialized with {model_provider}")
    
    def _initialize_llm(self, model_name: Optional[str] = None):
        """Initialize the LLM based on provider"""
        
        if self.model_provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = model_name or "gpt-3.5-turbo"
            
        elif self.model_provider == "huggingface":
            from transformers import pipeline
            # Using a free, smaller model for demonstration
            model_name = model_name or "google/flan-t5-base"
            logger.info(f"Loading HuggingFace model: {model_name}")
            import torch
            device = 0 if torch.cuda.is_available() else -1  # GPU if available, else CPU
            self.llm = pipeline(
                "text2text-generation",
                model=model_name,
                device=device
            )
            self.model_name = model_name
            
        elif self.model_provider == "ollama":
            # For local Ollama installation
            import requests
            self.ollama_url = "http://localhost:11434/api/generate"
            self.model_name = model_name or "llama2"
            
        else:
            raise ValueError(f"Unknown model provider: {self.model_provider}")
    
    def solve_doubt(
        self,
        question: str,
        grade: Optional[int] = None,
        subject: Optional[str] = None,
        language: Optional[str] = None,
        n_context_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Solve a student's doubt using RAG
        
        Args:
            question: Student's question
            grade: Student's grade (5-10)
            subject: Subject area
            language: Preferred language
            n_context_chunks: Number of context chunks to retrieve
        
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Solving doubt: {question[:100]}...")
        
        # Step 1: Retrieve relevant context
        retrieval_result = self.retriever.retrieve(
            query=question,
            grade=grade,
            subject=subject,
            language=language,
            n_results=n_context_chunks
        )
        
        context_str = self.retriever.get_context_string(retrieval_result)
        
        # Step 2: Generate answer using LLM
        answer = self._generate_answer(question, context_str, language)
        
        # Step 3: Format response
        response = {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'text': chunk['text'][:200] + '...',
                    'source': chunk['source'],
                    'grade': chunk['grade'],
                    'subject': chunk['subject']
                }
                for chunk in retrieval_result['context']
            ],
            'metadata': {
                'grade': grade,
                'subject': subject,
                'language': language,
                'num_sources': len(retrieval_result['context']),
                'model': f"{self.model_provider}/{self.model_name}"
            }
        }
        
        return response
    
    def _generate_answer(self, question: str, context: str, language: Optional[str] = None) -> str:
        """Generate answer using the configured LLM"""
        
        # Create prompt
        language_instruction = f" Answer in {language}." if language and language != 'english' else ""
        
        prompt = f"""You are a helpful NCERT textbook tutor for Indian students in grades 5-10. 
Answer the student's question based on the provided context from NCERT textbooks.
Be clear, accurate, and educational. Use simple language appropriate for the student's grade level.{language_instruction}

Context from NCERT textbooks:
{context}

Student's Question: {question}

Answer:"""
        
        # Generate based on provider
        if self.model_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful NCERT textbook tutor for Indian students."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        
        elif self.model_provider == "huggingface":
            result = self.llm(
                prompt,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
            return result[0]['generated_text']
        
        elif self.model_provider == "ollama":
            import requests
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json().get('response', 'Error generating response')
        
        return "Error: No valid LLM configured"


def main():
    """Test the doubt solver"""
    
    # Initialize (use HuggingFace for free testing)
    solver = NCERTDoubtSolver(model_provider="huggingface")
    
    # Test questions
    test_questions = [
        {
            'question': 'What is photosynthesis?',
            'grade': 6,
            'subject': 'Science',
            'language': 'english'
        },
        {
            'question': 'How do I solve a quadratic equation?',
            'grade': 10,
            'subject': 'Mathematics',
            'language': 'english'
        }
    ]
    
    for test in test_questions:
        logger.info(f"\n{'='*80}")
        logger.info(f"Question: {test['question']}")
        
        result = solver.solve_doubt(**test)
        
        logger.info(f"\nAnswer:\n{result['answer']}")
        logger.info(f"\nSources used: {result['metadata']['num_sources']}")
        for i, source in enumerate(result['sources'], 1):
            logger.info(f"  {i}. {source['source']}")


if __name__ == "__main__":
    main()
