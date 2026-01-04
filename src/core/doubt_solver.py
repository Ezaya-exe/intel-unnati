"""
NCERT Multilingual Doubt Solver - Core Module
Implements language detection, conversation history, citations, and fallback logic
"""
import sys
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from langdetect import detect, LangDetectException

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.vector_store import NCERTVectorStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WSL GGUF Server URL
WSL_SERVER_URL = "http://127.0.0.1:5000"

# Language mappings
LANGUAGE_NAMES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'ur': 'Urdu',
    'ta': 'Tamil',
    'te': 'Telugu',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'sa': 'Sanskrit'
}

# NCERT-related keywords for scope checking
NCERT_KEYWORDS = [
    'ncert', 'textbook', 'chapter', 'class', 'grade', 'math', 'science', 
    'social', 'history', 'geography', 'civics', 'economics', 'physics',
    'chemistry', 'biology', 'hindi', 'english', 'sanskrit', 'equation',
    'theorem', 'formula', 'definition', 'explain', 'what is', 'how to',
    'solve', 'calculate', 'find', 'prove', 'describe', 'compare',
    'photosynthesis', 'rational', 'integer', 'fraction', 'democracy',
    'constitution', 'climate', 'ecosystem', 'cell', 'atom', 'molecule'
]


def detect_language(text: str) -> str:
    """Detect the language of input text"""
    try:
        lang = detect(text)
        return lang if lang in LANGUAGE_NAMES else 'en'
    except LangDetectException:
        return 'en'


def is_in_scope(question: str, context_results: List[Dict]) -> bool:
    """Check if question is within NCERT scope based on context and keywords"""
    question_lower = question.lower()
    
    # Check if any NCERT keywords are present
    has_keywords = any(kw in question_lower for kw in NCERT_KEYWORDS)
    
    # Check if we got relevant context
    has_context = len(context_results) > 0
    
    # If we have context with decent similarity, it's likely in scope
    if has_context:
        # Check if any result has good relevance (low distance = good)
        return True
    
    return has_keywords


def format_citations(sources: List[Dict]) -> List[Dict]:
    """Format sources into citation format"""
    citations = []
    for i, source in enumerate(sources, 1):
        citation = {
            'index': i,
            'text': source.get('text', '')[:200] + '...' if len(source.get('text', '')) > 200 else source.get('text', ''),
            'grade': source.get('metadata', {}).get('grade', 'N/A'),
            'subject': source.get('metadata', {}).get('subject', 'N/A'),
            'language': source.get('metadata', {}).get('language', 'N/A'),
            'page': source.get('metadata', {}).get('page', 'N/A')
        }
        citations.append(citation)
    return citations


class NCERTDoubtSolver:
    """
    Main doubt solver class with:
    - Language detection
    - Conversation history
    - Citations
    - "I don't know" fallback
    """
    
    def __init__(self, use_wsl_server: bool = False):
        """Initialize the doubt solver"""
        logger.info("Initializing NCERT Doubt Solver...")
        
        # Initialize vector store
        self.vector_store = NCERTVectorStore()
        
        # Conversation history (per session)
        self.conversation_history: List[Dict] = []
        self.max_history = 5  # Keep last 5 turns
        
        # Native model flag
        self.use_native_model = True
        
        logger.info("Doubt Solver initialized!")
    
    def _generate_answer(self, context: str, question: str) -> str:
        """Generate answer using Native GGUF model (WSL)"""
        try:
            from generation.qwen_gguf import generate_answer_gguf
            result = generate_answer_gguf(context, question)
            # We can log thinking process or return it if UI supported it
            if result.get('thinking'):
                logger.debug(f"Thinking Process:\n{result['thinking']}")
            return result['answer']
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating answer: {str(e)}"

    def _build_context_with_history(self, question: str, context: str) -> str:
        """Build context string including recent conversation history"""
        if not self.conversation_history:
            return context
        
        # Include last few turns of conversation for continuity
        history_parts = []
        for turn in self.conversation_history[-self.max_history:]:
            history_parts.append(f"Previous Q: {turn['question']}\nPrevious A: {turn['answer']}")
        
        history_string = "\n\n".join(history_parts)
        
        # Combine history + new context
        full_context = f"=== Conversation History ===\n{history_string}\n\n=== Relevant NCERT Content ===\n{context}"
        return full_context
    
    def ask(
        self,
        question: str,
        grade: Optional[int] = None,
        subject: Optional[str] = None,
        language: Optional[str] = None,
        n_context_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Ask a doubt and get an answer
        """
        start_time = time.time()
        
        # Step 1: Detect language
        detected_lang = detect_language(question)
        language = language or detected_lang
        
        # Step 2: Build filters
        filters = {}
        if grade:
            filters['grade'] = grade
        if subject:
            filters['subject'] = subject
        
        # Step 3: Retrieve context using advanced search (hybrid + rerank + expansion)
        search_results = self.vector_store.advanced_search(
            query=question,
            n_results=n_context_chunks,
            filters=filters if filters else None,
            use_hybrid=True,
            use_rerank=True,
            use_expansion=True
        )
        
        context_chunks = search_results.get('results', [])
        in_scope = is_in_scope(question, context_chunks)
        
        # Step 4: Build context string
        context_parts = []
        citations = []
        
        if len(context_chunks) > 0:
            citations = format_citations(context_chunks)
            for i, chunk in enumerate(context_chunks, 1):
                source = f"[Source {i}: Grade {chunk['metadata'].get('grade', '?')}, {chunk['metadata'].get('subject', '?')}]"
                context_parts.append(f"{source}\n{chunk['text']}")
        
        context_string = "\n\n".join(context_parts)
        
        # Step 5: Generate
        if not in_scope and len(context_chunks) == 0:
            answer = (
                "I don't know the answer to this question based on the NCERT textbooks. "
                "Please ask a question related to Grade 5-10 curriculum."
            )
        else:
            # Add conversation history
            full_context = self._build_context_with_history(question, context_string)
            answer = self._generate_answer(full_context, question)
            
        latency = time.time() - start_time
        
        # Update history
        self.conversation_history.append({
            'question': question,
            'answer': answer[:500]
        })
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history:]
            
        return {
            'question': question,
            'answer': answer,
            'citations': citations,
            'metadata': {
                'detected_language': detected_lang,
                'response_language': language,
                'grade_filter': grade,
                'subject_filter': subject,
                'num_sources': len(citations),
                'in_scope': in_scope,
                'latency_seconds': round(latency, 2)
            }
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return self.vector_store.get_collection_stats()


# Quick test
if __name__ == "__main__":
    solver = NCERTDoubtSolver(use_wsl_server=True)
    
    # Test 1: In-scope question
    print("\n" + "="*60)
    print("Test 1: NCERT Question")
    result = solver.ask("What is a rational number?", grade=8, subject="Mathematics")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"Citations: {len(result['citations'])}")
    print(f"Latency: {result['metadata']['latency_seconds']}s")
    
    # Test 2: Out-of-scope question
    print("\n" + "="*60)
    print("Test 2: Out-of-scope Question")
    result = solver.ask("Who won the FIFA World Cup 2022?")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"In Scope: {result['metadata']['in_scope']}")
