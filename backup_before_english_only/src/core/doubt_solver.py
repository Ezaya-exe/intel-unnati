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
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.vector_store import NCERTVectorStore
from utils.timer import Timer
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


def is_in_scope(question: str, context_results: List[Dict], threshold: float = 0.5) -> bool:
    """Check if question is within NCERT scope based on context and keywords"""
    question_lower = question.lower()
    
    # Check if any NCERT keywords are present
    has_keywords = any(kw in question_lower for kw in NCERT_KEYWORDS)
    
    # Check if we got relevant context based on reranker score
    has_relevant_context = False
    for chunk in context_results:
        # Cross-encoder typically returns logits or probabilities.
        # Assuming hybrid_search/reranker adds 'relevance_score' metadata or similar.
        score = chunk.get('relevance_score', chunk.get('score', 0.0))
        if score > threshold:
            has_relevant_context = True
            break
            
    if has_relevant_context:
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
        """Generate answer using Groq or local GGUF model."""
        try:
            import os
            provider = os.environ.get("GENERATION_PROVIDER", "groq").lower()

            if provider == "groq":
                from generation.groq_generator import generate_answer_groq
                result = generate_answer_groq(context, question)
            else:
                from generation.qwen_gguf import generate_answer_gguf
                result = generate_answer_gguf(context, question)

            if result.get("thinking"):
                logger.debug(f"Thinking Process:\n{result['thinking']}")

            return result["answer"]

        except Exception as e:
            logger.error(f"Generation error: {e}")
            # Do NOT silently return an error string as the answer —
            # that hides failures from the eval script.
            raise

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
        n_context_chunks: int = 3
        ) -> Dict[str, Any]:
        """
        Ask a doubt and get an answer (profiled version)
        """
        timer = Timer()
        start_time = time.time()
        
        # Step 1: Detect language
        with timer.step("1_language_detection"):
            detected_lang = detect_language(question)
            language = language or detected_lang
        
        # Step 2: Build filters
        with timer.step("2_build_filters"):
            filters = {}
            if grade:
                filters['grade'] = grade
            if subject:
                filters['subject'] = subject
        
        # Step 3: Retrieve context
        with timer.step("3_advanced_search_TOTAL"):
            search_results = self.vector_store.advanced_search(
                query=question,
                n_results=n_context_chunks,
                filters=filters if filters else None,
                use_hybrid=True,
                use_rerank=True,
                use_expansion=True
            )
        
        context_chunks = search_results.get('results', [])
        
        # Step 4: In-scope check
        with timer.step("4_in_scope_check"):
            in_scope = is_in_scope(question, context_chunks)
        
        # Step 5: Build context string
        with timer.step("5_build_context"):
            context_parts = []
            citations = []
            
            if len(context_chunks) > 0:
                citations = format_citations(context_chunks)
                for i, chunk in enumerate(context_chunks, 1):
                    source = f"[Source {i}: Grade {chunk['metadata'].get('grade', '?')}, {chunk['metadata'].get('subject', '?')}]"
                    context_parts.append(f"{source}\n{chunk['text']}")
            
            context_string = "\n\n".join(context_parts)
        
        # Step 6: Generate
        with timer.step("6_generation_TOTAL"):
            if not in_scope and len(context_chunks) == 0:
                answer = (
                    "I don't know the answer to this question based on the NCERT textbooks. "
                    "Please ask a question related to Grade 5-10 curriculum."
                )
            else:
                with timer.step("6a_build_context_with_history"):
                    full_context = self._build_context_with_history(question, context_string)
                with timer.step("6b_llm_call"):
                    answer = self._generate_answer(full_context, question)
        
        with timer.step("7_update_history"):
            self.conversation_history.append({
                'question': question,
                'answer': answer[:500]
            })
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history:]
        
        latency = time.time() - start_time
        
        timer.print_report()
        
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
                'latency_seconds': round(latency, 2),
                'timings': timer.report()
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
