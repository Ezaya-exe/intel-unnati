"""
NCERT Doubt Solver - Core Module
Implements conversation history, citations, and fallback logic
"""
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.vector_store import NCERTVectorStore
from utils.timer import Timer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NCERT-related keywords for scope checking
NCERT_KEYWORDS = [
    'ncert', 'textbook', 'chapter', 'class', 'grade', 'math', 'science',
    'social', 'history', 'geography', 'civics', 'economics', 'physics',
    'chemistry', 'biology', 'english', 'equation',
    'theorem', 'formula', 'definition', 'explain', 'what is', 'how to',
    'solve', 'calculate', 'find', 'prove', 'describe', 'compare',
    'photosynthesis', 'rational', 'integer', 'fraction', 'democracy',
    'constitution', 'climate', 'ecosystem', 'cell', 'atom', 'molecule',
    'water cycle', 'solar system', 'energy', 'force', 'motion',
]


def is_in_scope(question: str, context_results: List[Dict], threshold: float = 0.3) -> bool:
    """Check if question is within NCERT scope based on retrieved context relevance."""
    if not context_results:
        return False

    # Check if we got relevant context based on reranker score
    for chunk in context_results:
        score = chunk.get('relevance_score', chunk.get('score', 0.0))
        if score > threshold:
            return True

    return False


def format_citations(sources: List[Dict]) -> List[Dict]:
    """Format sources into citation format"""
    citations = []
    for i, source in enumerate(sources, 1):
        citation = {
            'index': i,
            'text': source.get('text', '')[:200] + '...' if len(source.get('text', '')) > 200 else source.get('text', ''),
            'grade': source.get('metadata', {}).get('grade', 'N/A'),
            'subject': source.get('metadata', {}).get('subject', 'N/A'),
            'page': source.get('metadata', {}).get('page', 'N/A'),
            'source_file': source.get('metadata', {}).get('source_file', 'N/A'),
        }
        citations.append(citation)
    return citations


class NCERTDoubtSolver:
    """
    Main doubt solver class with:
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
            raise

    def _build_context_with_history(self, question: str, context: str) -> str:
        """Build context string including recent conversation history."""
        if not self.conversation_history:
            return context

    # Compact history: last 2 turns, truncated
        history_parts = []
        for turn in self.conversation_history[-2:]:
            q = turn['question'][:150]
            a = turn['answer'][:200]
            history_parts.append(f"Q: {q}\nA: {a}")

        history_string = "\n".join(history_parts)

    # Context first, history second, with explicit instruction
        full_context = f"""NCERT Context:
    {context}

    Recent Conversation (use this to resolve vague references like "it", "this", "that"):
    {history_string}"""

        return full_context

    def ask(
        self,
        question: str,
        grade: Optional[int] = None,
        subject: Optional[str] = None,
        n_context_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Ask a doubt and get an answer (profiled version)
        """
        timer = Timer()
        start_time = time.time()

        # Step 1: Build filters
        with timer.step("1_build_filters"):
            filters = {}
            if grade:
                filters['grade'] = grade
            if subject:
                filters['subject'] = subject

        # Step 2: Retrieve context
        with timer.step("2_advanced_search_TOTAL"):
            search_results = self.vector_store.advanced_search(
                query=question,
                n_results=n_context_chunks,
                filters=filters if filters else None,
                use_hybrid=True,
                use_rerank=True,
                use_expansion=False
            )

        context_chunks = search_results.get('results', [])

        # Step 3: In-scope check
        with timer.step("3_in_scope_check"):
            in_scope = is_in_scope(question, context_chunks)

        # Step 4: Build context string
        with timer.step("4_build_context"):
            context_parts = []
            citations = []

            if len(context_chunks) > 0:
                citations = format_citations(context_chunks)
                for i, chunk in enumerate(context_chunks, 1):
                    source = f"[Source {i}: Grade {chunk['metadata'].get('grade', '?')}, {chunk['metadata'].get('subject', '?')}]"
                    context_parts.append(f"{source}\n{chunk['text']}")

            context_string = "\n\n".join(context_parts)

        # Step 5: Generate
        with timer.step("5_generation_TOTAL"):
            if not in_scope and len(context_chunks) == 0:
                answer = (
                    "I don't know the answer to this question based on the NCERT textbooks. "
                    "Please ask a question related to Grade 5-10 curriculum."
                )
            else:
                with timer.step("5a_build_context_with_history"):
                    full_context = self._build_context_with_history(question, context_string)
                with timer.step("5b_llm_call"):
                    answer = self._generate_answer(full_context, question)

        with timer.step("6_update_history"):
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
                'grade_filter': grade,
                'subject_filter': subject,
                'num_sources': len(citations),
                'in_scope': in_scope,
                'latency_seconds': round(latency, 2),
                'timings': timer.report()
            }
        }
    def ask_stream(
    self,
    question: str,
    grade: Optional[int] = None,
    subject: Optional[str] = None,
    n_context_chunks: int = 5
):
        """
    Streaming version of ask().
    Yields dicts with partial state:
        {"type": "citations", "citations": [...]}   # once, at start
        {"type": "token", "text": "..."}             # many times
        {"type": "done", "answer": "...", "metadata": {...}}
    """
        start_time = time.time()

    # Step 1: Build filters
        filters = {}
        if grade:
            filters['grade'] = grade
        if subject:
            filters['subject'] = subject

    # Step 2: Retrieve context
        search_results = self.vector_store.advanced_search(
        query=question,
        n_results=n_context_chunks,
        filters=filters if filters else None,
        use_hybrid=True,
        use_rerank=True,
        use_expansion=False
        )
        context_chunks = search_results.get('results', [])

    # Step 3: In-scope check
        in_scope = is_in_scope(question, context_chunks)

    # Step 4: Build context string + citations
        context_parts = []
        citations = []
        if len(context_chunks) > 0:
            citations = format_citations(context_chunks)
            for i, chunk in enumerate(context_chunks, 1):
                source = f"[Source {i}: Grade {chunk['metadata'].get('grade', '?')}, {chunk['metadata'].get('subject', '?')}]"
                context_parts.append(f"{source}\n{chunk['text']}")
        context_string = "\n\n".join(context_parts)

    # Emit citations early so UI can render them
        yield {"type": "citations", "citations": citations}

    # Step 5: Generate (streaming)
        if not in_scope and len(context_chunks) == 0:
            answer = (
            "I don't know the answer to this question based on the NCERT textbooks. "
            "Please ask a question related to Grade 5-10 curriculum."
        )
            yield {"type": "token", "text": answer}
        else:
            full_context = self._build_context_with_history(question, context_string)

            from generation.groq_generator import generate_answer_groq_stream

            full_answer = []
            for token in generate_answer_groq_stream(full_context, question):
                full_answer.append(token)
            yield {"type": "token", "text": token}
            answer = "".join(full_answer)

    # Step 6: Update history
        self.conversation_history.append({
            'question': question,
            'answer': answer[:500]
        })
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history:]

        latency = time.time() - start_time

        yield {
        "type": "done",
        "answer": answer,
        "metadata": {
            'grade_filter': grade,
            'subject_filter': subject,
            'num_sources': len(citations),
            'in_scope': in_scope,
            'latency_seconds': round(latency, 2),
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
    solver = NCERTDoubtSolver()

    print("\n" + "=" * 60)
    print("Test 1: NCERT Question")
    result = solver.ask("What is a rational number?", grade=8, subject="Mathematics")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"Citations: {len(result['citations'])}")
    print(f"Latency: {result['metadata']['latency_seconds']}s")
    print(f"In Scope: {result['metadata']['in_scope']}")

    print("\n" + "=" * 60)
    print("Test 2: Out-of-scope Question")
    result = solver.ask("Who won the FIFA World Cup 2022?")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer'][:300]}...")
    print(f"In Scope: {result['metadata']['in_scope']}")