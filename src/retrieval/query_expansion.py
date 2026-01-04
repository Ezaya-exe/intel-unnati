"""
Query Expansion Module - Generate related search terms to improve retrieval
Uses LLM to generate synonyms and related concepts, with caching.
"""

import logging
import re
from typing import List, Dict, Optional
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Predefined expansions for common NCERT topics (no LLM needed)
STATIC_EXPANSIONS = {
    # Science
    "photosynthesis": ["chlorophyll", "sunlight", "carbon dioxide", "glucose", "plants", "leaves"],
    "respiration": ["breathing", "oxygen", "cellular respiration", "mitochondria", "energy"],
    "evaporation": ["water cycle", "condensation", "precipitation", "boiling", "vapour"],
    "magnetism": ["magnetic field", "compass", "poles", "electromagnet", "attraction"],
    "electricity": ["current", "voltage", "resistance", "circuit", "conductor"],
    "force": ["motion", "Newton", "friction", "gravity", "acceleration"],
    "atom": ["molecule", "electron", "proton", "neutron", "nucleus", "element"],
    "cell": ["nucleus", "cytoplasm", "membrane", "mitochondria", "organelle"],
    
    # Mathematics
    "pythagoras": ["pythagorean theorem", "right triangle", "hypotenuse", "a² + b² = c²"],
    "rational numbers": ["integers", "fractions", "p/q form", "irrational", "number line"],
    "quadratic": ["quadratic equation", "ax² + bx + c", "roots", "factorization", "discriminant"],
    "triangle": ["vertices", "angles", "sides", "congruence", "similarity", "area"],
    "circle": ["radius", "diameter", "circumference", "area", "chord", "arc"],
    "algebra": ["variables", "equations", "expressions", "polynomial", "linear"],
    
    # Social Science
    "democracy": ["voting", "elections", "constitution", "rights", "government"],
    "french revolution": ["1789", "Bastille", "Napoleon", "liberty", "equality"],
    "world war": ["WWI", "WWII", "Treaty of Versailles", "Hitler", "allies"],
    "india independence": ["1947", "freedom struggle", "Gandhi", "partition", "British rule"],
    "constitution": ["fundamental rights", "directive principles", "preamble", "amendments"],
    
    # Hindi
    "कविता": ["poem", "poetry", "छंद", "रस", "अलंकार"],
    "व्याकरण": ["grammar", "संज्ञा", "सर्वनाम", "क्रिया", "विशेषण"],
}


class QueryExpander:
    """
    Expands queries with related terms to improve retrieval coverage.
    Uses a combination of static lookups and LLM-based generation.
    """
    
    def __init__(self, use_llm: bool = True, cache_path: Optional[str] = None):
        """
        Initialize query expander
        
        Args:
            use_llm: Whether to use LLM for expansion (slower but better)
            cache_path: Path to cache expansion results
        """
        self.use_llm = use_llm
        self.cache_path = cache_path
        self.cache = {}
        
        # Load cache if exists
        if cache_path and Path(cache_path).exists():
            try:
                with open(cache_path, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached expansions")
            except:
                pass
        
        # LLM for dynamic expansion
        self._llm = None
    
    def _get_llm(self):
        """Lazy load LLM"""
        if self._llm is None and self.use_llm:
            try:
                from generation.qwen_gguf import load_model
                self._llm = load_model()
            except Exception as e:
                logger.warning(f"Failed to load LLM for query expansion: {e}")
                self.use_llm = False
        return self._llm
    
    def _static_expand(self, query: str) -> List[str]:
        """Get static expansions for known terms"""
        query_lower = query.lower()
        expansions = []
        
        for term, related in STATIC_EXPANSIONS.items():
            if term in query_lower:
                expansions.extend(related)
        
        return list(set(expansions))
    
    def _llm_expand(self, query: str) -> List[str]:
        """Use LLM to generate related terms"""
        llm = self._get_llm()
        
        if llm is None:
            return []
        
        prompt = f"""<|im_start|>system
You are a search query expansion assistant. Given a student's question, generate 3-5 related search terms that would help find relevant information in NCERT textbooks.
Only output the terms, one per line. No explanations.<|im_end|>
<|im_start|>user
Question: {query}

Related search terms:<|im_end|>
<|im_start|>assistant
"""
        
        try:
            output = llm(
                prompt,
                max_tokens=100,
                stop=["<|im_end|>"],
                echo=False,
                temperature=0.3
            )
            
            text = output['choices'][0]['text']
            
            # Parse terms (one per line)
            terms = [t.strip() for t in text.strip().split('\n') if t.strip()]
            
            # Clean up
            terms = [re.sub(r'^[\d\.\-\*]+\s*', '', t) for t in terms]  # Remove numbering
            terms = [t for t in terms if len(t) > 2 and len(t) < 50]
            
            return terms[:5]
            
        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}")
            return []
    
    def expand(self, query: str, max_terms: int = 5) -> List[str]:
        """
        Expand query with related terms
        
        Args:
            query: Original search query
            max_terms: Maximum number of expansion terms
            
        Returns:
            List of expansion terms (not including original query)
        """
        # Check cache
        cache_key = query.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key][:max_terms]
        
        expansions = []
        
        # 1. Static expansions (fast)
        static = self._static_expand(query)
        expansions.extend(static)
        
        # 2. LLM expansions (if enabled and needed)
        if self.use_llm and len(expansions) < max_terms:
            llm_terms = self._llm_expand(query)
            expansions.extend(llm_terms)
        
        # Deduplicate and limit
        expansions = list(dict.fromkeys(expansions))[:max_terms]
        
        # Cache result
        self.cache[cache_key] = expansions
        self._save_cache()
        
        return expansions
    
    def expand_with_query(self, query: str, max_terms: int = 5) -> List[str]:
        """
        Get expansion terms plus the original query
        
        Returns:
            List with original query first, then expansion terms
        """
        expansions = self.expand(query, max_terms)
        return [query] + expansions
    
    def get_expanded_query_string(self, query: str, max_terms: int = 3) -> str:
        """
        Get a single expanded query string
        
        Example: "What is evaporation" -> "What is evaporation water cycle condensation"
        """
        terms = self.expand(query, max_terms)
        if terms:
            return f"{query} {' '.join(terms)}"
        return query
    
    def _save_cache(self):
        """Save cache to disk"""
        if self.cache_path:
            try:
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, 'w') as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")


class MultiQueryRetriever:
    """
    Retrieves results for multiple query variants and combines them
    """
    
    def __init__(self, searcher, expander: QueryExpander):
        """
        Initialize multi-query retriever
        
        Args:
            searcher: Search function (e.g., HybridSearcher.search)
            expander: QueryExpander instance
        """
        self.searcher = searcher
        self.expander = expander
    
    def search(self, query: str, n_results: int = 5, 
               expansion_queries: int = 2, **kwargs) -> Dict:
        """
        Search with query expansion
        
        Args:
            query: Original query
            n_results: Final number of results
            expansion_queries: Number of expansion variants to search
            **kwargs: Passed to searcher
            
        Returns:
            Combined search results
        """
        # Get expansion terms
        expansions = self.expander.expand(query, max_terms=expansion_queries)
        
        # Search original query
        all_results = {}
        main_results = self.searcher(query, n_results=n_results * 2, **kwargs)
        
        for r in main_results.get('results', []):
            doc_id = r['id']
            if doc_id not in all_results:
                all_results[doc_id] = r
                all_results[doc_id]['query_hits'] = 1
            else:
                all_results[doc_id]['query_hits'] += 1
        
        # Search expansion queries
        for exp_term in expansions:
            exp_query = f"{query} {exp_term}"
            exp_results = self.searcher(exp_query, n_results=n_results, **kwargs)
            
            for r in exp_results.get('results', []):
                doc_id = r['id']
                if doc_id not in all_results:
                    all_results[doc_id] = r
                    all_results[doc_id]['query_hits'] = 1
                else:
                    all_results[doc_id]['query_hits'] += 1
        
        # Sort by query_hits (more hits = more relevant)
        sorted_results = sorted(
            all_results.values(), 
            key=lambda x: (x.get('query_hits', 0), x.get('rrf_score', 0)),
            reverse=True
        )
        
        return {
            'query': query,
            'expansions': expansions,
            'results': sorted_results[:n_results]
        }
