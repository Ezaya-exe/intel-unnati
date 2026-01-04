"""
Reranker Module - Cross-Encoder based reranking for improved relevance
Uses ms-marco-MiniLM model to reorder retrieved candidates by relevance to query.
"""

import logging
from typing import List, Dict, Any, Optional
import torch

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder based reranker for improving retrieval quality.
    Reorders candidates based on relevance scores computed by the cross-encoder.
    """
    
    # Model options (ordered by speed/accuracy tradeoff)
    MODELS = {
        'fast': 'cross-encoder/ms-marco-MiniLM-L-6-v2',     # Fast, good quality
        'balanced': 'cross-encoder/ms-marco-MiniLM-L-12-v2', # Balanced
        'accurate': 'cross-encoder/ms-marco-TinyBERT-L-2-v2' # Tiny but effective
    }
    
    def __init__(self, model_name: str = 'fast', device: Optional[str] = None):
        """
        Initialize the reranker
        
        Args:
            model_name: One of 'fast', 'balanced', 'accurate' or a full model path
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Resolve model name
        if model_name in self.MODELS:
            model_path = self.MODELS[model_name]
        else:
            model_path = model_name
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        
        logger.info(f"Loading reranker model: {model_path} on {device.upper()}")
        
        # Load cross-encoder
        self.model = CrossEncoder(model_path, device=device)
        
        logger.info("✓ Reranker model loaded successfully")
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]], 
               top_k: Optional[int] = None, 
               score_key: str = 'rerank_score') -> List[Dict[str, Any]]:
        """
        Rerank candidates based on relevance to query
        
        Args:
            query: The search query
            candidates: List of candidate documents (must have 'text' key)
            top_k: Number of top results to return (None = return all)
            score_key: Key to store reranking score in results
            
        Returns:
            Reranked list of candidates with scores
        """
        if not candidates:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, c['text']) for c in candidates]
        
        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Add scores to candidates
        for i, candidate in enumerate(candidates):
            candidate[score_key] = float(scores[i])
        
        # Sort by rerank score (descending)
        reranked = sorted(candidates, key=lambda x: x[score_key], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked
    
    def rerank_with_threshold(self, query: str, candidates: List[Dict[str, Any]],
                               threshold: float = 0.0,
                               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank and filter candidates below a score threshold
        
        Args:
            query: The search query
            candidates: List of candidate documents
            threshold: Minimum score to include in results
            top_k: Number of top results to return
            
        Returns:
            Filtered and reranked candidates
        """
        reranked = self.rerank(query, candidates, top_k=None)
        
        # Filter by threshold
        filtered = [c for c in reranked if c.get('rerank_score', 0) >= threshold]
        
        if top_k is not None:
            filtered = filtered[:top_k]
        
        return filtered
    
    def batch_rerank(self, queries: List[str], 
                     candidates_list: List[List[Dict[str, Any]]],
                     top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple query-candidates pairs in batch
        
        Args:
            queries: List of queries
            candidates_list: List of candidate lists (one per query)
            top_k: Number of top results per query
            
        Returns:
            List of reranked candidate lists
        """
        results = []
        
        for query, candidates in zip(queries, candidates_list):
            reranked = self.rerank(query, candidates, top_k=top_k)
            results.append(reranked)
        
        return results


class CachedReranker(Reranker):
    """
    Reranker with caching for repeated queries
    """
    
    def __init__(self, model_name: str = 'fast', device: Optional[str] = None,
                 cache_size: int = 1000):
        super().__init__(model_name, device)
        
        from functools import lru_cache
        
        # Cache for query-doc pairs
        self._cache = {}
        self._cache_size = cache_size
    
    def _get_cache_key(self, query: str, text: str) -> str:
        """Generate cache key from query and text"""
        return f"{hash(query)}_{hash(text)}"
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]], 
               top_k: Optional[int] = None,
               score_key: str = 'rerank_score') -> List[Dict[str, Any]]:
        """Rerank with caching"""
        
        # Check cache for each candidate
        uncached_pairs = []
        uncached_indices = []
        scores = [None] * len(candidates)
        
        for i, candidate in enumerate(candidates):
            cache_key = self._get_cache_key(query, candidate['text'])
            
            if cache_key in self._cache:
                scores[i] = self._cache[cache_key]
            else:
                uncached_pairs.append((query, candidate['text']))
                uncached_indices.append(i)
        
        # Compute scores for uncached pairs
        if uncached_pairs:
            new_scores = self.model.predict(uncached_pairs, show_progress_bar=False)
            
            for idx, score in zip(uncached_indices, new_scores):
                scores[idx] = float(score)
                cache_key = self._get_cache_key(query, candidates[idx]['text'])
                
                # Add to cache (simple LRU-like behavior)
                if len(self._cache) >= self._cache_size:
                    # Remove oldest entry
                    self._cache.pop(next(iter(self._cache)))
                
                self._cache[cache_key] = float(score)
        
        # Add scores to candidates
        for i, candidate in enumerate(candidates):
            candidate[score_key] = scores[i]
        
        # Sort and return
        reranked = sorted(candidates, key=lambda x: x[score_key], reverse=True)
        
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked
