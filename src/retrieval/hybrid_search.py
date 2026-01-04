"""
Hybrid Search Module - BM25 + Semantic Search with Reciprocal Rank Fusion
Combines keyword-based BM25 search with semantic vector search for improved retrieval.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Please install rank_bm25: pip install rank-bm25")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 index for keyword-based search"""
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize BM25 index
        
        Args:
            persist_path: Path to save/load the index
        """
        self.persist_path = persist_path
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self.doc_metadata = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing"""
        # Basic tokenization - split on non-alphanumeric
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def build_index(self, documents: List[str], doc_ids: List[str], 
                    metadatas: Optional[List[Dict]] = None):
        """
        Build BM25 index from documents
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs
            metadatas: Optional list of metadata dicts
        """
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        
        self.documents = documents
        self.doc_ids = doc_ids
        self.doc_metadata = metadatas or [{} for _ in documents]
        
        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logger.info("✓ BM25 index built successfully")
        
        # Persist if path provided
        if self.persist_path:
            self.save()
    
    def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search the BM25 index
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of results with scores
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built yet")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-n indices
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    'text': self.documents[idx],
                    'id': self.doc_ids[idx],
                    'metadata': self.doc_metadata[idx],
                    'bm25_score': float(scores[idx])
                })
        
        return results
    
    def save(self):
        """Save index to disk"""
        if not self.persist_path:
            return
            
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'doc_metadata': self.doc_metadata,
            'bm25': self.bm25
        }
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"✓ BM25 index saved to {self.persist_path}")
    
    def load(self) -> bool:
        """Load index from disk"""
        if not self.persist_path or not Path(self.persist_path).exists():
            return False
        
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.doc_ids = data['doc_ids']
            self.doc_metadata = data['doc_metadata']
            self.bm25 = data['bm25']
            
            logger.info(f"✓ BM25 index loaded from {self.persist_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False


class HybridSearcher:
    """
    Combines BM25 keyword search with semantic vector search
    using Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(self, vector_store, bm25_index: BM25Index, 
                 semantic_weight: float = 0.5, rrf_k: int = 60):
        """
        Initialize hybrid searcher
        
        Args:
            vector_store: NCERTVectorStore instance
            bm25_index: BM25Index instance
            semantic_weight: Weight for semantic search (0-1). BM25 weight = 1 - semantic_weight
            rrf_k: Constant for RRF formula (default 60)
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.semantic_weight = semantic_weight
        self.bm25_weight = 1 - semantic_weight
        self.rrf_k = rrf_k
    
    def _reciprocal_rank_fusion(self, rankings: List[List[str]], 
                                 scores_map: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute RRF scores for document IDs across multiple rankings
        
        RRF formula: score(d) = sum(1 / (k + rank(d)))
        
        Args:
            rankings: List of ranked document ID lists
            scores_map: Dict mapping doc_id -> {source: score}
            
        Returns:
            Dict mapping doc_id -> RRF score
        """
        rrf_scores = {}
        
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1 / (self.rrf_k + rank)
        
        return rrf_scores
    
    def search(self, query: str, n_results: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform hybrid search combining BM25 and semantic search
        
        Args:
            query: Search query
            n_results: Number of results to return
            filters: Metadata filters for semantic search
            
        Returns:
            Combined search results
        """
        # Get more candidates than needed for fusion
        n_candidates = n_results * 3
        
        # 1. BM25 Search
        bm25_results = self.bm25_index.search(query, n_results=n_candidates)
        bm25_ranking = [r['id'] for r in bm25_results]
        bm25_docs = {r['id']: r for r in bm25_results}
        
        # 2. Semantic Search
        semantic_results = self.vector_store.search(
            query, 
            n_results=n_candidates, 
            filters=filters
        )
        semantic_ranking = [r['id'] for r in semantic_results.get('results', [])]
        semantic_docs = {r['id']: r for r in semantic_results.get('results', [])}
        
        # 3. Reciprocal Rank Fusion
        rrf_scores = self._reciprocal_rank_fusion(
            [bm25_ranking, semantic_ranking],
            {}  # We're not using raw scores, just ranks
        )
        
        # 4. Sort by RRF score and get top results
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # 5. Build final results
        results = []
        for doc_id in sorted_ids[:n_results]:
            # Prefer semantic doc info (has embeddings), fall back to BM25
            doc_info = semantic_docs.get(doc_id) or bm25_docs.get(doc_id)
            
            if doc_info:
                result = {
                    'text': doc_info['text'],
                    'metadata': doc_info.get('metadata', {}),
                    'id': doc_id,
                    'rrf_score': rrf_scores[doc_id],
                    'in_bm25': doc_id in bm25_docs,
                    'in_semantic': doc_id in semantic_docs
                }
                
                # Add original scores if available
                if doc_id in bm25_docs:
                    result['bm25_score'] = bm25_docs[doc_id].get('bm25_score', 0)
                if doc_id in semantic_docs:
                    result['semantic_distance'] = semantic_docs[doc_id].get('distance', 0)
                
                results.append(result)
        
        return {
            'query': query,
            'results': results,
            'bm25_hits': len(bm25_results),
            'semantic_hits': len(semantic_results.get('results', []))
        }


def build_bm25_from_chromadb(vector_store, persist_path: str) -> BM25Index:
    """
    Build BM25 index from existing ChromaDB collection
    
    Args:
        vector_store: NCERTVectorStore instance
        persist_path: Path to save BM25 index
        
    Returns:
        BM25Index instance
    """
    logger.info("Building BM25 index from ChromaDB collection...")
    
    # Get all documents from ChromaDB
    collection = vector_store.collection
    count = collection.count()
    
    if count == 0:
        logger.warning("No documents in collection")
        return BM25Index(persist_path)
    
    # Fetch all documents (in batches if large)
    all_docs = collection.get(limit=count, include=['documents', 'metadatas'])
    
    documents = all_docs['documents']
    doc_ids = all_docs['ids']
    metadatas = all_docs['metadatas']
    
    # Build index
    bm25_index = BM25Index(persist_path)
    bm25_index.build_index(documents, doc_ids, metadatas)
    
    return bm25_index
