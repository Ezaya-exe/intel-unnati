import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NCERTVectorStore:
    """ChromaDB-based vector store for NCERT textbooks"""
    
    def __init__(self, persist_directory="data/vector_db", model_name="paraphrase-multilingual-mpnet-base-v2"):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Path to persist the database
            model_name: Name of the sentence-transformer model for embeddings
        """
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(exist_ok=True, parents=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize embedding model 
        # WSL with CUDA support -> Use CUDA
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading embedding model: {model_name} on {device.upper()}")
        self.embedding_model = SentenceTransformer(f"sentence-transformers/{model_name}", device=device)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="ncert_textbooks",
            metadata={"description": "NCERT textbooks for grades 5-10 in multiple languages"}
        )
        
        logger.info(f"Initialized vector store at {persist_directory}")
        logger.info(f"Collection contains {self.collection.count()} documents")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add text chunks to the vector store
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'chunk_id', and 'metadata'
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk['text'])
            metadatas.append(chunk['metadata'])
            ids.append(chunk['chunk_id'])
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedding_model.encode(
            documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        
        # Add to collection
        logger.info("Adding to ChromaDB collection...")
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✓ Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, n_results: int = 5, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Search for relevant chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            filters: Metadata filters (e.g., {'grade': 5, 'subject': 'Mathematics'})
        
        Returns:
            Dictionary with results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).tolist()[0]
        
        # Build where clause for filtering
        where = None
        if filters:
            # Handle multiple filters with logical AND
            conditions = [{k: v} for k, v in filters.items()]
            if len(conditions) > 1:
                where = {"$and": conditions}
            elif len(conditions) == 1:
                where = conditions[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = {
            'query': query,
            'results': []
        }
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results['results'].append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'id': results['ids'][0][i]
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        # Sample some metadata to understand distribution
        if count > 0:
            sample = self.collection.get(limit=min(100, count))
            
            # Count by grade, subject, language
            grades = {}
            subjects = {}
            languages = {}
            
            for meta in sample['metadatas']:
                grades[meta.get('grade')] = grades.get(meta.get('grade'), 0) + 1
                subjects[meta.get('subject')] = subjects.get(meta.get('subject'), 0) + 1
                languages[meta.get('language')] = languages.get(meta.get('language'), 0) + 1
            
            return {
                'total_chunks': count,
                'grades': grades,
                'subjects': subjects,
                'languages': languages
            }
        
        return {'total_chunks': 0}
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection("ncert_textbooks")
        logger.info("Collection deleted")


def main():
    """Load chunks and populate vector store"""
    vector_store = NCERTVectorStore()
    
    chunks_dir = Path("data/processed/chunks")
    
    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        return
    
    # Process all chunk files
    for chunks_file in chunks_dir.glob("*_chunks.json"):
        logger.info(f"\n📚 Loading chunks from: {chunks_file.name}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        vector_store.add_chunks(chunks)
    
    # Print statistics
    stats = vector_store.get_collection_stats()
    logger.info(f"\n📊 Vector Store Statistics:")
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Grades: {stats.get('grades', {})}")
    logger.info(f"Subjects: {stats.get('subjects', {})}")
    logger.info(f"Languages: {stats.get('languages', {})}")
    
    # Test search
    test_query = "What is photosynthesis?"
    logger.info(f"\n🔍 Test search: '{test_query}'")
    results = vector_store.search(test_query, n_results=3)
    
    for i, result in enumerate(results['results'], 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"  Grade: {result['metadata'].get('grade')}")
        logger.info(f"  Subject: {result['metadata'].get('subject')}")
        logger.info(f"  Language: {result['metadata'].get('language')}")
        logger.info(f"  Text: {result['text'][:200]}...")


if __name__ == "__main__":
    main()
