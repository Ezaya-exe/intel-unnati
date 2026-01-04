import chromadb
from sentence_transformers import SentenceTransformer
import json

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "ncert_chunks"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

# Same model you used to create embeddings
embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def build_where_filter(grade=None, subject=None, language=None):
    """Build ChromaDB-compatible where filter"""
    conditions = []
    
    if grade is not None:
        conditions.append({"grade": {"$eq": int(grade)}})
    if subject is not None:
        conditions.append({"subject": {"$eq": subject}})
    if language is not None:
        conditions.append({"language": {"$eq": language}})
    
    if not conditions:
        return None
    
    # Use $and for multiple conditions
    return {"$and": conditions}

def search_chunks(user_query, grade=None, subject=None, language=None, top_k=8):
    # Embed query
    query_emb = embed_model.encode([user_query])[0].tolist()

    # Build proper ChromaDB filter
    where_filter = build_where_filter(grade, subject, language)

    print(f"🔍 Query: '{user_query}'")
    print(f"   Filter: {where_filter}")

    res = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where=where_filter,
    )

    # Convert to list of dicts
    results = []
    for i in range(len(res["ids"][0])):
        results.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i] if "distances" in res else None,
        })
    
    return results

def rerank_chunks(query, chunks):
    """Simple rerank - return as-is for now"""
    return chunks[:5]

if __name__ == "__main__":
    # Test it
    results = search_chunks(
        "What is Pythagoras theorem?", 
        grade=8, 
        subject="Mathematics", 
        language="English"
    )
    print(f"✅ Found {len(results)} chunks")
    for r in results[:2]:
        print(f"Text: {r['text'][:100]}...")
        print(f"Metadata: {r['metadata']}\n")
