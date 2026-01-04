import chromadb
import json
from pathlib import Path

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "ncert_chunks"

def create_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop old collection if exists (for rebuilds)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    embeddings_dir = Path("data/processed/embeddings")
    all_ids = []
    all_texts = []
    all_embeddings = []
    all_metadatas = []

    for emb_file in embeddings_dir.glob("*_embeddings.json"):
        print(f"Loading {emb_file.name}")
        with open(emb_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            all_ids.append(chunk["chunk_id"])
            all_texts.append(chunk["text"])
            all_embeddings.append(chunk["embedding"])
            all_metadatas.append(chunk["metadata"])

    print(f"Adding {len(all_ids)} chunks to Chroma...")
    collection.add(
        ids=all_ids,
        documents=all_texts,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )

    print("✅ Chroma collection created!")
    return collection

if __name__ == "__main__":
    create_chroma_collection()
