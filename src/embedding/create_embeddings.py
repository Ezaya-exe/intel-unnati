import cohere
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingGenerator:
    def __init__(self, method="cohere"):
        self.method = method
        
        if method == "cohere":
            # Get API key from environment
            self.co = cohere.Client(os.getenv("COHERE_API_KEY"))
            self.model_name = "embed-multilingual-v3.0"
        else:
            # Use local SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    def generate_embeddings(self, texts, batch_size=96):
        """Generate embeddings for list of texts"""
        
        if self.method == "cohere":
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
                batch = texts[i:i+batch_size]
                
                response = self.co.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type="search_document"
                )
                
                all_embeddings.extend(response.embeddings)
            
            return np.array(all_embeddings)
        else:
            # Local model
            return self.model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    
    def process_chunks(self, chunks_json_path):
        """Generate embeddings for all chunks"""
        
        with open(chunks_json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()
        
        return chunks

def main():
    # Choose method: "cohere" (requires API key) or "local" (free, offline)
    generator = EmbeddingGenerator(method="local")  # Change to "cohere" if you have API key
    
    chunks_dir = Path("data/processed/chunks")
    output_dir = Path("data/processed/embeddings")
    output_dir.mkdir(exist_ok=True)
    
    for chunks_file in chunks_dir.glob("*_chunks.json"):
        print(f"\n🔢 Embedding: {chunks_file.name}")
        
        chunks_with_embeddings = generator.process_chunks(str(chunks_file))
        
        # Save
        output_file = output_dir / chunks_file.name.replace('_chunks.json', '_embeddings.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_with_embeddings, f, indent=2)
        
        print(f"✓ Saved {len(chunks_with_embeddings)} embeddings → {output_file}")

if __name__ == "__main__":
    main()
