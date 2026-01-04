
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from pathlib import Path
import tiktoken

class NCERTChunker:
    def __init__(self, chunk_size=512, chunk_overlap=128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Custom separators for NCERT structure
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(self.encoding.encode(x)),
            separators=[
                "\n\n\n",  # Chapter breaks
                "\n\n",    # Section breaks
                "\n",      # Paragraph breaks
                ". ",      # Sentence breaks
                " ",       # Word breaks
                ""
            ]
        )
    
    def create_chunks(self, ocr_json_path):
        """Create chunks from OCR output"""
        
        with open(ocr_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data['metadata']
        chunks = []
        
        # Combine all pages
        full_text = ""
        page_mapping = {}
        
        for page in data['pages']:
            page_start = len(full_text)
            full_text += page['text'] + "\n\n"
            page_end = len(full_text)
            page_mapping[page['page_number']] = (page_start, page_end)
        
        # Create chunks
        text_chunks = self.splitter.split_text(full_text)
        
        # Add metadata to each chunk
        for idx, chunk_text in enumerate(text_chunks):
            # Find which page this chunk belongs to
            chunk_start = full_text.find(chunk_text)
            page_num = self._find_page_number(chunk_start, page_mapping)
            
            chunk_data = {
                "chunk_id": f"{metadata['grade']}_{metadata['subject']}_{metadata['language']}_chunk_{idx}",
                "text": chunk_text,
                "metadata": {
                    "grade": metadata['grade'],
                    "subject": metadata['subject'],
                    "language": metadata['language'],
                    "textbook": metadata['title'],
                    "page_number": page_num,
                    "chunk_index": idx
                }
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _find_page_number(self, position, page_mapping):
        """Find page number for given text position"""
        for page_num, (start, end) in page_mapping.items():
            if start <= position < end:
                return page_num
        return 1  # Default to first page

def main():
    chunker = NCERTChunker(chunk_size=512, chunk_overlap=128)
    
    ocr_dir = Path("data/processed/ocr_output")
    output_dir = Path("data/processed/chunks")
    output_dir.mkdir(exist_ok=True)
    
    for ocr_file in ocr_dir.glob("*.json"):
        print(f"\n📄 Chunking: {ocr_file.name}")
        
        chunks = chunker.create_chunks(str(ocr_file))
        
        # Save chunks
        output_file = output_dir / ocr_file.name.replace('.json', '_chunks.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Created {len(chunks)} chunks → {output_file}")

if __name__ == "__main__":
    main()
