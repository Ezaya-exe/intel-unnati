import fitz  # PyMuPDF
import os
from pathlib import Path
from tqdm import tqdm

def pdf_to_images(pdf_path, output_folder):
    """Convert PDF pages to images"""
    
    pdf_name = Path(pdf_path).stem
    page_folder = os.path.join(output_folder, pdf_name)
    os.makedirs(page_folder, exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        
        for page_num in tqdm(range(page_count), desc=f"Processing {pdf_name}"):
            page = doc[page_num]
            
            # Render page to image (150 DPI for good quality)
            pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
            
            # Save as PNG
            image_path = os.path.join(page_folder, f"page_{page_num+1:03d}.png")
            pix.save(image_path)
        
        print(f"✓ Converted {page_count} pages from {pdf_name}")
        
    except Exception as e:
        print(f"❌ Error processing {pdf_name}: {e}")
    
    finally:
        # Explicitly close document
        if 'doc' in locals():
            doc.close()

def process_all_pdfs():
    """Process all PDFs in raw_pdfs folder"""
    pdf_dir = "data/raw_pdfs"
    output_dir = "data/processed/images"
    
    if not os.path.exists(pdf_dir):
        print(f"❌ No PDFs found in {pdf_dir}")
        return
    
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        return
    
    print(f"📁 Found {len(pdf_files)} PDFs to process")
    
    for pdf_path in pdf_files:
        print(f"\n📄 Processing: {pdf_path.name}")
        pdf_to_images(str(pdf_path), output_dir)

if __name__ == "__main__":
    process_all_pdfs()
