import pytesseract
import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np  # ← ADD THIS LINE


# Tesseract path (Windows - update if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class NCERTOCRProcessor:
    def __init__(self):
        print("✅ Tesseract OCR ready")
    
    def preprocess_image(self, image_path):
        """Clean up image for better OCR"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise + sharpen
        denoised = cv2.fastNlMeansDenoising(gray)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Binarization
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def extract_text_from_image(self, image_path):
        """Extract text from single image using Tesseract"""
        processed_img = self.preprocess_image(image_path)
        
        # Tesseract OCR with custom config
        config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?()+-*/= '
        text = pytesseract.image_to_string(processed_img, config=config)
        
        return {
            "full_text": text.strip(),
            "confidence": "N/A"  # Tesseract doesn't provide per-line confidence easily
        }
    
    def process_textbook(self, textbook_folder, metadata):
        """Process entire textbook folder"""
        image_files = sorted(Path(textbook_folder).glob("*.png"))
        
        textbook_data = {
            "metadata": metadata,
            "pages": []
        }
        
        for idx, image_path in enumerate(tqdm(image_files, desc=f"OCR {metadata['title']}")):
            page_num = idx + 1
            
            ocr_result = self.extract_text_from_image(str(image_path))
            
            textbook_data["pages"].append({
                "page_number": page_num,
                "text": ocr_result["full_text"],
                "confidence": ocr_result["confidence"]
            })
        
        return textbook_data

def main():
    processor = NCERTOCRProcessor()
    
    # UPDATE THIS for your PDF (eemh101 or whatever you downloaded)
    textbooks = [
        {
            "folder": "data/processed/images/grade_8_math_hi",  # ← CHANGE THIS TO YOUR FOLDER NAME
            "metadata": {
                "grade": 8,
                "subject": "Mathematics",
                "language": "Hindi",
                "title": "Mathematics - Class 8"
            }
        }
    ]
    
    output_dir = "data/processed/ocr_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for textbook in textbooks:
        print(f"\n📖 Processing: {textbook['metadata']['title']}")
        
        if not os.path.exists(textbook["folder"]):
            print(f"❌ Folder not found: {textbook['folder']}")
            continue
            
        result = processor.process_textbook(
            textbook["folder"],
            textbook["metadata"]
        )
        
        # Save to JSON
        output_file = os.path.join(
            output_dir,
            f"grade_{textbook['metadata']['grade']}_{textbook['metadata']['subject']}_{textbook['metadata']['language']}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved to: {output_file}")

if __name__ == "__main__":
    main()
