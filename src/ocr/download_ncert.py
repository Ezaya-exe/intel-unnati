import requests
import os
from tqdm import tqdm

def download_ncert_pdfs():
    """Download NCERT textbooks from official website"""
    
    # Sample URLs (you'll need to expand this)
    ncert_urls = {
        "grade_5_math_en": "https://ncert.nic.in/textbook/pdf/eemh101.pdf",
        "grade_6_science_en": "https://ncert.nic.in/textbook/pdf/fesc101.pdf",
        "grade_8_math_hi": "https://ncert.nic.in/textbook/pdf/hemh101.pdf",
        # Add more URLs for all grades, subjects, languages
    }
    
    output_dir = "data/raw_pdfs"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, url in tqdm(ncert_urls.items()):
        output_path = os.path.join(output_dir, f"{filename}.pdf")
        
        if os.path.exists(output_path):
            print(f"✓ {filename} already exists")
            continue
            
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Downloaded: {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_ncert_pdfs()
