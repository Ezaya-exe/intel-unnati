#!/usr/bin/env python3
"""
NCERT Textbook Downloader - Downloads all textbooks for Classes 5-10
Organizes them by grade in the data folder.
"""
import os
import requests
import zipfile
from pathlib import Path
import time

# Base paths
BASE_DIR = Path("/mnt/d/study/python/intel unnati/data")
RAW_PDFS_DIR = BASE_DIR / "raw_pdfs"

# NCERT textbooks organized by grade (English medium + Hindi + Sanskrit)
# Format: book_code: (subject, title)
TEXTBOOKS = {
    "5": {
        # Class 5 (New NCF 2024-25)
        "eemm1": ("Mathematics", "Math-Mela"),
        "eesa1": ("English", "Santoor"),
        "ehve1": ("Hindi", "Veena"),
        "eeev1": ("EVS", "Our Wondrous World"),
    },
    "6": {
        # Class 6 (New NCF 2024-25)
        "fegp1": ("Mathematics", "Ganita Prakash"),
        "fecu1": ("Science", "Curiosity"),
        "fees1": ("Social Science", "Exploring Society"),
        "fepr1": ("English", "Poorvi"),
        "fhml1": ("Hindi", "Malhar"),
        "fsrj1": ("Sanskrit", "Ranjini-1"),
    },
    "7": {
        # Class 7 (New NCF 2024-25)
        "gegp1": ("Mathematics", "Ganita Prakash Part-1"),
        "gegp2": ("Mathematics", "Ganita Prakash Part-2"),
        "gecu1": ("Science", "Curiosity"),
        "gees1": ("Social Science", "Exploring Society Part-1"),
        "gees2": ("Social Science", "Exploring Society Part-2"),
        "gepr1": ("English", "Poorvi"),
        "ghml1": ("Hindi", "Malhar"),
        "gsrj1": ("Sanskrit", "Ranjini-1 Part-1"),
        "gsrj2": ("Sanskrit", "Ranjini-1 Part-2"),
    },
    "8": {
        # Class 8 (New NCF 2024-25)
        "hegp1": ("Mathematics", "Ganita Prakash Part-1"),
        "hegp2": ("Mathematics", "Ganita Prakash Part-2"),
        "hecu1": ("Science", "Curiosity"),
        "hees1": ("Social Science", "Exploring Society"),
        "hepr1": ("English", "Poorvi"),
        "hhml1": ("Hindi", "Malhar"),
        "hsrj1": ("Sanskrit", "Ranjini-1"),
    },
    "9": {
        # Class 9 (Secondary)
        "iemh1": ("Mathematics", "Mathematics"),
        "iesc1": ("Science", "Science"),
        "iess1": ("Social Science", "Democratic Politics-I"),
        "iess2": ("Social Science", "Economics"),
        "iess3": ("Social Science", "India and Contemporary World-I"),
        "iess4": ("Social Science", "Contemporary India-I"),
        "iebe1": ("English", "Beehive"),
        "iemo1": ("English", "Moments"),
        "ihkk1": ("Hindi", "Kshitij-1"),
        "ihks1": ("Hindi", "Kritika-1"),
        "ihsp1": ("Hindi", "Sparsh-1"),
        "ihsn1": ("Hindi", "Sanchayan-1"),
        "isrr1": ("Sanskrit", "Shemushi-1"),
        "isab1": ("Sanskrit", "Abhyaswaan Bhav-1"),
    },
    "10": {
        # Class 10 (Secondary)
        "jemh1": ("Mathematics", "Mathematics"),
        "jesc1": ("Science", "Science"),
        "jess1": ("Social Science", "Contemporary India-II"),
        "jess2": ("Social Science", "Democratic Politics-II"),
        "jess3": ("Social Science", "Understanding Economic Development"),
        "jess4": ("Social Science", "India and Contemporary World-II"),
        "jeff1": ("English", "First Flight"),
        "jefo1": ("English", "Footprints without Feet"),
        "jhkk1": ("Hindi", "Kshitij-2"),
        "jhks1": ("Hindi", "Kritika-2"),
        "jhsp1": ("Hindi", "Sparsh-2"),
        "jhsn1": ("Hindi", "Sanchayan-2"),
        "jsrr1": ("Sanskrit", "Shemushi-2"),
        "jsab1": ("Sanskrit", "Abhyaswaan Bhav-2"),
    }
}

def make_safe_filename(name):
    """Create a safe filename from book title"""
    return name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")

def download_textbook(book_code, grade, subject, title, retries=3):
    """Download a single textbook ZIP and extract PDFs"""
    url = f"https://ncert.nic.in/textbook/pdf/{book_code}dd.zip"
    
    grade_dir = RAW_PDFS_DIR / f"grade_{grade}"
    grade_dir.mkdir(parents=True, exist_ok=True)
    
    safe_title = make_safe_filename(title)
    zip_path = grade_dir / f"{book_code}.zip"
    
    print(f"  📥 Downloading: {title} ({book_code})...")
    
    for attempt in range(retries):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=120, stream=True)
            response.raise_for_status()
            
            # Save ZIP
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract ZIP
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract to grade folder
                    for member in zip_ref.namelist():
                        if member.endswith('.pdf'):
                            # Rename to include subject prefix
                            filename = os.path.basename(member)
                            new_name = f"{subject.replace(' ', '_')}_{safe_title}_{filename}"
                            
                            # Extract and rename
                            with zip_ref.open(member) as source:
                                target_path = grade_dir / new_name
                                with open(target_path, 'wb') as target:
                                    target.write(source.read())
                            print(f"    ✅ Extracted: {new_name}")
                
                # Clean up ZIP
                os.remove(zip_path)
                
            except zipfile.BadZipFile:
                print(f"    ⚠️ Bad ZIP file, keeping as-is")
                # Rename the file with proper extension if it's actually a PDF
                final_path = grade_dir / f"{subject.replace(' ', '_')}_{safe_title}.zip"
                os.rename(zip_path, final_path)
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"    ⚠️ Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
            continue
    
    print(f"    ❌ Failed to download: {title}")
    return False

def main():
    print("=" * 60)
    print("🏫 NCERT Textbook Downloader")
    print("=" * 60)
    print(f"\n📁 Download directory: {RAW_PDFS_DIR}")
    
    # Create base directory
    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    
    total = sum(len(books) for books in TEXTBOOKS.values())
    downloaded = 0
    failed = 0
    
    for grade, books in sorted(TEXTBOOKS.items()):
        print(f"\n📚 Grade {grade} ({len(books)} textbooks)")
        print("-" * 40)
        
        for book_code, (subject, title) in books.items():
            success = download_textbook(book_code, grade, subject, title)
            if success:
                downloaded += 1
            else:
                failed += 1
            
            # Be nice to the server
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print(f"✅ Downloaded: {downloaded}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    print(f"📁 Location: {RAW_PDFS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
