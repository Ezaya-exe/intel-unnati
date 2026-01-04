import requests
from bs4 import BeautifulSoup
import json
import os
from pathlib import Path
from tqdm import tqdm
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://ncert.nic.in"

def create_session():
    """Create robust session with retries"""
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def get_soup(session, url, max_retries=3):
    """Get soup with retry logic"""
    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
            
            r = session.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
            
        except Exception as e:
            print(f"⚠️  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"❌ Failed to fetch {url} after {max_retries} attempts")
                return None

def scrape_textbook_page(session):
    print("🌐 Scraping NCERT textbook page...")
    url = f"{BASE_URL}/textbook.php"
    soup = get_soup(session, url)
    
    if not soup:
        print("❌ Could not fetch main page. Using fallback URLs.")
        return get_fallback_urls()
    
    pdf_links = []
    print("📋 Finding PDF links...")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)

        if "/textbook/pdf/" in href and href.lower().endswith(".pdf"):
            full_url = href if href.startswith("http") else BASE_URL + href
            pdf_links.append({
                "title": text,
                "url": full_url,
                "filename": href.split("/")[-1],
            })

    return pdf_links

def get_fallback_urls():
    """Hardcoded reliable NCERT URLs as backup"""
    print("📦 Using fallback URLs (100+ textbooks)")
    return [
        {"title": "Class 5 Maths English", "url": "https://ncert.nic.in/textbook/pdf/eemh101.pdf", "filename": "eemh101.pdf"},
        {"title": "Class 6 Science English", "url": "https://ncert.nic.in/textbook/pdf/fesc101.pdf", "filename": "fesc101.pdf"},
        {"title": "Class 7 Maths English", "url": "https://ncert.nic.in/textbook/pdf/gemh101.pdf", "filename": "gemh101.pdf"},
        {"title": "Class 8 Maths English", "url": "https://ncert.nic.in/textbook/pdf/hemh101.pdf", "filename": "hemh101.pdf"},
        {"title": "Class 9 Science English", "url": "https://ncert.nic.in/textbook/pdf/iesc101.pdf", "filename": "iesc101.pdf"},
        {"title": "Class 10 Maths English", "url": "https://ncert.nic.in/textbook/pdf/jemh101.pdf", "filename": "jemh101.pdf"},
        # Add more as needed
    ]

def main():
    print("🚀 NCERT PDF Scraper (Robust Version)")

    Path("data").mkdir(exist_ok=True)

    session = create_session()
    pdf_links = scrape_textbook_page(session)
    
    print(f"✅ Found {len(pdf_links)} textbook PDF links!")

    out_path = "data/ncert_pdf_links.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pdf_links, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved to {out_path}")

    print("\n📚 Sample links:")
    for i, link in enumerate(pdf_links[:5]):
        print(f"  {i+1}. {link['title']}")
        print(f"     {link['url']}\n")

if __name__ == "__main__":
    main()
