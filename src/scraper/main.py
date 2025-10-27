import os
import logging

from scraper.modules.fetcher import fetch_file
from scraper.modules.parser import extract_links
from scraper.modules.uploader import save_new_files
from utils.paths import get_data_dir

BASE_URL = "https://ww2.mini.pw.edu.pl/"
OUTPUT_DIR = get_data_dir("raw", "scraper")
STORAGE_DIR = get_data_dir("final_storage")
# WILL REQUIRE MODIFICATION FOR ACTUAL UPLOADING TO A REMOTE SERVER.


os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)


def scrape_new_files():
    """
    Perform a complete scraping session.

    Steps:
    1. Deals with all websites, starting from BASE_URL.
    2. Fetch files (HTML, PDF, DOCX) using fetcher.py.
    3. Extract links from HTML files using parser.py.
    4. Upload all fetched files using uploader.py.
    """

    visited = set()
    to_visit = [BASE_URL]
    all_files = []

    while to_visit:

        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        file_path = fetch_file(url, OUTPUT_DIR)
        if file_path:
            all_files.append(file_path)
            if file_path.endswith(".html"):
                with open(file_path, "r", encoding="utf-8") as f:
                    f.readline()
                    links = extract_links(f.read(), BASE_URL)
                    to_visit.extend(links - visited)
    
    logging.info("Scraping complete. Uploading newly fetched files...")

    uploaded_files = save_new_files(all_files, STORAGE_DIR)
    logging.info("Successfully uploaded %d new files.", len(uploaded_files))