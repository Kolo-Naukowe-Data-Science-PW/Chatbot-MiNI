import os
import logging

from scraper.modules.fetcher import fetch_file
from scraper.modules.parser import extract_links
from scraper.modules.uploader import save_new_files

BASE_URL = "https://ww2.mini.pw.edu.pl/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "scraper")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "final_storage")
# WILL REQUIRE MODIFICATION FOR ACTUAL UPLOADING TO A REMOTE SERVER.
VISITED_FILE = os.path.join(OUTPUT_DIR, "visited.txt")


os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)


def load_visited():
    """
    Load the set of URLs that have already been visited from a local file.
    """

    if os.path.exists(VISITED_FILE):
        with open(VISITED_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()


def save_visited(visited):
    """
    Saves the set of visited URLs to a local file.
    """

    with open(VISITED_FILE, "w") as f:
        for url in visited:
            f.write(url + "\n")


def scrape_new_files():
    """
    Perform a complete scraping session.

    Steps:
    1. Load previously visited URLs.
    2. Start crawling from BASE_URL.
    3. Fetch files (HTML, PDF, DOCX) using fetcher.py.
    4. Extract new links from HTML files using parser.py.
    5. Upload newly fetched files using uploader.py.
    6. Update the visited URLs record.
    """

    visited = load_visited()
    to_visit = [BASE_URL]
    all_files = []

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        file_path = fetch_file(url, OUTPUT_DIR)
        if file_path:
            all_files.append(file_path)
            if file_path.endswith(".html"):
                with open(file_path, "r", encoding="utf-8") as f:
                    f.readline()
                    links = extract_links(f.read(), BASE_URL)
                    to_visit.extend(links - visited)
        visited.add(url)
    
    logging.info("Scraping complete. Uploading newly fetched files...")

    uploaded_files = save_new_files(all_files, STORAGE_DIR) 
    logging.info("Successfully uploaded %d new files.", len(uploaded_files))

    save_visited(visited)