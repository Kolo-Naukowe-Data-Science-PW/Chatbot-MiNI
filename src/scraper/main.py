import json
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from scraper.modules.fetcher import fetch_file
from scraper.modules.parser import extract_links
from utils.paths import get_data_dir

BASE_URL = "https://ww2.mini.pw.edu.pl/"
OUTPUT_DIR = get_data_dir("raw", "scraper")
STORAGE_DIR = get_data_dir("final_storage")
# WILL REQUIRE MODIFICATION FOR ACTUAL UPLOADING TO A REMOTE SERVER.


os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)


def get_processed_urls(storage_dir: str) -> set[str]:
    """
    Scans the storage directory for .json metadata files
    to find out which URLs have already been scraped.
    """
    processed_urls = set()

    if not os.path.exists(storage_dir):
        return processed_urls

    for filename in os.listdir(storage_dir):
        if filename.endswith(".json"):
            meta_path = os.path.join(storage_dir, filename)
            try:
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
                    url = data.get("source_url")
                    if url:
                        processed_urls.add(url)
            except Exception as e:
                logging.warning(f"Could not read metadata from {filename}: {e}")

    return processed_urls


def process_url(url: str, storage_dir: str, session: requests.Session) -> list[str]:
    """
    Worker function to fetch a single URL and extract links if it's HTML.
    """
    new_links = []
    file_path = fetch_file(url, storage_dir, session=session)

    if file_path and file_path.endswith(".html"):
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()

            new_links = list(extract_links(content, url))
        except Exception as e:
            logging.error(f"Error parsing links from {file_path}: {e}")

    return new_links


def scrape_new_files(start_urls: list[str]):
    """
    Main scraping loop using BFS + ThreadPoolExecutor + Session Reuse.
    """
    processed_urls = get_processed_urls(STORAGE_DIR)
    logging.info(f"Found {len(processed_urls)} already processed URLs.")

    queue = deque(start_urls)

    visited = set(processed_urls)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # max_workers=10 allows up to 10 simultaneous downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        while queue:
            current_batch = []

            while queue:
                url = queue.popleft()
                if url not in visited:
                    visited.add(url)
                    current_batch.append(url)

            if not current_batch:
                break

            logging.info(f"Processing batch of {len(current_batch)} URLs...")

            future_to_url = {
                executor.submit(process_url, url, STORAGE_DIR, session): url
                for url in current_batch
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    found_links = future.result()
                    for link in found_links:
                        if link not in visited:
                            queue.append(link)
                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")

    logging.info("Scraping complete. Uploading newly fetched files...")


if __name__ == "__main__":
    scrape_new_files([BASE_URL])
