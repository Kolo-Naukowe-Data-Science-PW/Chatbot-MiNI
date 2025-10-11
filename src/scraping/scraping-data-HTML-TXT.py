"""
Web crawler for scraping HTML and PDF content from a website.

This module uses BeautifulSoup for HTML parsing and LangChain loaders for
PDF/website content extraction. All extracted content is saved in TXT and HTML
format in a temporary directory and then zipped into a single archive.

Author: Barbara Gawlik
"""


from bs4 import BeautifulSoup
import hashlib
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
import logging
import os
import requests
import shutil
from typing import List
from urllib.parse import urljoin, urlparse, urldefrag


TOTAL_SIZE = 0
WEBSITE_URL = "https://ww2.mini.pw.edu.pl/"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "raw", "ignore", "scraped-temp")
ZIP_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "ignore", "scraped_data.zip")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)

visited = set()
to_visit = [WEBSITE_URL]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def get_safe_name(url: str, max_len: int = 100) -> str:
    """
    Generate a filesystem-safe filename based on the URL.
    Args:
        url (str): The URL to convert.
        max_len (int): Maximum length of the filename.
    Returns:
        str: Safe filename derived from the URL.
    """
    parsed_url = urlparse(url)
    base_name = parsed_url.netloc.replace(".", "_") + parsed_url.path.replace("/", "_")
    base_name = base_name.replace(":", "_")
    if len(base_name) > max_len:
        hash_part = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        base_name = base_name[:50] + "_" + hash_part
    return base_name


def save_docs(url: str, output_dir: str = OUTPUT_DIR):
    """
    Load content from a URL (HTML or PDF) and save as TXT and HTML files.
    Args:
        url (str): The URL to scrape.
        output_dir (str): Directory to save the files.
    """

    global TOTAL_SIZE
    safe_name = get_safe_name(url)
    logging.info("Processing URL: %s", url)
    full_text_content = ""
    documents_to_save: List[Document] = []
    main_title = safe_name

    try:

        if url.lower().endswith(".pdf"):
            logging.info("Detected PDF. Loading with PyPDFLoader...")
            loader = PyPDFLoader(url)
            documents = loader.load()
            full_text_content = "\n\n".join([doc.page_content for doc in documents])
            full_text_content = "Warning! This is PDF content.\n\n" + full_text_content
            single_document = Document(
                page_content=full_text_content,
                metadata={
                    "source": url,
                    "title": f"PDF Content from {safe_name}",
                    "source_type": "PDF",
                },
            )
            documents_to_save = [single_document]
            main_title = single_document.metadata.get("title", safe_name)

        else:
            logging.info("Detected website. Loading with WebBaseLoader...")
            loader = WebBaseLoader(url)
            documents = loader.load()
            if not documents:
                logging.warning("No content found at URL: %s", url)
                return
            main_doc = documents[0]
            full_text_content = main_doc.page_content
            documents_to_save = documents
            main_title = main_doc.metadata.get("title", safe_name)

    except Exception as exc:
        logging.error("Failed to load URL %s: %s", url, exc)
        return

    txt_path = os.path.join(output_dir, f"{safe_name}.txt")

    try:

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text_content)
        TOTAL_SIZE += os.path.getsize(txt_path)

    except Exception as exc:
        logging.error("Failed to save TXT for URL %s: %s", url, exc)

    html_path = os.path.join(output_dir, f"{safe_name}.html")

    try:

        if url.lower().endswith(".pdf"):
            html_content = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>{main_title}</title></head><body>"
                f"<h1>{main_title}</h1>"
                "<p>PDF content extracted from website.</p>"
                f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{full_text_content}</pre>"
                "</body></html>"
            )

        else:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                link = urljoin(url, a_tag["href"])
                norm_link, _ = urldefrag(link)
                if (
                    norm_link.startswith("http")
                    and not norm_link.startswith("javascript:")
                    and urlparse(norm_link).netloc == urlparse(WEBSITE_URL).netloc
                    and norm_link not in visited
                    and norm_link not in to_visit
                ):
                    to_visit.append(norm_link)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        TOTAL_SIZE += os.path.getsize(html_path)

    except Exception as exc:
        logging.error("Failed to save HTML for URL %s: %s", url, exc)

    logging.info("Total size so far: %.2f KB", TOTAL_SIZE / 1024)


def main():
    
    while to_visit:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
        save_docs(current_url)
        visited.add(current_url)

    logging.info("Packing data into ZIP: %s", ZIP_PATH)
    shutil.make_archive(ZIP_PATH.replace(".zip", ""), "zip", OUTPUT_DIR)

    shutil.rmtree(OUTPUT_DIR)
    logging.info("Temporary scraped files deleted.")
    logging.info("Scraping completed. Data archived at: %s", ZIP_PATH)


if __name__ == "__main__":
    main()