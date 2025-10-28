"""
Module for cleaning TXT files from scraped data.
"""

import html
import os
import re

from bs4 import BeautifulSoup

from utils.paths import get_data_dir


def remove_html_tags(text: str | None) -> str:
    """
    Remove HTML tags and decode HTML entities from a string.
    Args:
        text (Optional[str]): The input HTML or text content.

    Returns:
        str: Cleaned text with HTML removed and whitespace normalized.
    """
    if not text:
        return ""

    soup = BeautifulSoup(text, "html.parser")

    for script in soup(["script", "style"]):
        script.decompose()

    extracted = soup.get_text(separator=" ", strip=True)
    unescaped = html.unescape(extracted)
    cleaned = re.sub(r"\s+", " ", unescaped).strip()

    return cleaned


def clean_txt_folder(input_dir: str, output_dir: str):
    """
    Clean all .txt files in input_dir and save results in output_dir.

    Args:
        input_dir (str): Folder with raw TXT files (relative path).
        output_dir (str): Folder to save cleaned TXT files.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(os.getcwd())

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir, filename.replace(".txt", "_clean.txt")
            )

            print(f"Cleaning: {filename}")

            with open(input_path, encoding="utf-8") as f:
                raw_content = f.read()

            cleaned_text = remove_html_tags(raw_content)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"Saved: {output_path}")

    print("\nAll TXT files have been cleaned and saved in:", output_dir)


if __name__ == "__main__":
    # --------------------------
    # Folder paths
    # --------------------------

    INPUT_DIR = get_data_dir("raw", "ignore", "scraped-html")
    OUTPUT_DIR = get_data_dir("cleaned")

    clean_txt_folder(INPUT_DIR, OUTPUT_DIR)
