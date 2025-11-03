"""
Module for cleaning HTML files from scraped data.
"""

import html
import os
import re
import logging
from bs4 import BeautifulSoup


def clean_html(text: str | None) -> str:
    """
    Clean a single HTML or text string by removing tags, scripts, and normalizing whitespace.

    Args:
        text (str | None): Raw HTML or text input.

    Returns:
        str: Clean, readable text with no tags and normalized spacing.
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


# raczej nie będzie używane
def clean_txt_folder(input_dir: str, output_dir: str):
    """
    Clean all .txt files in input_dir and save results in output_dir.

    Args:
        input_dir (str): Folder with raw TXT files (relative path).
        output_dir (str): Folder to save cleaned TXT files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir, filename.replace(".txt", "_clean.txt")
            )

            print(f"Cleaning: {filename}")

            with open(input_path, encoding="utf-8") as f:
                raw_content = f.read()

            cleaned_text = clean_html(raw_content)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"Saved: {output_path}")

    print("\nAll TXT files have been cleaned and saved in:", output_dir)
