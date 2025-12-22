import json
import os
import pandas as pd
from docx import Document

from src.pipeline.common import logger, get_config, CURRENT_VERSION

config = get_config()

INPUT_DIR = "src/data/complex_files"
OUTPUT_DIR = "src/data/processed_text"


def process_xlsx(path):
    """
    Ingests Excel files, extracting text from each sheet.
    Returns a formatted string with sheet names and their content.
    """

    try:
        xls = pd.ExcelFile(path)
        text = f"Source: Excel file {os.path.basename(path)}\n"
        for sheet in xls.sheet_names:
            text += f"\nSheet: {sheet}\n"
            df = pd.read_excel(xls, sheet_name=sheet)
            text += df.to_markdown(index=False)
        return text

    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        return None


def process_docx(path):
    """
    Ingests Word documents, extracting text from paragraphs.
    Returns a formatted string with the document name and its content.
    """

    try:
        doc = Document(path)
        text = f"Source: Word document {os.path.basename(path)}\n"
        text += "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text

    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        return None


def save_text_and_meta(filename, text):
    """
    Saves the extracted text and metadata to the output directory.
    Metadata includes the filename FOR NOW but it needs to be changed to the actual URL.
    """

    base_name = filename

    txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    meta_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
    meta_data = {"source_url": filename}  # SHOULD BE CHANGED TO THE ACTUAL URL
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    logger.info(f"Processed: {filename}")


def main():
    """
    Main function to process files in the input directory and save processed text files.
    1. Processes Excel and Word files.
    2. Saves the extracted text in the output directory.
    3. Logs progress and errors.
    """

    if not config["process_complex_files"]:
        logger.info(
            f"SKIP: Pipeline Version {CURRENT_VERSION} does not support complex files (XLSX/DOCX)."
        )
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        logger.warning(f"Input directory does not exist: {INPUT_DIR}")
        return

    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)
        content = None

        if filename.endswith(".xlsx"):
            logger.info(f"Processing a XLSX file: {filename}")
            content = process_xlsx(file_path)
        elif filename.endswith(".docx"):
            logger.info(f"Processing a DOCX file: {filename}")
            content = process_docx(file_path)
        # in the future, more file types can be added here
        # for example scan PDFs, images, etc.

        if content:
            save_text_and_meta(filename, content, file_path)


if __name__ == "__main__":
    main()
