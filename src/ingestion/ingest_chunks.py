"""
Ingest scraped pages as raw text chunks (no LLM fact extraction) into the
mini_chunks Qdrant collection.

Usage
-----
    python -m src.ingestion.ingest_chunks
    python -m src.ingestion.ingest_chunks --scraped-dir /root/scraped_raw_backup
    python -m src.ingestion.ingest_chunks --scraped-dir src/data/subset --drop-existing
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.embedder import Embedder
from src.ingestion.vector_db import (
    COLLECTION_NAME_CHUNKS,
    ensure_chunks_collection,
    save_chunks_to_vector_db,
)
from src.utils.paths import get_data_dir

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SCRAPED_DIR = Path(PROJECT_ROOT) / "src" / "data" / "scraped_raw"
DATABASE_PATH = os.environ.get("QDRANT_CHUNKS_DIR", get_data_dir("qdrant_chunks_db"))
BATCH_SIZE = 64


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _words(text: str) -> list[str]:
    return text.split()


def _chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    segments: list[str] = []
    for para in paragraphs:
        para_words = _words(para)
        if len(para_words) <= CHUNK_SIZE:
            segments.append(para)
        else:
            sentences = _split_sentences(para)
            current_words: list[str] = []
            for sent in sentences:
                sent_words = _words(sent)
                if len(current_words) + len(sent_words) > CHUNK_SIZE and current_words:
                    segments.append(" ".join(current_words))
                    current_words = sent_words
                else:
                    current_words.extend(sent_words)
            if current_words:
                segments.append(" ".join(current_words))

    chunks: list[str] = []
    current_words: list[str] = []
    prev_tail: list[str] = []

    for seg in segments:
        seg_words = _words(seg)
        if len(current_words) + len(seg_words) <= CHUNK_SIZE:
            current_words.extend(seg_words)
        else:
            if current_words:
                chunk_text = " ".join(prev_tail + current_words) if prev_tail else " ".join(current_words)
                chunks.append(chunk_text)
                prev_tail = current_words[-CHUNK_OVERLAP:] if len(current_words) >= CHUNK_OVERLAP else current_words[:]
            current_words = seg_words

    if current_words:
        chunk_text = " ".join(prev_tail + current_words) if prev_tail else " ".join(current_words)
        chunks.append(chunk_text)

    return chunks


def _parse_file(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    url = ""
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("URL:"):
            url = stripped[len("URL:"):].strip()
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]).strip()
    return url, body


def _load_scraped_files(scraped_dir: Path, skip_schedules: bool = False) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    if not scraped_dir.exists():
        logger.error("Scraped directory not found: %s", scraped_dir)
        return records

    txt_files = sorted(scraped_dir.glob("*.txt"))
    logger.info("Found %d .txt files in %s", len(txt_files), scraped_dir)

    skipped = 0
    for fpath in txt_files:
        if skip_schedules and fpath.name.startswith("schedule_"):
            logger.debug("Skipping schedule file: %s", fpath.name)
            skipped += 1
            continue
        try:
            url, body = _parse_file(fpath)
            if url and body:
                records.append((url, body))
            else:
                logger.warning("Skipping %s: missing URL or body.", fpath.name)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", fpath.name, exc)

    if skipped:
        logger.info("Skipped %d schedule files.", skipped)
    logger.info("Loaded %d usable files.", len(records))
    return records


def ingest_chunks(
    scraped_dir: Path = SCRAPED_DIR,
    database_path: str = DATABASE_PATH,
    drop_existing: bool = False,
    skip_schedules: bool = False,
) -> int:
    """Ingest all scraped pages as chunks into mini_chunks. Returns total chunks ingested."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if drop_existing:
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(path=database_path)
            if client.collection_exists(COLLECTION_NAME_CHUNKS):
                client.delete_collection(COLLECTION_NAME_CHUNKS)
                logger.info("Dropped existing collection: %s", COLLECTION_NAME_CHUNKS)
        except Exception as exc:
            logger.warning("Could not drop collection: %s", exc)

    logger.info("Starting chunk ingestion from %s", scraped_dir)
    logger.info("Target Qdrant collection: %s at %s", COLLECTION_NAME_CHUNKS, database_path)

    ensure_chunks_collection(database_path)

    records = _load_scraped_files(scraped_dir, skip_schedules=skip_schedules)
    if not records:
        logger.error("No records to ingest.")
        return 0

    embedder = Embedder()

    all_texts: list[str] = []
    all_urls: list[str] = []

    for url, body in records:
        chunks = _chunk_text(body)
        if not chunks:
            continue
        all_texts.extend(chunks)
        all_urls.extend([url] * len(chunks))

    logger.info("Total chunks to embed: %d", len(all_texts))

    total_saved = 0
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i : i + BATCH_SIZE]
        batch_urls = all_urls[i : i + BATCH_SIZE]
        try:
            embeddings = embedder.generate_embeddings(batch_texts)
            save_chunks_to_vector_db(batch_texts, embeddings, batch_urls, database_path)
            total_saved += len(batch_texts)
            logger.info(
                "Saved batch %d/%d (%d chunks so far).",
                i // BATCH_SIZE + 1,
                (len(all_texts) + BATCH_SIZE - 1) // BATCH_SIZE,
                total_saved,
            )
        except Exception as exc:
            logger.error("Failed to save batch starting at index %d: %s", i, exc)

    logger.info("Chunk ingestion complete. Total chunks saved: %d", total_saved)
    return total_saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest scraped pages as text chunks into mini_chunks.")
    parser.add_argument(
        "--scraped-dir",
        default=str(SCRAPED_DIR),
        help="Path to directory with scraped .txt files.",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop the mini_chunks collection before ingestion.",
    )
    parser.add_argument(
        "--skip-schedules",
        action="store_true",
        help="Exclude schedule_*.txt files (by default they ARE included).",
    )
    args = parser.parse_args()
    ingest_chunks(
        scraped_dir=Path(args.scraped_dir),
        drop_existing=args.drop_existing,
        skip_schedules=args.skip_schedules,
    )
