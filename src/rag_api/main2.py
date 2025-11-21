import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_api.modules.prompt_builder import build_prompt
from rag_api.modules.retrieval import get_top_k_chunks
from src.rag_api.modules.logs import setup_logging

LOG_FILE_PATH = os.getenv("RAG_LOG_FILE", "logs/rag_api.log")
setup_logging(log_file=LOG_FILE_PATH)

logger = logging.getLogger(__name__)

MODEL_NAME = "bigscience/bloom-560m"
logger.info("Loading tokenizer: %s", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

logger.info("Loading model: %s", MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
logger.info("Model and tokenizer loaded successfully.")


def query_llm(prompt: str, max_tokens: int = 300) -> str:
    """
    Generates an answer from the Hugging Face model.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.5,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    answer = output_text[len(prompt) :].strip()

    return answer


def main():
    logger.info("RAG API script started.")

    while True:
        query = input("\nEnter your query (or 'q' to quit): ").strip()

        if query.lower() == "q":
            break
        if not query:
            logger.error("Query cannot be empty.")
            continue

        logger.info("Retrieving top K chunks...")

        sorted_chunks = get_top_k_chunks(query)

        if not sorted_chunks:
            print("No relevant information found.")
            continue

        text_only_chunks = [chunk["text_chunk"] for chunk in sorted_chunks]
        prompt = build_prompt(query, text_only_chunks)

        print("\nThinking...")
        answer = query_llm(prompt)

        print("\n=== Answer ===")
        print(answer)

        print("\n=== Sources ===")
        for i, chunk_data in enumerate(sorted_chunks, start=1):
            source = chunk_data.get("source_url", "Unknown")
            print(f"{i}. {source}")


if __name__ == "__main__":
    main()
