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
    Generates an answer from a free Hugging Face model.
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
    logger.debug("LLM query successful, answer generated.")
    return answer


def main():
    logger.info("RAG API script started.")
    query = input("Enter your query: ").strip()

    if not query:
        logger.error("Query cannot be empty.")
        return

    logger.info("New query received: '%s'", query)

    logger.info("Retrieving top K chunks...")
    sorted_chunks = get_top_k_chunks(query)

    text_chunks = [chunk["text_chunk"] for chunk in sorted_chunks]

    prompt = build_prompt(query, text_chunks)

    answer = query_llm(prompt)

    print("\n=== Answer ===")
    print(answer)
    print("\n=== Sources ===")
    for i, chunk in enumerate(text_chunks, start=1):
        source_info = chunk.get("source_url", "Unknown source")
        print(f"{i}. {source_info}")

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
