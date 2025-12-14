import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_api.modules.logs import setup_logging

LOG_FILE_PATH = os.getenv("RAG_LOG_FILE", "logs/rag_api.log")
setup_logging(log_file=LOG_FILE_PATH)

logger = logging.getLogger(__name__)

MODEL_NAME = "bigscience/bloom-560m"

logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

logger.info(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def query_llm(prompt: str, max_tokens: int = 300) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(query_llm("Hello"))
