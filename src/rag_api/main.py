import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from rag_api.modules.prompt_builder import build_prompt
from rag_api.modules.retrieval import get_top_k_chunks

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not found in environment variables.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
# Highly recommended for usage with RAG, because it's free and has a good performance.
# In order to run it, one needs to create an account on OpenRouter and get the API key.
# Then put the API key in the .env file

MODEL_NAME = "mistralai/mistral-7b-instruct:free" # "openai/gpt-oss-20b:free"


def query_llm(prompt: str) -> str:
    """
    Generates an answer using the OpenRouter API.

    Parameters
    ----------
    prompt : str
        The full prompt string containing the system instructions,
        context, and user query.

    Returns
    -------
    str
        The generated text response from the LLM.
    """
    try:
        logger.debug("Sending request to OpenRouter model: %s", MODEL_NAME)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500,
        )

        answer = completion.choices[0].message.content.strip()
        logger.debug("LLM query successful.")
        return answer

    except Exception as e:
        logger.error("Failed to query OpenRouter: %s", e)
        return "Sorry, I encountered an error while generating the response."


def main() -> None:
    """
    Runs the interactive command-line interface (CLI) for the RAG API.

    Loops indefinitely, accepting user queries via stdin, retrieving context,
    generating answers, and printing them to stdout.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logger.info("RAG API script started.")
    logger.info(f"Using Model: {MODEL_NAME}")

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
            print("No relevant information found in the database.")
            continue

        text_only_chunks = [chunk["text_chunk"] for chunk in sorted_chunks]
        prompt = build_prompt(query, text_only_chunks)

        print("\nThinking...")
        answer = query_llm(prompt)

        print("\n=== Answer ===")
        print(answer)


if __name__ == "__main__":
    main()