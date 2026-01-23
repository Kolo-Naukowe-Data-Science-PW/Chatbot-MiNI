import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.rag_api.models import Message
from src.rag_api.modules.prompt_builder import build_messages
from src.rag_api.modules.retrieval import get_top_k_chunks

env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # Fallback to default behavior (search cwd and parents)
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
# Then put the API key in the .env file.

MODEL_NAME = "openai/gpt-4o-mini"


def query_llm(messages: list[dict[str, str]]) -> str:
    """
    Generates an answer using the OpenRouter API.

    Parameters
    ----------
    messages : list[dict[str, str]]
        A list of message dicts with 'role' and 'content' keys,
        containing system instructions, conversation history, and current query.

    Returns
    -------
    str
        The generated text response from the LLM.
    """
    try:
        logger.debug("Sending request to OpenRouter model: %s", MODEL_NAME)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
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
    generating answers, and printing them to stdout. Maintains conversation history
    throughout the session.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    logger.info("RAG API script started.")
    logger.info(f"Using Model: {MODEL_NAME}")

    conversation_history: list[Message] = []

    while True:
        query = input(
            "\nEnter your query (or 'q' to quit, 'clear' to reset history): "
        ).strip()

        if query.lower() == "q":
            break
        if query.lower() == "clear":
            conversation_history = []
            logger.info("Conversation history cleared.")
            print("History cleared.")
            continue
        if not query:
            logger.error("Query cannot be empty.")
            continue

        logger.info("Retrieving top K chunks...")

        sorted_chunks = get_top_k_chunks(query)

        if not sorted_chunks:
            answer = "No relevant information found in the database."
            print(answer)
            conversation_history.append(Message(role="user", content=query))
            conversation_history.append(Message(role="assistant", content=answer))
            continue

        text_only_chunks = [chunk["text_chunk"] for chunk in sorted_chunks]
        messages = build_messages(
            query, text_only_chunks, conversation_history=conversation_history
        )

        print("\nThinking...")
        answer = query_llm(messages)

        print("\n=== Answer ===")
        print(answer)

        # Update conversation history
        conversation_history.append(Message(role="user", content=query))
        conversation_history.append(Message(role="assistant", content=answer))

        logger.info(
            f"Conversation history now has {len(conversation_history)} messages."
        )


if __name__ == "__main__":
    main()
