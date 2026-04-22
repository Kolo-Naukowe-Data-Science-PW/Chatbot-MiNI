import logging
import os
import random

from dotenv import load_dotenv
from openai import OpenAI

from src.api.models import Message
from src.api.prompt_builder import build_messages
from src.api.retrieval import get_top_k_chunks

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


MODEL_NAME = "openai/gpt-4o-mini"

AVAILABLE_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3-8b-instruct",
]


def query_llm(messages: list[dict[str, str]], model_config: dict | None = None) -> str:
    """
    Generates an answer using the OpenRouter API.
    Parameters: a list of messages (conversation history and current query) formatted for the LLM.
                model_config: Optional dictionary containing model parameters (e.g., temperature, max_tokens).
    Returns: the generated answer as a string. If an error occurs, returns an error message.
    """

    config = model_config or {}

    chosen_model = config.get("model")
    if not chosen_model:
        chosen_model = random.choice(AVAILABLE_MODELS)

    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 500)
    top_p = config.get("top_p", 1.0)
    frequency_penalty = config.get("frequency_penalty", 0.0)
    presence_penalty = config.get("presence_penalty", 0.0)

    try:
        logger.info(
            f"Sending request to OpenRouter model: {chosen_model} (Temp: {temperature})"
        )

        completion = client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
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

        conversation_history.append(Message(role="user", content=query))
        conversation_history.append(Message(role="assistant", content=answer))

        logger.info(
            f"Conversation history now has {len(conversation_history)} messages."
        )


if __name__ == "__main__":
    main()
