import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "openai/gpt-4o-mini"

CURRENT_VERSION = 1 

PIPELINE_CONFIG = {
    1: {
        "process_complex_files": False,
        "use_llm_for_facts": False,
        "chunking_strategy": "file_as_chunk"
    },
    2: {
        "process_complex_files": False,
        "use_llm_for_facts": True,
        "chunking_strategy": "fact_based"
    },
    3: {
        "process_complex_files": True,
        "use_llm_for_facts": True,
        "chunking_strategy": "fact_based"
    },
    4: {
        "process_complex_files": True,
        "use_llm_for_facts": True,
        "chunking_strategy": "fact_based"
    }
}


def get_config():
    """
    Retrieves the pipeline configuration for the current version.
    """

    return PIPELINE_CONFIG.get(CURRENT_VERSION, PIPELINE_CONFIG[1])


def get_llm_client():
    """
    Initializes and returns an OpenAI client for OpenRouter API.
    """

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not found in environment variables.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    return client