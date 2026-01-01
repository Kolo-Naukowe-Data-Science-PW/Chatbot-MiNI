import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_VERSION = int(os.getenv("PIPELINE_VERSION", 1))
MODEL_WORKER = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")


PIPELINE_CONFIG = {
    1: {
        "process_complex_files": False,
        "use_llm_for_facts": False,
        "chunking_strategy": "file_as_chunk",
    },
    2: {
        "process_complex_files": False,
        "use_llm_for_facts": True,
        "chunking_strategy": "fact_based",
    },
    3: {
        "process_complex_files": True,
        "use_llm_for_facts": True,
        "chunking_strategy": "fact_based",
    },
    4: {
        "process_complex_files": True,
        "use_llm_for_facts": True,
        "chunking_strategy": "fact_based",
    },
}


def get_config() -> dict[str, Any]:
    """
    Retrieves the pipeline configuration for the current version.

    Parameters
    ----------
    None

    Returns
    -------
    dict[str, Any]
        A dictionary containing configuration settings: 'process_complex_files',
        'use_llm_for_facts' and 'chunking_strategy'.
    """
    return PIPELINE_CONFIG.get(CURRENT_VERSION, PIPELINE_CONFIG[1])


def get_llm_client() -> OpenAI:
    """
    Initializes and returns an OpenAI client for OpenRouter API.

    Parameters
    ----------
    None

    Returns
    -------
    openai.OpenAI
        The initialized OpenAI client object configured for OpenRouter.
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY not found in environment variables.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    return client
