import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not any(
    isinstance(h, logging.FileHandler)
    and h.baseFilename.endswith("chatbot_metrics.log")
    for h in logger.handlers
):
    file_handler = logging.FileHandler("chatbot_metrics.log")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def log_chatbot_metrics(func: Callable[..., Any]) -> Callable[..., Any]:

    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        response = func(*args, **kwargs)
        end_time = time.perf_counter()

        delta_time = end_time - start_time

        metrics: dict[str, Any] = {
            "metric": "chatbot_response_time",
            "delta_time_seconds": round(delta_time, 4),
            "function": func.__name__,
        }

        user_input = kwargs.get("user_input") or (args[0] if args else "")
        if isinstance(user_input, str):
            metrics["input_length"] = len(user_input)
        if isinstance(response, str):
            metrics["response_length"] = len(response)

        log_message = (
            f"Chatbot response metrics | func={func.__name__} | "
            f"time={metrics['delta_time_seconds']:.3f}s | "
            f"input_len={metrics.get('input_length', -1)} | "
            f"response_len={metrics.get('response_length', -1)}"
        )

        logger.info(log_message)

        with open("chatbot_metrics.log", "a", encoding="utf-8") as f:
            f.write(f"{metrics}\n")

        return response

    return wrapper
