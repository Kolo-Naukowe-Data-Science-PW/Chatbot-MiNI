import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

REWRITE_MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = (
    "Jesteś optymalizatorem zapytań do wyszukiwarki. "
    "Twoim zadaniem jest przekształcenie pytania użytkownika w zwięzłe zapytanie "
    "do semantycznej bazy wiedzy Wydziału MiNI PW. "
    "Wyodrębnij kluczowe pojęcia, nazwy własne i terminy akademickie. "
    "Usuń zbędne słowa, zaimki i formy grzecznościowe. "
    "Odpowiadaj WYŁĄCZNIE przepisanym zapytaniem — bez wyjaśnień, bez cudzysłowów."
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


def rewrite_query(query: str) -> str:
    """
    Rewrite a user query into a concise, keyword-rich form optimised for vector retrieval.

    The rewritten query is used only for Qdrant search — the original query is still
    passed to the LLM so that the generated answer addresses the user's actual question.

    Falls back to the original query if the rewriting call fails.

    Parameters
    ----------
    query : str
        User question in Polish (translation to PL should happen before calling this).

    Returns
    -------
    str
        Rewritten query for retrieval, or the original query on error.

    Examples
    --------
    "kiedy mam maturę?" -> "egzamin sesja termin harmonogram WMiNI"
    "jacy profesorowie wykładają probabilistykę?" -> "probabilistyka prowadzący wykładowca WMiNI"
    "co muszę zrobić żeby pojechać na erasmusa?" -> "Erasmus wymiana zagraniczna dokumenty procedura"
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=60,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        if not rewritten:
            logger.warning("Query rewriter returned empty string, using original.")
            return query
        logger.info("Query rewritten: '%s' -> '%s'", query, rewritten)
        return rewritten
    except Exception as exc:
        logger.warning("Query rewriting failed (%s), falling back to original.", exc)
        return query
