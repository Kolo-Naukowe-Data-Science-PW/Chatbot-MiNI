import logging

logger = logging.getLogger(__name__)

ERROR_PROMPT = (
    "Przepraszamy, wystąpił wewnętrzny błąd podczas tworzenia zapytania. "
    "Prosimy spróbować ponownie później."
)


def build_prompt(query: str, context: list) -> str:
    """
    Builds a prompt for the LLM based on the provided user query and top-k text chunks.
    """
    logger.info("Building prompt for query: '%s'", query)

    try:
        labeled = [f"[S{i}]\n{c}" for i, c in enumerate(context, start=1)]
        joined_context = "\n\n---\n\n".join(labeled)

        logger.debug("Joined %d context chunks into prompt.", len(labeled))

        prompt = (
            "Jesteś pomocnym asystentem, który odpowiada na pytania użytkownika "
            "na podstawie dostarczonego kontekstu.\n"
            "Nazywasz sie MiNIonek."
            "Odpowiadaj krótko i rzeczowo po polsku. "
            "Używaj TYLKO informacji z kontekstu. "
            "Jeśli odpowiedzi brak w kontekście, odpowiedz dokładnie: "
            "'Nie wiem na podstawie dostarczonego kontekstu.'\n"
            "Każde stwierdzenie popieraj odwołaniem [Sx]."
            "Na końcu dodaj sekcję 'Źródła:' z listą wszystkich użytych odwołań.\n\n"
            f"---\nKontekst:\n{joined_context}\n---\n\n"
            f"Pytanie: {query}\n\n"
            "Odpowiedź:"
        )

        logger.info("Prompt built successfully.")
        return prompt

    except Exception as e:
        logger.error("Failed to build prompt: %s", e, exc_info=True)
        return ERROR_PROMPT
