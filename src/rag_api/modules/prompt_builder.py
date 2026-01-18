import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag_api.models import Message

logger = logging.getLogger(__name__)

ERROR_PROMPT = (
    "Przepraszamy, wystąpił wewnętrzny błąd podczas tworzenia zapytania. "
    "Prosimy spróbować ponownie później."
)

STATIC_FAQ = (
    "Wiedza ogólna i najczęstsze pytania (użyj tych informacji, jeśli brak ich w Kontekście):\n"
    "- Władze Wydziału: Dziekan: prof. dr hab. Grzegorz Świątek "
    "Prodziekan ds. Studenckich: dr hab. inż. Agata Pilitowska, prof. uczelni "
    "Prodziekan ds. Nauczania: dr inż. Krzysztof Kaczmarski "
    "Prodziekan ds. Nauki: prof. dr hab. Janina Kotus "
    "Prodziekan ds. Ogólnych: dr hab. Wojciech Matysiak, prof. uczelni "
    "Pełna lista: [dziekani] https://ww2.mini.pw.edu.pl/wydzial/dziekani/.\n"
    "- Kierunki studiów I stopnia (inżynierskie/licencjackie): "
    "1. Informatyka i Systemy Informacyjne (ISI), "
    "2. Inżynieria i Analiza Danych (IAD), "
    "3. Matematyka, "
    "4. Matematyka i Analiza Danych (MAD), "
    "5. Computer Science (studia w j. angielskim).\n"
    "- Kierunki studiów II stopnia (magisterskie): "
    "1. Informatyka i Systemy Informacyjne (ISI), "
    "2. Matematyka, "
    "3. Matematyka i Analiza Danych, "
    "4. Data Science (studia w j. angielskim).\n"
    "- Godziny otwarcia dziekanatu: PONIEDZIAŁEK, WTOREK, CZWARTEK, PIĄTEK 11:00-14:00, ŚRODA NIECZYNNE\n"
    "- Harmonogram roku akademickiego i sesji: Sprawdź aktualny kalendarz akademicki na stronie uczelni. https://www.pw.edu.pl/studia/harmonogram-roku-akademickiego \n"
    "- Punkty ECTS: Szczegóły w regulaminie. https://ww2.mini.pw.edu.pl/wp-content/uploads/Warunki-rejestracji-na-kolejny-semestr-rok-studiow-22.11.2023.pdf \n"
    "- Oferta przedmiotów obieralnych: Zależy od kierunku, dostępne w systemie USOS. https://ww2.mini.pw.edu.pl/wp-content/uploads/katalog-obieralne-2023.pdf \n"
    "- Wydarzenia wydziałowe: Śledź stronę wydziału i samorządu. https://ww2.mini.pw.edu.pl/ https://www.facebook.com/wrsminipw?locale=pl_PL \n"
)


def build_messages(
    query: str,
    context: list[str],
    field_of_study: str | None = None,
    semester: str | None = None,
    conversation_history: list["Message"] | None = None,
) -> list[dict[str, str]]:
    """
    Builds a messages array for the LLM based on the provided user query and context.

    Constructs a properly formatted messages array with:
    - System message: Instructions, static FAQ, and retrieved context
    - Conversation history: Previous user/assistant exchanges (including rag retrieved context for each user question)
    - Current query: As the latest user message

    Parameters
    ----------
    query : str
        The user's question or input.
    context : list[str]
        A list of text chunks retrieved from the vector database.
    field_of_study : str | None, optional
        The student's field of study (e.g., "Informatyka"), by default None.
    semester : str | None, optional
        The student's current semester, by default None.
    conversation_history : list[Message] | None, optional
        Previous conversation exchanges as Message objects, by default None.

    Returns
    -------
    list[dict[str, str]]
        A list of message dicts with 'role' and 'content' keys, ready for the LLM API.
    """
    logger.info(
        "Building messages for query: '%s', Field: %s, Sem: %s, History: %d messages",
        query,
        field_of_study,
        semester,
        len(conversation_history) if conversation_history else 0,
    )

    try:
        labeled = [f"[S{i}]\n{c}" for i, c in enumerate(context, start=1)]
        joined_context = "\n\n---\n\n".join(labeled)

        logger.debug("Joined %d context chunks into system message.", len(labeled))

        student_info = ""
        if field_of_study and semester:
            student_info = (
                f"Informacja o użytkowniku: Użytkownik studiuje na kierunku '{field_of_study}', "
                f"semestr {semester}. Wykorzystaj tę wiedzę przy pytaniach o plan zajęć, "
                "przedmioty, sale wykładowe lub egzaminy.\n\n"
            )
        elif field_of_study:
            student_info = f"Informacja o użytkowniku: Użytkownik studiuje na kierunku '{field_of_study}'.\n\n"

        # Build system message with instructions and FAQ only (static)
        system_message = (
            "Jesteś pomocnym asystentem o imieniu MiNIonek. Odpowiadasz na pytania studentów i pracowników Wydziału Matematyki i Nauk Informacyjnych (MiNI).\n"
            "Stworzyli Cię członkowie Koła Naukowego Data Science (KNDS), działającego przy Wydziale MiNI PW. Projekt merytorycznie nadzorowała dr inż. Anna Wróblewska.\n\n"
            "ZASADY ODPOWIADANIA:\n"
            "1. Priorytetyzacja wiedzy: Opieraj swoją odpowiedź głównie na informacjach z sekcji 'Kontekst', która zawiera informacje dostarczone przez system na podstawie wyszukiwania w bazie wiedzy. Wybierz z niej maksymalnie 5 najbardziej trafnych fragmentów [Sx] i na nich zbuduj odpowiedź. "
            "Jeśli nie znajdziesz tam odpowiedzi, sprawdź sekcję 'Wiedza ogólna'. "
            "Możesz korzystać z własnej wiedzy tylko wtedy, gdy informacji brakuje w obu powyższych źródłach.\n"
            "2. Kontekst rozmowy: Uwzględnij historię rozmowy - użytkownik może nawiązywać do wcześniejszych pytań lub odpowiedzi.\n"
            "3. Styl: Odpowiadaj krótko, rzeczowo i po polsku.\n"
            "4. WAŻNE: Odpowiadaj ZAWSZE w języku POLSKIM. Twoja odpowiedź zostanie automatycznie przetłumaczona na język wybrany przez użytkownika. Nie mieszaj języków i nie dodawaj komentarzy o tłumaczeniu.\n\n"
            f"{student_info}"
            f"---\n{STATIC_FAQ}\n---"
        )

        # Build context message for current query
        context_message = f"---\nKontekst (informacje, które mogą - ale nie muszą - okazać się przydatne przy odpowiadaniu na bieżące pytanie):\n{joined_context}\n---"

        # Build messages array: system + history + combined context + query
        messages = (
            [{"role": "system", "content": system_message}]
            + [msg.model_dump() for msg in (conversation_history or [])]
            + [
                {
                    "role": "user",
                    "content": f"{context_message}\n\nBieżące pytanie użytkownika:\n{query}",
                }
            ]
        )

        logger.info("Messages built successfully. Total messages: %d", len(messages))
        return messages

    except Exception as e:
        logger.error("Failed to build messages: %s", e, exc_info=True)
        raise
