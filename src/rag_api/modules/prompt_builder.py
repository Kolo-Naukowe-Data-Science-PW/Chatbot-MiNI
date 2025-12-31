import logging

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


def build_prompt(
    query: str,
    context: list[str],
    field_of_study: str | None = None,
    semester: str | None = None,
) -> str:
    """
    Builds a prompt for the LLM based on the provided user query and context.

    Constructs a formatted string including system instructions, static FAQ knowledge,
    retrieved context chunks, and optional student metadata (field of study, semester).

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

    Returns
    -------
    str
        The fully formatted prompt string ready to be sent to the LLM.
    """
    logger.info(
        "Building prompt for query: '%s', Field: %s, Sem: %s",
        query,
        field_of_study,
        semester,
    )

    try:
        labeled = [f"[S{i}]\n{c}" for i, c in enumerate(context, start=1)]
        joined_context = "\n\n---\n\n".join(labeled)

        logger.debug("Joined %d context chunks into prompt.", len(labeled))

        student_info = ""
        if field_of_study and semester:
            student_info = (
                f"Informacja o użytkowniku: Użytkownik studiuje na kierunku '{field_of_study}', "
                f"semestr {semester}. Wykorzystaj tę wiedzę przy pytaniach o plan zajęć, "
                "przedmioty, sale wykładowe lub egzaminy."
            )
        elif field_of_study:
            student_info = f"Informacja o użytkowniku: Użytkownik studiuje na kierunku '{field_of_study}'."

        prompt = (
            "Jesteś pomocnym asystentem o imieniu MiNIonek. Odpowiadasz na pytania studentów i pracowników Wydziału Matematyki i Nauk Informacyjnych (MiNI).\n"
            "ZASADY ODPOWIADANIA:\n"
            "1. Priorytetyzacja wiedzy: Opieraj swoją odpowiedź głównie na informacjach z sekcji 'Kontekst'. Wybierz z niej maksymalnie 5 najbardziej trafnych fragmentów [Sx] i na nich zbuduj odpowiedź."
            "Jeśli nie znajdziesz tam odpowiedzi, sprawdź sekcję 'Wiedza ogólna'. "
            "Możesz korzystać z własnej wiedzy tylko wtedy, gdy informacji brakuje w obu powyższych źródłach.\n"
            "2. Styl: Odpowiadaj krótko, rzeczowo i po polsku.\n"
            "3. Cytowanie: Każde stwierdzenie oparte na sekcji 'Kontekst' popieraj odwołaniem [Sx] w treści wypowiedzi.\n"
            "4. Lista źródeł: W sekcji 'Źródła' wymień tylko te identyfikatory, które faktycznie zostały użyte do udzielenia odpowiedzi (maksymalnie 5 najważniejszych)."
            "które zostały faktycznie użyte w odpowiedzi. Nie wymieniaj wszystkich dostępnych chunków.\n\n"
            f"{student_info}\n\n"
            f"---\n{STATIC_FAQ}\n---\n\n"
            f"---\nKontekst:\n{joined_context}\n---\n\n"
            f"Pytanie: {query}\n\n"
            "Odpowiedź:"
        )

        logger.info("Prompt built successfully.")
        return prompt

    except Exception as e:
        logger.error("Failed to build prompt: %s", e, exc_info=True)
        return ERROR_PROMPT