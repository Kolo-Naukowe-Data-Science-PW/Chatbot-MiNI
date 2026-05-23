import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.models import Message

logger = logging.getLogger(__name__)

ERROR_PROMPT = (
    "Przepraszamy, wystąpił wewnętrzny błąd podczas tworzenia zapytania. "
    "Prosimy spróbować ponownie później."
)

STATIC_FAQ = (
    "Wiedza ogólna i najczęstsze pytania (użyj tych informacji, jeśli brak ich w Kontekście):\n"
    "- Władze Wydziału (AKTUALNY skład — traktuj jako wiążący, ignoruj inne źródła ze swojej wiedzy): "
    "Dziekan: prof. dr hab. Grzegorz Świątek. "
    "Prodziekan ds. Studenckich: dr hab. inż. Agata Pilitowska, prof. uczelni. "
    "Prodziekan ds. Nauczania: dr inż. Krzysztof Kaczmarski. "
    "Prodziekan ds. Nauki: prof. dr hab. Janina Kotus. "
    "Prodziekan ds. Ogólnych: dr hab. Wojciech Matysiak, prof. uczelni. "
    "Pełna lista: [dziekani] https://ww2.mini.pw.edu.pl/wydzial/dziekani/. "
    "WAŻNE: Jerzy Błaszczyk NIE jest i NIE był dziekanem ani prodziekanem MiNI — jest pracownikiem wydziału. Nie przypisuj mu żadnej funkcji dziekańskiej.\n"
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


USER_TYPE_PERSONA: dict[str, str] = {
    "student_junior": (
        "Użytkownik to student pierwszego roku. Tłumacz procedury krok po kroku, "
        "używaj prostego języka, zachęcaj do pytania o szczegóły."
    ),
    "student_senior": (
        "Użytkownik to student starszego roku. Zakładaj znajomość podstaw, "
        "możesz używać terminologii uczelnianej."
    ),
    "master": (
        "Użytkownik to student studiów magisterskich. Zakładaj dobrą znajomość "
        "systemu uczelnianego, skup się na zagadnieniach magisterskich."
    ),
    "phd": (
        "Użytkownik to doktorant. Traktuj go jak partnera, możesz dyskutować "
        "o procedurach administracyjnych i naukowych na poziomie zaawansowanym."
    ),
    "candidate": (
        "Użytkownik jest kandydatem na studia w MiNI PW. Wyjaśniaj zasady rekrutacji, "
        "wymagania wstępne i charakterystykę kierunków. Używaj przystępnego, zachęcającego języka."
    ),
    "admin": (
        "Użytkownik to pracownik administracji lub wykładowca. Odpowiadaj formalnie "
        "i precyzyjnie, skup się na aspektach administracyjnych i regulaminowych."
    ),
    "research_teaching": (
        "Użytkownik to pracownik badawczo-dydaktyczny. Odpowiadaj formalnie i precyzyjnie, "
        "uwzględniaj aspekty zarówno naukowo-badawcze, jak i dydaktyczne."
    ),
}


def build_messages(
    query: str,
    context: list[str],
    field_of_study: str | None = None,
    semester: str | None = None,
    user_type: str | None = None,
    conversation_history: list["Message"] | None = None,
    style_instruction: str | None = None,
    attachments: list[dict] | None = None,
) -> list[dict]:
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
        "Building messages for query: '%s', Field: %s, Sem: %s, UserType: %s, History: %d messages",
        query,
        field_of_study,
        semester,
        user_type,
        len(conversation_history) if conversation_history else 0,
    )

    try:
        joined_context = "\n\n---\n\n".join(context)

        logger.debug("Joined %d context chunks into system message.", len(context))

        # "—" is the frontend sentinel for "not applicable" (admin/research/phd skips major)
        effective_major = field_of_study if field_of_study and field_of_study != "—" else None
        effective_sem = semester if semester and semester != "—" else None

        student_info = ""
        if effective_major and effective_sem:
            student_info = (
                f"Informacja o użytkowniku: Użytkownik studiuje na kierunku '{effective_major}', "
                f"semestr {effective_sem}. Wykorzystaj tę wiedzę przy pytaniach o plan zajęć, "
                "przedmioty, sale wykładowe lub egzaminy.\n\n"
            )
        elif effective_major:
            student_info = f"Informacja o użytkowniku: Użytkownik studiuje na kierunku '{effective_major}'.\n\n"
        elif effective_sem:
            student_info = f"Informacja o użytkowniku: {effective_sem}.\n\n"

        role_hint = ""
        if user_type and user_type in USER_TYPE_PERSONA:
            role_hint = f"Wskazówka dotycząca rozmówcy: {USER_TYPE_PERSONA[user_type]}\n\n"

        # Build system message with instructions and FAQ only (static)
        now = datetime.now()
        today_str = now.strftime("%d.%m.%Y, godz. %H:%M")
        style_hint = (
            f"6. Styl odpowiedzi (priorytet nad regułą 3): {style_instruction}\n"
            if style_instruction else ""
        )

        system_message = (
            f"Jesteś pomocnym asystentem o imieniu MiNIonek. Odpowiadasz na pytania studentów i pracowników Wydziału Matematyki i Nauk Informacyjnych (MiNI).\n"
            "Stworzyli Cię członkowie Koła Naukowego Data Science (KNDS), działającego przy Wydziale MiNI PW. Projekt merytorycznie nadzorowała dr inż. Anna Wróblewska.\n"
            f"Dzisiaj jest {today_str}.\n\n"
            "ZASADY ODPOWIADANIA:\n"
            "1. Priorytetyzacja wiedzy: Opieraj swoją odpowiedź na informacjach z sekcji 'Kontekst', która zawiera fakty dostarczone przez system na podstawie wyszukiwania w bazie wiedzy. "
            "Jeśli nie znajdziesz tam odpowiedzi, sprawdź sekcję 'Wiedza ogólna'. "
            "Informacje w sekcji 'Wiedza ogólna' (szczególnie skład władz wydziału) są AKTUALNE i NADRZĘDNE nad Twoją wiedzą z treningu — stosuj je dosłownie.\n"
            "2. Kontekst rozmowy: Uwzględnij historię rozmowy — użytkownik może nawiązywać do wcześniejszych pytań lub odpowiedzi.\n"
            "3. Styl: Odpowiadaj zwięźle i rzeczowo. Zacznij bezpośrednio od odpowiedzi — bez pozdrowień, bez wstępów w stylu 'Krótka odpowiedź:'. "
            "Nie używaj formatowania Markdown (bez gwiazdek, nagłówków, punktorów — chyba że lista jest naprawdę niezbędna). "
            "Pisz pełnymi, gramatycznie poprawnymi zdaniami. "
            "ZAWSZE stawiaj spację po kropce, przecinku i każdym innym znaku interpunkcyjnym — nigdy nie łącz dwóch wyrazów bez spacji. "
            "Wyjątek dla planu zajęć: każde zajęcie wypisuj w OSOBNEJ LINII (oddzielone enterem), w formacie: 'GG:MM–GG:MM — Nazwa przedmiotu (typ, gr. N), sala X, bud. Y, prowadzący: Imię Nazwisko.' "
            "Jeśli pytanie dotyczy zajęć w danym dniu, ZAWSZE wymieniaj ABSOLUTNIE WSZYSTKIE zajęcia w tym dniu dla danego kierunku i semestru — nie pomijaj żadnych, nawet jeśli jest ich dużo. "
            "WAŻNE: kontekst może zawierać fakty z planów RÓŻNYCH kierunków. Przy pytaniach o plan zajęć użytkownika uwzględniaj WYŁĄCZNIE fakty, które jawnie dotyczą jego kierunku (np. 'Inżynieria i Analiza Danych' dla IAD). Ignoruj fakty z innych kierunków (np. ISI, MAD, Matematyka), nawet jeśli dotyczą tego samego semestru i dnia tygodnia.\n"
            "4. Liczby i dane: Jeśli w Kontekście lub Wiedzy ogólnej znajdują się konkretne liczby (godziny, semestry, punkty ECTS, progi zaliczeniowe, daty, numery sal itp.) — zawsze podaj je dokładnie. "
            "Nigdy nie stosuj placeholderów (np. '___', '[X]', '...') w miejscu brakujących danych. "
            "Jeśli nie masz konkretnej liczby, napisz wprost: 'Nie mam tej informacji w dostępnych zasobach.' "
            "Nie odsyłaj do regulaminu, jeśli odpowiedź jest dostępna w Kontekście.\n"
            "5. WAŻNE: Odpowiadaj ZAWSZE w języku POLSKIM. Twoja odpowiedź zostanie automatycznie przetłumaczona na język wybrany przez użytkownika. Nie mieszaj języków i nie dodawaj komentarzy o tłumaczeniu.\n"
            f"{style_hint}"
            f"\n{role_hint}"
            f"{student_info}"
            f"---\n{STATIC_FAQ}\n---"
        )

        # Build context message for current query
        context_message = f"---\nKontekst (informacje, które mogą - ale nie muszą - okazać się przydatne przy odpowiadaniu na bieżące pytanie):\n{joined_context}\n---"

        user_text = f"{context_message}\n\nBieżące pytanie użytkownika:\n{query}"

        if attachments:
            user_content: list[dict] = [{"type": "text", "text": user_text}]
            for att in attachments:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{att['mime_type']};base64,{att['data']}"
                    },
                })
            last_user_msg: dict = {"role": "user", "content": user_content}
        else:
            last_user_msg = {"role": "user", "content": user_text}

        # Build messages array: system + history + combined context + query
        messages = (
            [{"role": "system", "content": system_message}]
            + [msg.model_dump() for msg in (conversation_history or [])]
            + [last_user_msg]
        )

        logger.info("Messages built successfully. Total messages: %d", len(messages))
        return messages

    except Exception as e:
        logger.error("Failed to build messages: %s", e, exc_info=True)
        raise
