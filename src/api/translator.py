import logging

from src.ingestion.common import MODEL_WORKER, get_llm_client

logger = logging.getLogger(__name__)


def translate_text(text: str, target_lang_code: str) -> str:
    """
    Translates the input text into the target language specified by target_lang_code.
    Supported target_lang_code values: "pl" (Polish), "en" (English), "ua" (Ukrainian).
    """
    if not text:
        return ""

    lang_map = {"pl": "Polish", "en": "English", "ua": "Ukrainian"}

    target_lang_name = lang_map.get(target_lang_code, "Polish")

    client = get_llm_client()

    system_prompt = (
        f"You are a professional academic translator. Translate the following text into {target_lang_name}. "
        "The text is from a university chatbot (MiNIonek) serving students and staff of the Faculty of Mathematics and Information Science (MiNI) at Warsaw University of Technology (Politechnika Warszawska). "
        "Use accurate academic and administrative terminology. "
        "Key term translations — always use these: "
        "Wydział → Faculty, kierunek → field of study / degree programme, semestr → semester, "
        "dziekanat → dean's office, indeks → student record book, USOS → USOS (student information system), "
        "praca dyplomowa → thesis / dissertation, egzamin → exam, zaliczenie → credit / pass, "
        "Politechnika Warszawska → Warsaw University of Technology, "
        "Koło Naukowe → student scientific club. "
        "Preserve the original meaning, tone, and formatting. "
        "Return ONLY the translated text, without any additional comments or explanations."
    )

    try:
        logger.debug(f"Translating text to {target_lang_name}...")
        response = client.chat.completions.create(
            model=MODEL_WORKER,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
        )
        translated_text = response.choices[0].message.content.strip()
        return translated_text

    except Exception as e:
        logger.error(f"Translation to {target_lang_name} failed: {e}")
        return text
