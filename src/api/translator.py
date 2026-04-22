import logging

from src.pipeline.common import MODEL_WORKER, get_llm_client

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
        f"You are a professional translator. Translate the following text into {target_lang_name}."
        "Preserve the original meaning, tone, formatting, and specific terminology (e.g. university names)."
        "Return ONLY the translated text, without any additional comments, introductory phrases, and explanations."
        "Use appropriate vocabulary and grammatical structures specific to the given language."
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
