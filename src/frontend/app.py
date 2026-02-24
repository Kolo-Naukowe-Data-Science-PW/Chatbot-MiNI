import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")
st.set_page_config(page_title="Chatbot Wydziału MiNI PW", page_icon="🎓")

with st.sidebar:
    st.title("Język / Language / Мова")
    selected_lang = st.selectbox(
        label="Wybierz język / Select language / Виберіть мову",
        options=["pl", "en", "ua"],
        format_func=lambda x: {
            "pl": "Polski 🇵🇱",
            "en": "English 🇬🇧",
            "ua": "Українська 🇺🇦",
        }[x],
        label_visibility="collapsed",
    )

st.title("Chatbot Wydziału MiNI PW 🎓")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


UI_TEXTS = {
    "placeholders": {
        "pl": "O co chcesz zapytać?",
        "en": "What do you want to ask?",
        "ua": "Що ви хочете запитати?",
    },
    "thinking": {
        "pl": "Szukam informacji...",
        "en": "Searching for information...",
        "ua": "Шукаю інформацію...",
    },
    "no_answer": {
        "pl": "Błąd braku odpowiedzi.",
        "en": "No answer returned.",
        "ua": "Відповідь відсутня.",
    },
    "sources": {"pl": "Źródła", "en": "Sources", "ua": "Джерела"},
    "api_error": {"pl": "Błąd API", "en": "API error", "ua": "Помилка API"},
    "connection_error": {
        "pl": "Nie udało się połączyć z chatbotem. Błąd:",
        "en": "Failed to connect to the chatbot. Error:",
        "ua": "Не вдалося підключитися до чатбота. Помилка:",
    },
}


def t(key: str, lang: str) -> str:
    """
    Returns UI text for given key and language.
    Falls back to Polish if language or key is missing.
    """
    return UI_TEXTS.get(key, {}).get(lang, UI_TEXTS.get(key, {}).get("pl", ""))


if prompt := st.chat_input(t("placeholders", selected_lang)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(t("thinking", selected_lang)):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": prompt,
                        "language": selected_lang,
                        "conversation_history": st.session_state.conversation_history,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", t("no_answer", selected_lang))
                    sources = list(dict.fromkeys(data.get("sources", [])[:5]))

                    st.session_state.conversation_history = data.get(
                        "conversation_history", []
                    )

                    full_response = answer
                    if sources:
                        full_response += (
                            "\n\n**"
                            + t("sources", selected_lang)
                            + ":**\n"
                            + "\n".join([f"- {s}" for s in sources])
                        )

                    st.markdown(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                else:
                    st.error(f"{t('api_error', selected_lang)}: {response.status_code}")

            except Exception as e:
                st.error(f"{t('connection_error', selected_lang)} {e}")
