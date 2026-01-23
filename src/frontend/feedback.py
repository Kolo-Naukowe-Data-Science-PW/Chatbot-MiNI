import csv
import logging
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")
LOG_FILE = "feedback_logs.csv"

st.set_page_config(page_title="Chatbot Feedback 📝", page_icon="📝")


def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "query", "response", "rating", "explanation", "language"]
            )


def log_feedback(query, response, rating, explanation, lang):
    init_log_file()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [datetime.now().isoformat(), query, response, rating, explanation, lang]
        )


def load_stats():
    if not os.path.exists(LOG_FILE):
        return {"count": 0, "average": 0.0}

    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            return {"count": 0, "average": 0.0}

        return {"count": len(df), "average": df["rating"].mean()}
    except Exception as e:
        logger.error(f"Error loading stats: {e}")
        return {"count": 0, "average": 0.0}


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
    "feedback_title": {
        "pl": "Oceń odpowiedź (1-5)",
        "en": "Rate this answer (1-5)",
        "ua": "Оцініть відповідь (1-5)",
    },
    "explanation_label": {
        "pl": "Wyjaśnienie (opcjonalne)",
        "en": "Explanation (optional)",
        "ua": "Пояснення (необов'язково)",
    },
    "submit_btn": {
        "pl": "Wyślij opinię",
        "en": "Submit Feedback",
        "ua": "Надіслати відгук",
    },
    "feedback_thanks": {
        "pl": "Dziękujemy za opinię!",
        "en": "Thank you for your feedback!",
        "ua": "Дякуємо за відгук!",
    },
    "stats_title": {
        "pl": "Statystyki Opinii",
        "en": "Feedback Stats",
        "ua": "Статистика відгуків",
    },
}


def t(key: str, lang: str) -> str:
    return UI_TEXTS.get(key, {}).get(lang, UI_TEXTS.get(key, {}).get("pl", ""))


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

    # st.divider()
    # stats = load_stats()
    # st.subheader(t("stats_title", selected_lang))
    # st.metric("Total Feedbacks", stats["count"])
    # st.metric("Average Rating", f"{stats['average']:.2f} ⭐")

st.title("Chatbot Feedback 📝")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "awaiting_feedback" not in st.session_state:
    st.session_state.awaiting_feedback = None

for _, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "assistant"
    and st.session_state.awaiting_feedback == len(st.session_state.messages) - 1
):

    st.divider()
    with st.container(border=True):
        st.subheader(t("feedback_title", selected_lang))

        rating = st.slider("Rating", 1, 5, 3, label_visibility="collapsed")
        explanation = st.text_input(t("explanation_label", selected_lang))

        if st.button(t("submit_btn", selected_lang)):
            last_query = (
                st.session_state.messages[-2]["content"]
                if len(st.session_state.messages) >= 2
                else ""
            )
            last_response = st.session_state.messages[-1]["content"]

            log_feedback(last_query, last_response, rating, explanation, selected_lang)

            st.session_state.awaiting_feedback = None
            st.success(t("feedback_thanks", selected_lang))
            # Optional: rerun to update stats immediately
            # st.rerun()


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

                    st.session_state.awaiting_feedback = (
                        len(st.session_state.messages) - 1
                    )

                    st.rerun()

                else:
                    st.error(f"{t('api_error', selected_lang)}: {response.status_code}")

            except Exception as e:
                st.error(f"{t('connection_error', selected_lang)} {e}")
