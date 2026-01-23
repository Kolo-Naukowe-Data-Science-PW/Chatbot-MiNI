import concurrent.futures
import csv
import os
from datetime import datetime

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")
LOG_FILE = "arena_votes.csv"

st.set_page_config(page_title="Chatbot Arena ⚔️", page_icon="⚔️")


def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "query", "answer_a", "answer_b", "vote", "language"]
            )


def log_vote(query, ans_a, ans_b, vote, lang):
    init_log_file()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), query, ans_a, ans_b, vote, lang])


def load_stats():
    if not os.path.exists(LOG_FILE):
        return {"A": 0, "B": 0, "Tie": 0, "Total": 0}

    stats = {"A": 0, "B": 0, "Tie": 0, "Total": 0}
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["Total"] += 1
                v = row.get("vote")
                if v in stats:
                    stats[v] += 1
    except Exception:
        pass
    return stats


UI_TEXTS = {
    "placeholders": {
        "pl": "O co chcesz zapytać?",
        "en": "What do you want to ask?",
        "ua": "Що ви хочете запитати?",
    },
    "thinking": {
        "pl": "Generuję odpowiedzi...",
        "en": "Generating responses...",
        "ua": "Генерую відповіді...",
    },
    "vote_prompt": {
        "pl": "Która odpowiedź jest lepsza?",
        "en": "Which answer is better?",
        "ua": "Яка відповідь краща?",
    },
    "vote_a": {"pl": "👈 Odpowiedź A", "en": "👈 Answer A", "ua": "👈 Відповідь A"},
    "vote_b": {"pl": "Odpowiedź B 👉", "en": "Answer B 👉", "ua": "Відповідь B 👉"},
    "vote_tie": {"pl": "🤝 Remis", "en": "🤝 Tie", "ua": "🤝 Нічия"},
    "score_title": {
        "pl": "Wyniki Areny",
        "en": "Arena Scores",
        "ua": "Результати Арени",
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
    # st.subheader(t("score_title", selected_lang))
    # st.write(f"🅰️ A: {stats['A']}")
    # st.write(f"🅱️ B: {stats['B']}")
    # st.write(f"🤝 Tie: {stats['Tie']}")
    # st.write(f"Total: {stats['Total']}")

st.title("Chatbot Arena ⚔️")
st.markdown("Compare two LLM responses and vote for the best one!")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "pending_vote" not in st.session_state:
    st.session_state.pending_vote = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def call_api(query, lang, history):
    try:
        response = requests.post(
            API_URL,
            json={
                "query": query,
                "language": lang,
                "conversation_history": history,
            },
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Error {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


if st.session_state.pending_vote:
    vote_data = st.session_state.pending_vote
    st.divider()
    st.subheader(t("vote_prompt", selected_lang))

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Answer A**")
        st.markdown(vote_data["ans_a_text"])
        if st.button(t("vote_a", selected_lang), use_container_width=True):
            log_vote(
                vote_data["query"],
                vote_data["ans_a_text"],
                vote_data["ans_b_text"],
                "A",
                selected_lang,
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": vote_data["ans_a_text"]}
            )
            st.session_state.conversation_history = vote_data["res_a"].get(
                "conversation_history", []
            )
            st.session_state.pending_vote = None
            st.rerun()

    with col2:
        st.info("**Answer B**")
        st.markdown(vote_data["ans_b_text"])
        if st.button(t("vote_b", selected_lang), use_container_width=True):
            log_vote(
                vote_data["query"],
                vote_data["ans_a_text"],
                vote_data["ans_b_text"],
                "B",
                selected_lang,
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": vote_data["ans_b_text"]}
            )
            st.session_state.conversation_history = vote_data["res_b"].get(
                "conversation_history", []
            )
            st.session_state.pending_vote = None
            st.rerun()

    if st.button(t("vote_tie", selected_lang), use_container_width=True):
        log_vote(
            vote_data["query"],
            vote_data["ans_a_text"],
            vote_data["ans_b_text"],
            "Tie",
            selected_lang,
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": vote_data["ans_a_text"]}
        )
        st.session_state.conversation_history = vote_data["res_a"].get(
            "conversation_history", []
        )
        st.session_state.pending_vote = None
        st.rerun()

else:
    if prompt := st.chat_input(t("placeholders", selected_lang)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner(t("thinking", selected_lang)):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_a = executor.submit(
                    call_api,
                    prompt,
                    selected_lang,
                    st.session_state.conversation_history,
                )
                future_b = executor.submit(
                    call_api,
                    prompt,
                    selected_lang,
                    st.session_state.conversation_history,
                )

                res_a = future_a.result()
                res_b = future_b.result()

            ans_a = (
                res_a.get("answer", "Error in A")
                if "error" not in res_a
                else res_a["error"]
            )
            ans_b = (
                res_b.get("answer", "Error in B")
                if "error" not in res_b
                else res_b["error"]
            )

            st.session_state.pending_vote = {
                "query": prompt,
                "res_a": res_a,
                "res_b": res_b,
                "ans_a_text": ans_a,
                "ans_b_text": ans_b,
            }
            st.rerun()
