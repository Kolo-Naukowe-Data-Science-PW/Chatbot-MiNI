import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000/chat")

st.set_page_config(page_title="Chatbot MiNI", page_icon="đźŽ“")
st.title("Chatbot Wydziału MiNI PW“")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("O co chcesz zapytać?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Szukam informacji..."):
            try:
                response = requests.post(API_URL, json={"query": prompt})
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Błąd braku odpowiedzi.")
                    sources = list(set(data.get("sources", [])))

                    full_response = answer
                    if sources:
                        full_response += "\n\n**Źródła:**\n" + "\n".join(
                            [f"- {s}" for s in sources]
                        )

                    st.markdown(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                else:
                    st.error(f"Błąd API: {response.status_code}")
            except Exception as e:
                st.error(f"Nie udało się połączyć z chatbotem. Błąd: {e}")
