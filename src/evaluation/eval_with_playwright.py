import csv
import os

import pandas as pd
from playwright.sync_api import sync_playwright

# path to skrypt.py
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(SCRIPT_PATH, "pytania.tsv")
OUTPUT_FILE = os.path.join(SCRIPT_PATH, "odp.tsv")


def uruchom_makro():
    print(f"Wczytywanie pytań z {INPUT_FILE}...")

    try:
        df = pd.read_csv(INPUT_FILE, sep="\t", encoding="utf-8")
    except Exception as e:
        print(f"Błąd odczytu fileu: {e}")
        return

    # second column from pytania.csv with PYTANIA
    question_column = df.iloc[:, 1].dropna().tolist()

    print(f"Przygotowywanie fileu wynikowego: {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            [
                "Zadane question",
                "Odpowiedz Systemu",
                "Zrodlo 1",
                "Zrodlo 2",
                "Zrodlo 3",
                "Zrodlo 4",
                "Zrodlo 5",
            ]
        )

    print("Uruchamianie przeglądarki (makro)...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        print("Ładowanie strony chatbota...")
        page.goto("https://chatbotknds.mini.pw.edu.pl/")

        page.wait_for_timeout(3000)

        for question in question_column:
            print(f"Zadawanie pytania: {question}")

            # Before asking questions check the number of messages so as not to save twice
            num_of_answers_before = page.locator(".message.bot").count()

            text_field = page.get_by_placeholder("Zadaj question...")
            text_field.clear()
            text_field.fill(str(question))
            text_field.press("Enter")

            # wait for new answer (max 15 seconds)
            for _ in range(15):
                if page.locator(".message.bot").count() > num_of_answers_before:
                    break
                page.wait_for_timeout(1000)

            last_bubble = page.locator(".message.bot .message-bubble").last
            for _ in range(30):
                try:
                    if "ŹRÓDŁA:" in last_bubble.inner_text():
                        break
                except Exception:
                    pass
                page.wait_for_timeout(1000)

            page.wait_for_timeout(1000)

            raw_output = "BŁĄD POBIERANIA"
            sources = ["", "", "", "", ""]

            try:
                full_text = last_bubble.inner_text()
                if "ŹRÓDŁA:" in full_text:
                    parts = full_text.split("ŹRÓDŁA:")
                    raw_output = parts[0].strip()
                    sources_text = parts[1].strip()

                    sources_found = [
                        z.strip() for z in sources_text.split("\n") if z.strip()
                    ]

                    for i in range(min(5, len(sources_found))):
                        sources[i] = sources_found[i]

                else:
                    raw_output = full_text.strip()

                print("Skopiowano odpowiedź i przefiltrowano źródła.")
            except Exception:
                print("Nie udało się skopiować odpowiedzi.")

            with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter="\t")

                writer.writerow([question, raw_output] + sources)

            page.wait_for_timeout(1000)

        browser.close()
        print("Gotowe! Wszystkie pytania zostały przetworzone.")


if __name__ == "__main__":
    uruchom_makro()
