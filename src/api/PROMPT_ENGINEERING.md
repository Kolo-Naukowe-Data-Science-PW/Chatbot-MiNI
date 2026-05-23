# Prompt Engineering – MiNIonek

Dokumentacja systemu promptów chatbota MiNIonek. Główna logika w `prompt_builder.py`.

---

## Architektura wiadomości

Każde zapytanie do modelu składa się z:

```
[system message]   — instrukcje + STATIC_FAQ + info o użytkowniku
[historia rozmowy] — ostatnie 20 wiadomości (10 tur)
[user message]     — kontekst RAG + bieżące pytanie
```

Jeśli użytkownik załączył PDF, ostatnia wiadomość użytkownika jest multimodalna:
```json
{"role": "user", "content": [
    {"type": "text",      "text": "<kontekst RAG + pytanie>"},
    {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,..."}}
]}
```

---

## System message

Budowany dynamicznie w `build_messages()`. Składa się z:

### 1. Tożsamość i data
```
Jesteś pomocnym asystentem o imieniu MiNIonek. Odpowiadasz na pytania studentów i pracowników MiNI PW.
Stworzyli Cię członkowie KNDS, merytorycznie nadzorowała dr inż. Anna Wróblewska.
Dzisiaj jest DD.MM.YYYY, godz. HH:MM.
```

### 2. Sześć zasad odpowiadania (ZASADY ODPOWIADANIA)

| Nr | Zasada | Cel |
|----|--------|-----|
| 1 | **Priorytetyzacja wiedzy**: Kontekst > Wiedza ogólna > Wiedza z treningu | Ogranicza halucynacje; skład władz wydziału z STATIC_FAQ jest nadrzędny |
| 2 | **Kontekst rozmowy**: Uwzględnij historię | Spójność wieloturowa |
| 3 | **Styl**: Zwięźle, bez Markdowna, bez pozdrowień; dla planu zajęć — każde zajęcie w osobnej linii + **ZAWSZE wymień WSZYSTKIE zajęcia w danym dniu** | Czytelność i kompletność odpowiedzi |
| 4 | **Liczby i dane**: Podaj konkretne wartości lub napisz „Nie mam tej informacji" | Zapobiega placeholderom i zmyślonym danym |
| 5 | **Zawsze po polsku** — frontend tłumaczy na język użytkownika | Jakość tłumaczenia > jakość generacji w obcym języku |
| 6 | **Styl niestandardowy** (opcjonalny, z `styleInstruction`) — nadrzędny nad zasadą 3 | Eksperymenty A/B z personami |

### 3. Wskazówka dotycząca rozmówcy (`USER_TYPE_PERSONA`)

Dobierana na podstawie pola `user_type` w requescie:

| Klucz | Opis |
|-------|------|
| `student_junior` | Pierwszy rok — krok po kroku, prosty język |
| `student_senior` | Starszy rok — zakłada znajomość terminologii |
| `master` | Magisterski — skupia się na zagadnieniach magisterskich |
| `phd` | Doktorant — partnerski ton, poziom zaawansowany |
| `candidate` | Kandydat na studia — wyjaśnia rekrutację, zachęca |
| `admin` | Administracja / wykładowca — formalnie, aspekty regulaminowe |
| `research_teaching` | Pracownik B+D — aspekty naukowo-badawcze i dydaktyczne |

### 4. Informacja o użytkowniku

Jeśli znany jest kierunek (`major`) i/lub semestr (`semester`):
```
Informacja o użytkowniku: Użytkownik studiuje na kierunku 'IAD', semestr 4.
Wykorzystaj tę wiedzę przy pytaniach o plan zajęć, przedmioty, sale wykładowe lub egzaminy.
```

### 5. STATIC_FAQ (wiedza ogólna)

Stałe fakty, niezależne od RAG:
- Aktualny skład władz wydziału (dziekan + 4 prodziekanów) z explicite wymienioną osobą, która **nie** jest dziekanem
- Kierunki I i II stopnia (5 + 4)
- Godziny otwarcia dziekanatu
- Linki: harmonogram akademicki, regulamin ECTS, katalog obieralnych, strona wydziału

---

## Wzbogacanie zapytania retriwalowego

Przed wysłaniem do Qdrant zapytanie jest wzbogacane w `_enrich_retrieval_query()`:

| Typ pytania | Co jest dodawane | Przykład |
|-------------|------------------|---------|
| **Plan zajęć** (`_SCHEDULE_KEYWORDS`) | `kierunek <major> semestr <N>` (aktualny) | `"kiedy są zajęcia IAD semestr 4"` |
| **Plan studiów** (`_CURRICULUM_KEYWORDS`) | `kierunek <major>` | `"przedmioty IAD"` |
| **Następny semestr** (`_NEXT_SEM_KEYWORDS`) | `kierunek <major> semestr <N+1>` | `"przedmioty IAD semestr 5"` |
| Pozostałe pytania | bez zmian | — |

Wzbogacanie kierunkiem/semestrem aktywowane **tylko** gdy query zawiera odpowiednie słowa kluczowe — celowe ograniczenie, by plan zajęć nie dominował w retriwalu dla niezwiązanych pytań.

---

## Persony eksperymentalne

4 style odpowiedzi używane w trybie A/B (`testPro`):

| Indeks | Styl |
|--------|------|
| 0 | Bardzo krótko i konkretnie |
| 1 | Luzno, prosto i przyjaźnie |
| 2 | Formalnie i akademicko |
| 3 | Wyczerpująco z detalami i przykładami *(domyślne)* |

Wybierany przez `EXPERIMENT_PERSONA` (env var) lub losowo w trybie test.

---

## Rewriter zapytań

Przed retriwalem zapytanie użytkownika przechodzi przez `query_rewriter.py` (LLM call),
który:
- usuwa slang / błędy ortograficzne
- rozpisuje skróty (np. `IAD` → `Inżynieria i Analiza Danych`)
- przeformułowuje niejasne pytania

Rewriter działa **przed** wzbogacaniem kierunkiem/semestrem.

---

## Modele i konfiguracja

Domyślny pool modeli (`AVAILABLE_MODELS` w `main.py`):
- `openai/gpt-4o-mini`
- `google/gemini-2.5-flash`
- `meta-llama/llama-3-8b-instruct`

Przez OpenRouter — wszystkie modele przez jeden endpoint OpenAI-compatible.

Parametry konfigurowalne przez `modelConfig` (frontend):
- `model`, `temperature` (def. 0.0), `max_tokens` (def. 1024)
- `top_p`, `frequency_penalty`, `presence_penalty`
- `styleInstruction` — dowolna instrukcja stylistyczna (zasada 6)

---

## Pliki PDF jako załączniki

Gdy użytkownik dołączy PDF, frontend odczytuje go jako base64 i wysyła
w polu `attachments` requesta. Backend umieszcza go w wiadomości użytkownika jako
`image_url` z `data:application/pdf;base64,...`.

Wymaga modelu wspierającego multimodalność (GPT-4o, Gemini 2.5, Claude Opus/Sonnet).
Pliki tekstowe (.txt, .md, .csv) nadal parsowane po stronie serwera i wklejane jako tekst.
