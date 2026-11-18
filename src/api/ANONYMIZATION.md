# Anonimizacja promptów

Dokumentacja lokalnej anonimizacji zapytań przed wysłaniem do zewnętrznych modeli (OpenRouter).
Główna logika w `anonymizer.py`, lista pracowników w `staff_names.py`, wpięcie w endpointy w `request_privacy.py`.

---

## Cel

Pytanie użytkownika trafia do zewnętrznych modeli przez OpenRouter (tłumaczenie, przepisanie zapytania,
generowanie odpowiedzi). Zanim cokolwiek wyjdzie z serwera, dane osobowe są lokalnie zamieniane na
placeholdery (`[PESEL]`, `[OSOBA]`, …). Na zewnątrz trafia wyłącznie wersja zamaskowana.

Anonimizacja dotyczy tylko tego, co wychodzi na zewnątrz. Logi i pliki zapisywane lokalnie
(`/feedback`, `/errors`) nie są objęte.

---

## Architektura

```
request (/chat, /chat/stream, /retrieval)
  │
  ├─ pytanie użytkownika
  │     1. regex         — identyfikatory po formacie (PESEL, telefon, adres w 1. osobie, …)
  │     2. model NER     — imiona i nazwiska (GLiNER, w procesie API, CPU, ~0.7 s na VM)
  │     3. kto zostaje   — reguła tytułu („dr / prof. / mgr / inż. X”), lista pracowników,
  │                        patroni („sala im. X”, „twierdzenie X”)
  │
  ├─ załączniki (.txt/.md/.csv, tekst z PDF) — regex, nazwa pliku usuwana
  │
  ├─ historia rozmowy — wiadomości użytkownika jak wyżej (wyniki z cache)
  │
  ▼
tłumaczenie → rewrite → retrieval (lokalny) → OpenRouter
```

- Model NER (`urchade/gliner_multi_pii-v1`, ~300 mln parametrów) nie generuje tekstu: oznacza fragmenty
  „osoba”, a podmiany na `[OSOBA]` dokonuje kod. Nie czyta poleceń, więc prompt injection go nie dotyczy.
  Działa po polsku, angielsku i ukraińsku, także na nazwiskach pisanych małymi literami i z literówkami.
  Słowo-rolę wciągnięte do fragmentu („pani Anna Nowak”, „brat Piotr Nowak”, „student Wydziału MiNI”)
  kod odcina, a resztę sprawdza modelem osobno: „Anna Nowak” jest osobą, „Wydziału MiNI” nie.
- Model nie odróżnia studenta od pracownika. Pracowników chroni reguła tytułu i **lista osób znanych
  chatbotowi**, budowana automatycznie przy starcie API z dwóch publicznych źródeł: linków do stron
  pracowników w `src/ingestion/links_extended.py` (`.../wydzial/pracownicy/prac_<nazwisko>_<imię>/`) oraz
  tekstów w bazie wiedzy (Qdrant): każde nazwisko po tytule naukowym („prof. dr hab. X”) albo po funkcji
  z dwukropkiem („Kierownik dziekanatu: X”). W aktualnej bazie daje to ok. 430 nazwisk, m.in. prowadzących
  z planu zajęć i władze uczelni. Zasada: jeśli bot zna osobę (np. rektora PW), może o niej
  odpowiedzieć; jeśli nie zna, maskowanie nazwiska nic nie kosztuje. Dopasowanie uwzględnia odmianę („Kaczmarskiego”),
  brak polskich znaków („Mandziuk”) i jedną literówkę w nazwisku, ale tylko wtedy, gdy podano też imię
  („Krzysztof Kaczmarsiego”); samo nazwisko z literówką jest maskowane, bo student „Kellner” nie może
  zostać uznany za pracownika „Keller”.
- Cache wyników po hashu SHA-256 tekstu (1000 wpisów, surowy tekst nie jest przechowywany). Frontend odsyła
  całą historię przy każdym pytaniu, więc wcześniejsze wiadomości nie są analizowane ponownie.

---

## Co jest maskowane

| Placeholder | Co | Warstwa |
|-------------|----|---------|
| `[PESEL]` | 11 cyfr, także zmyślony lub z literówką (bez sumy kontrolnej); ze spacjami tylko obok słowa „PESEL” | regex |
| `[TELEFON]` | polskie (9 cyfr, +48) i międzynarodowe (+380, 0044…) | regex |
| `[EMAIL]`, `[IP]` | | regex |
| `[NR_KONTA]`, `[NR_KARTY]` | IBAN / 26 cyfr, karta 13–19 cyfr | regex |
| `[NIP]` | 10 cyfr, także z myślnikami | regex |
| `[NR_DOWODU]`, `[NR_PASZPORTU]`, `[NR_PRAWA_JAZDY]` | `ABC123456`, `AB1234567`, `КВ123456`, `00123/12/1234` | regex |
| `[NR_ALBUMU]` | 5–6 cyfr tylko obok słowa album / indeks / legitymacja / student ID / студентський квиток | regex |
| `[DATA_URODZENIA]` | data po „urodzony / born / date of birth / народився” | regex |
| `[ADRES]` | adres w 1. osobie: „mieszkam na…”, „zameldowany na…”, „mój adres to…”, „na adres ul. …”, „mieszkam w akademiku X, pokój N”, „I live at…”, „живу на…”; razem z numerem mieszkania i kodem pocztowym | regex |
| `[HASLO]` | po „hasło to / password is / пароль:” | regex |
| `[NUMER]` | każdy pozostały ciąg 7+ cyfr | regex |
| `[OSOBA]` | imię i nazwisko osób prywatnych, w dowolnej odmianie | model |

Wszystkie reguły działają na każdym tekście, niezależnie od języka wybranego w interfejsie (PL / EN / UA).

### Czego nie maskujemy

| Co | Dlaczego |
|----|----------|
| Nazwiska pracowników (z tytułem albo z listy) | Poprawne pytania do chatbota |
| Patroni i nazwy od nazwisk („sala im. Kopernika”, „twierdzenie Banacha”, „ul. Noakowskiego”) | To nie osoby w rozmowie |
| Adresy, o które ktoś pyta jako o miejsce („co jest w Rektorskiej 4?”) | Pytanie o miejsce, nie dane osoby |
| Treść pytania („mam depresję, czy dostanę urlop?”) | Bez niej nie da się odpowiedzieć |
| Kontekst z bazy wiedzy | Publiczne strony wydziału |
| Odpowiedzi bota w historii | Powstały z zamaskowanych pytań |

---

## Wpięcie w API

| Endpoint | Co się dzieje |
|----------|---------------|
| `/chat`, `/chat/stream` | `anonymize_request()` zaraz po skróceniu pytania przez ochronę przed nadużyciami (`guard.limit_query`, `security.py`), a przed rezerwacją budżetu: pytanie, historia, załączniki. Wszystkie dalsze kroki dostają wersję zamaskowaną. W `/chat/stream` przed zbudowaniem `StreamingResponse`, żeby błąd dało się zwrócić. |
| `/retrieval` | pytanie anonimizowane przed `rewrite_query` (endpoint wymaga `X-Internal-Key`) |
| pozostałe | bez zmian (nic nie wysyłają na zewnątrz) |

Model ładuje się w tle przy starcie API (`lifespan` w `api.py`), więc API startuje od razu.

### Załączniki

Frontend wkleja pliki .txt/.md/.csv do treści pytania w formacie `"<pytanie>\n\n[Załącznik: <nazwa>]\n<tekst>"`,
ze słowem w języku interfejsu (`Załącznik` / `Attachment` / `Вкладення`).
`request_privacy.py` oddziela pytanie od załącznika: pytanie idzie przez regex i model, treść załącznika przez
regex, a nazwa pliku jest zastępowana przez `[Załącznik]`.

PDF-y nie są wysyłane do modelu jako base64. API wyciąga z nich tekst (`pypdf`), maskuje go regexem
i dołącza do pytania (`build_messages()` dostaje `attachments=[]`). PDF bez warstwy tekstu (skan) albo
uszkodzony jest odrzucany z kodem 422 i komunikatem, żeby wkleić treść jako tekst. Skanu nie da się
zanonimizować bez OCR, więc nie jest wysyłany.

Treść załączników przechodzi tylko przez regex: identyfikatory są maskowane, nazwiska w dokumentach nie.

---

## Kody błędów

| Kod | Kiedy | Komunikat (PL / EN / UA wg `language`) |
|-----|-------|------------------------------------------|
| 503 | model niedostępny i `ANONYMIZER_FALLBACK=block` (np. przez kilkanaście sekund po starcie, przy pierwszym deployu 1–2 min na pobranie modelu) | „Filtr prywatności jest chwilowo niedostępny…” |
| 413 | pytanie dłuższe niż ok. 12 000 znaków (`MAX_CHUNKS` × `CHUNK_CHARS` w `anonymizer.py`); załączniki się nie liczą. W `/chat` w praktyce nie występuje, bo `security.py` wcześniej skraca pytanie do `QUERY_MAX_CHARS` (1000 znaków) | „Wiadomość jest za długa…” |
| 422 | PDF bez warstwy tekstu albo uszkodzony | „Nie udało się odczytać tekstu z pliku PDF…” |

Tryb awaryjny (`ANONYMIZER_FALLBACK`):
- `block` (domyślnie) — prompt nie jest wysyłany, API zwraca 503. Imię i nazwisko razem z kierunkiem
  i semestrem (doklejanymi do zapytań o plan) identyfikują osobę, więc sam regex nie wystarcza.
- `regex` — prompt jest wysyłany z zamaskowanymi tylko identyfikatorami.

Nieudane ładowanie modelu (np. brak sieci) jest ponawiane co 60 s.

---

## Zmienne środowiskowe

| Zmienna | Domyślnie | Opis |
|---------|-----------|------|
| `ANONYMIZER_MODEL` | `urchade/gliner_multi_pii-v1` | model GLiNER z Hugging Face |
| `ANONYMIZER_FALLBACK` | `block` | `block` / `regex` |
| `ANONYMIZER_STAFF_FILE` | — | dodatkowa lista pracowników bez strony w `links_extended.py` (np. dziekanat, władze uczelni): „Imię Nazwisko” albo „Nazwisko” na linię, `#` komentarz |
| `ANONYMIZER_CACHE_DIR` | cache Hugging Face | katalog z pobranym modelem (w Docker Compose wolumen `anonymizer_cache`) |

Uruchomienie: [DEPLOYMENT.md](../../DEPLOYMENT.md).

---

## Zmiany w innych plikach

| Plik | Zmiana |
|------|--------|
| `api.py` | wywołania anonimizacji w `/chat`, `/chat/stream`, `/retrieval`; `lifespan`; `attachments=[]` |
| `translator.py` | prompt tłumacza: tokeny w nawiasach kwadratowych (`[OSOBA]`, `[PESEL]`) pozostają bez zmian |
| `docker-compose.yml` | zmienne `ANONYMIZER_FALLBACK`, `ANONYMIZER_CACHE_DIR` i wolumen `anonymizer_cache` w `api` |
| `environment-linux.yml` | `gliner==0.2.30`, `sentencepiece`, `tiktoken` (numpy zostaje 1.26.4: CPU na VM nie obsługuje numpy ≥ 2.4) |

---

## Ograniczenia

- Osoba, której nie ma w bazie wiedzy ani na stronach pracowników, podana bez tytułu, jest maskowana.
  Bot i tak nie miałby o niej informacji. W razie potrzeby można ją dopisać przez `ANONYMIZER_STAFF_FILE`.
- Nazwiska pracowników pisane cyrylicą („Качмарського”) nie trafiają w listę (łacinka) i są maskowane.
- Nazwiska w treści załączników nie są maskowane (tylko regex).
- Skany PDF nie są obsługiwane.
- Samo imię („jestem Kasia”) zwykle nie jest maskowane.
- Telefon zapisany słownie i login bez hasła („mój login to jkowalski04”) nie są wykrywane.
- Student o takim samym nazwisku jak pracownik, podany bez imienia, jest traktowany jak pracownik.
- Postać historyczna bez słowa-patrona („Kim był Banach?”) jest maskowana.
- W tekście pisanym w całości małymi literami nazwisko po „pan/pani” bywa pomijane
  („do pana marka wiśniewskiego”), bo model sam z siebie nie jest go pewien.

---

## Testy

```bash
pytest src/api/tests
```

`src/api/tests/anonymization_test.py` — model zastąpiony atrapą, testy nie wymagają pobierania modelu.
