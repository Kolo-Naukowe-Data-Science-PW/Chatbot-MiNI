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
  ├─ załączniki (PDF, DOCX, XLSX, TXT/MD/CSV/JSON — tekst z POST /attachments) — regex,
  │                        nazwa pliku zamieniana na „plik N (.ext)”; obrazy bez zmian (patrz niżej)
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

Od uploadu plików w `chatbot_final` (opis: [ATTACHMENTS.md](ATTACHMENTS.md)) frontend nie wkleja już
plików do pytania. Każdy plik jest wysyłany do `POST /attachments`, który wyciąga z niego tekst
(PDF, DOCX, XLSX, TXT/MD/CSV/JSON) albo przygotowuje obraz, a potem wraca w polu `attachments`
requestu `/chat` / `/chat/stream`. Tam `_prepare_attachments()` w `api.py` woła `mask_attachments()`
z `request_privacy.py`:

- treść każdego załącznika przechodzi przez regex (`mask_identifiers`) — PESEL, telefon, e-mail itd.,
- prawdziwa nazwa pliku nie trafia do modelu — zamieniana na „plik 1 (.pdf)”, „plik 2 (.docx)”…,
- to samo dotyczy załączników z historii rozmowy (`mask_history_attachments()`).

`anonymize_request()` dostaje request z `attachments=[]`, więc stara ścieżka dla PDF-ów w base64
(`pdf_attachments_as_text`) nie jest już używana przez API, ale została w module (testy, zgodność).
Stary format „pytanie + `[Załącznik: nazwa]` + tekst” nadal jest rozpoznawany przez `split_attachment()`.

**Obrazy i skany PDF** nie dają się zamaskować regexem ani modelem NER. Domyślnie
(`ATTACHMENT_ALLOW_IMAGES=true`) są wysyłane do modelu jako obrazy — to świadomy wyjątek, bo obsługa
JPG/PNG była wymaganiem. Ustawienie `ATTACHMENT_ALLOW_IMAGES=false` przywraca wcześniejszą politykę:
obrazy są odrzucane (HTTP 415), a skany PDF — HTTP 422 z prośbą o wklejenie tekstu.

Treść załączników przechodzi tylko przez regex: identyfikatory są maskowane, nazwiska w dokumentach nie.

---

## Kody błędów

| Kod | Kiedy | Komunikat (PL / EN / UA wg `language`) |
|-----|-------|------------------------------------------|
| 503 | model niedostępny i `ANONYMIZER_FALLBACK=block` (np. przez kilkanaście sekund po starcie, przy pierwszym deployu 1–2 min na pobranie modelu) | „Filtr prywatności jest chwilowo niedostępny…” |
| 413 | pytanie dłuższe niż ok. 12 000 znaków (`MAX_CHUNKS` × `CHUNK_CHARS` w `anonymizer.py`); załączniki się nie liczą. W `/chat` w praktyce nie występuje, bo `security.py` wcześniej skraca pytanie do `QUERY_MAX_CHARS` (1000 znaków) | „Wiadomość jest za długa…” |
| 422 | uszkodzony plik albo (przy `ATTACHMENT_ALLOW_IMAGES=false`) skan PDF | komunikat z `attachments.py`, np. „Nie udało się odczytać pliku PDF…” |
| 415 | nieobsługiwany format albo obraz przy `ATTACHMENT_ALLOW_IMAGES=false` | „Nieobsługiwany format pliku…” / „Obrazy nie są obecnie przyjmowane…” |

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
| `api.py` | wywołania anonimizacji w `/chat`, `/chat/stream`, `/retrieval`; `lifespan`; maskowanie załączników w `_prepare_attachments()` / `_start_chat()` |
| `translator.py` | prompt tłumacza: tokeny w nawiasach kwadratowych (`[OSOBA]`, `[PESEL]`) pozostają bez zmian |
| `docker-compose.yml` | zmienne `ANONYMIZER_FALLBACK`, `ANONYMIZER_CACHE_DIR` i wolumen `anonymizer_cache` w `api` |
| `environment-linux.yml` | `gliner==0.2.30`, `sentencepiece`, `tiktoken` (numpy zostaje 1.26.4: CPU na VM nie obsługuje numpy ≥ 2.4) |

---

## Ograniczenia

- Osoba, której nie ma w bazie wiedzy ani na stronach pracowników, podana bez tytułu, jest maskowana.
  Bot i tak nie miałby o niej informacji. W razie potrzeby można ją dopisać przez `ANONYMIZER_STAFF_FILE`.
- Nazwiska pracowników pisane cyrylicą („Качмарського”) nie trafiają w listę (łacinka) i są maskowane.
- Nazwiska w treści załączników nie są maskowane (tylko regex).
- Obrazy i skany PDF nie są anonimizowane (wysyłane jako obrazy, chyba że `ATTACHMENT_ALLOW_IMAGES=false`).
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
