# Instrukcja deploymentu chatbota MiNIonek

Chatbot wystawiony jest na: **`chatbotknds.mini.pw.edu.pl:8501`**
Dostęp do VM tylko z sieci wydziałowej (przez `ssh.mini.pw.edu.pl`).

---

## Kiedy używać którego workflow

| Workflow | Kiedy odpalić |
|---|---|
| **Scrape and Ingest** | Pełna re-ingestia: nowe linki w `links_extended.py`, nowe PDFy, zmiana promptu do faktów |
| **Ingest Only** | Dodałaś/zmieniłaś PDFy w `src/manual_pdfs/`, scraping już był — tylko ponów ekstrakcję faktów |
| **Deploy Chatbot MiNI** | Zmieniłaś kod (api.py, frontend, prompt_builder) bez zmiany danych |
| **Run Tests** | Ręcznie, przed mergem PR — testy jednostkowe na ubuntu-latest |

Żaden workflow nie odpala się automatycznie przy merge PR.

---

## Krok po kroku: wypchnięcie na nowy branch

### 1. Utwórz branch `chatbot_v3`

```bash
git checkout -b chatbot_v3
```

### 2. Dodaj PDFy (raz, przed pierwszym ingestem)

Skopiuj pobrane PDFy do katalogu:
```
src/manual_pdfs/
```

Pliki z BIP PW z zakomentowanym mapowaniem URL — dodaj je do `URL_MAP`
w `src/ingestion/ingest_manual_pdfs.py`, żeby miały poprawne metadane źródłowe.

### 3. Zacommituj i wypchnij

```bash
git add src/ingestion/
git add docker-compose.yml
git add .github/workflows/
git add src/frontend/vite.config.js
git add src/manual_pdfs/          # katalog + PDFy
git commit -m "feat: chatbot_v3 — new pipeline, restructured workflows"
git push -u origin chatbot_v3
```

---

## Krok po kroku: uruchomienie scrapingu i ingestii

1. Wejdź na GitHub → zakładka **Actions**
2. Kliknij **"Scrape and Ingest"** → **"Run workflow"**
3. Wybierz branch: `chatbot_v3`
4. Kliknij **"Run workflow"**

Workflow:
- uruchamia scraper (zescrapuje wszystkie linki z `links_extended.py`, domyślnie `PIPELINE_VERSION=3`)
- uruchamia ingest: `ingest_manual_pdfs` → `describe_files` → `extract_facts` → `ingest_facts`

Czas: kilkanaście–kilkadziesiąt minut zależnie od liczby URLi i kosztów API.

---

## Krok po kroku: uruchomienie chatbota (deploy)

1. Wejdź na GitHub → zakładka **Actions**
2. Kliknij **"Deploy Chatbot MiNI"** → **"Run workflow"**
3. Wybierz branch: `chatbot_v3`
4. Kliknij **"Run workflow"**

Workflow zatrzymuje stare kontenery (`docker compose down`) i startuje nowe (`docker compose up -d --build`).

Chatbot dostępny pod: **`chatbotknds.mini.pw.edu.pl:8501`**

---

## Krok po kroku: sam ingest (bez scrapowania)

Używaj gdy PDFy się zmieniły albo zmienił się prompt do faktów, ale linki webowe nie.

1. Wrzuć PDFy do `src/manual_pdfs/`
2. Zacommituj i pushuj na `chatbot_v3`
3. GitHub Actions → **"Ingest Only"** → **"Run workflow"** → branch `chatbot_v3`

---

## PDFy z BIP PW — lista do ręcznego pobrania

Poniższe pliki PDF pochodzą z BIP PW i nie mogą być automatycznie scrapowane
(cookiewall / JS rendering). Pobierz ręcznie i wrzuć do `src/manual_pdfs/`.

### PDFy z sekcji "Ważne dokumenty" (links_extended.py)

```
https://bip.pw.edu.pl/var/pw/storage/original/application/6ae70089d460bc9ba71566298fa79fcf.pdf
https://bip.pw.edu.pl/var/pw/storage/original/application/5e62794f9cf903f4ef2e16869f704020.pdf
https://bip.pw.edu.pl/var/pw/storage/original/application/09dde30f2523fec5088b62dbdd727a73.pdf
https://bip.pw.edu.pl/var/pw/storage/original/application/df978333880183b4bc4906362e74a2f8.pdf
```

### PDFy z sekcji "BIP PW" (links_extended.py)

```
https://www.bip.pw.edu.pl/var/pw/storage/original/application/d306a4288f0943c31b5e9cd8fcd33f73.pdf
https://www.bip.pw.edu.pl/var/pw/storage/original/application/a7f6351019e1d70a951c7a1ac1bb0e28.pdf
https://www.bip.pw.edu.pl/var/pw/storage/original/application/62f1b26a2cee774b5699342e29970bbf.pdf
https://www.bip.pw.edu.pl/var/pw/storage/original/application/bc54edfe5cc419713f931181db92ef46.pdf
https://www.bip.pw.edu.pl/var/pw/storage/original/application/a3de1af7fada945beb87a825dbb0c262.pdf
https://www.bip.pw.edu.pl/var/pw/storage/original/application/b316729d09938e94c7eb5528b0e0744d.pdf
https://bip.pw.edu.pl/var/pw/storage/original/application/7fe181576ef6f818438bc0b3df98e13f.pdf
https://www.bip.pw.edu.pl/var/pw/storage/original/application/84cf2faf2873685ac94833c869fe866f.pdf
https://bip.pw.edu.pl/var/pw/storage/original/application/35ef86bbcf1cd086fee4fd97f088aa47.pdf
https://bip.pw.edu.pl/var/pw/storage/original/application/c82085781d743edb519bd365df87aded.pdf
https://www.bip.pw.edu.pl/var/pw/storage/original/application/ee0f54fc508ca6edb94560f70b4dcd8b.pdf
https://bip.pw.edu.pl/var/pw/storage/original/application/390bf595ba50c2b63d98e7056c54b61f.pdf
https://www.bip.pw.edu.pl/var/pw/storage/original/application/2b1c27e1c5245605dfdf71342a8b11b9.pdf
https://www.bip.pw.edu.pl/var/pw/storage/original/application/e047e0025a88927817f200f6ef12d364.pdf
```

Po pobraniu: dodaj mapowanie `filename → URL` w `src/ingestion/ingest_manual_pdfs.py`
w słowniku `URL_MAP`, żeby fakty miały poprawne metadane źródłowe.

---

## Architektura deploymentu

```
przeglądarka
    ↓ port 8501
frontend (Docker — npm run preview)
    ↓ /api/* → proxy → http://api:8000
api (Docker — uvicorn)
    ↓
qdrant_db (Docker volume)
```

Frontend serwuje pre-built React app (Vite preview server).
Proxy `/api` → `http://api:8000` działa przez Docker internal DNS — `api` to
nazwa serwisu w docker-compose.yml.
