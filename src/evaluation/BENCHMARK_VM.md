# Uruchamianie benchmarku na VMce

Benchmark uruchamiamy przez GitHub Actions — tak samo jak deploy, ingest itd. Zakładamy, że chatbot jest już zdeployowany i kontener `api` stoi.

---

## Jak odpalić

1. Wejdź na GitHub → zakładka **Actions**
2. Wybierz workflow **Run Benchmark** z listy po lewej
3. Kliknij **Run workflow** (prawy górny róg)
4. Opcjonalnie zmień parametry:
   - **Rank cut-offs** — wartości k do metryk (domyślnie `3,5,7,10`)
   - **HTTP timeout** — limit czasu na jedno pytanie w sekundach (domyślnie `60`)
5. Kliknij zielony przycisk **Run workflow**

---

## Jak pobrać wyniki

Po zakończeniu workflowu (ikona zielonego checkboxa):

1. Kliknij w ukończony run
2. Przewiń na dół do sekcji **Artifacts**
3. Pobierz archiwum `eval-results-<run_id>.zip`

W archiwum znajdziesz:

| Plik | Zawartość |
|---|---|
| `eval_per_query_<ts>.csv` | Jeden wiersz na pytanie: `query`, `chatbot_answer`, `chatbot_links` (`;`-separated), `gold_link`, `hit@k`, `mrr@k`, `mrrw@k`, `recall@k`, `precision@k`, `f1@k`, `ndcg@k`, `map@k` (dla każdego k), `r_prec` |
| `eval_summary_<ts>.csv` | Jeden wiersz — średnia każdej metryki |
| `eval_summary_<ts>.json` | To samo w JSON |

---

## Które pytania są oceniane

Tylko te z `questions_with_links.csv`, dla których `wymagany kontekst = 0`.  
Pytania z wartościami `1`, `2`, `3`, `0*`, `0**`, `?` są pomijane — nie mają pewnego złotego URL-a.

---

## Jak to działa pod spodem

Workflow (`.github/workflows/benchmark.yml`) uruchamia się na self-hosted runnerze (VMce) i:

1. Checkoutuje kod (ten branch, z którego odpalono workflow)
2. **Buduje obraz** `benchmark` z aktualnie wycheckoutowanego kodu — dlatego zmiany w `benchmark.py` zawsze trafiają do kontenera
3. Tworzy lokalny katalog `src/evaluation/results/` (bind-mount do kontenera)
4. Uruchamia jednorazowy kontener `benchmark` z `docker compose run --rm`, który:
   - wysyła każde pytanie do działającego kontenera `api` przez `http://api:8000/chat`
   - oblicza metryki porównując zwrócone linki ze złotym URL-em
   - zapisuje wyniki do `/app/src/evaluation/results/` (= `src/evaluation/results/` na hoście)
5. Uploaduje pliki z wynikami jako GitHub Actions artifact

Kontener `benchmark` i `api` są w tej samej sieci Docker Compose, więc hostname `api` rozwiązuje się automatycznie.

---

## Rozwiązywanie problemów

**Workflow kończy się błędem `connection refused`:**
- API może nie działać — sprawdź czy deploy jest aktualny i kontener `api` stoi
- Możesz to zweryfikować ręcznie na VMce: `docker ps | grep api`

**Brak plików w Artifacts:**
- Upewnij się, że krok "Run benchmark" się ukończył (nawet z błędem — krok upload ma `if: always()`)
- Sprawdź logi w kroku "Run benchmark" po szczegóły błędu

**Benchmark trwa bardzo długo:**
- Każde pytanie to oddzielne zapytanie HTTP do chatbota (~3–10 s/pytanie)
- Przy ~200 pytaniach typowy czas to 10–30 minut
- Możesz skrócić przez zmniejszenie `--timeout` lub przetestować na `questions_filtered.csv` (dodaj `--input-csv src/evaluation/data/questions_filtered.csv` w komendzie w `docker-compose.yml`)
