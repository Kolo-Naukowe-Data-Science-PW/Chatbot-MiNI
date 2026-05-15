# Coverage Analysis Findings — 2026-05-15

**Source:** `questions_with_links.csv` — tylko pytania z `wymagany kontekst=0`  
**Raport:** `coverage_report_20260515T212447.json`

> Uwaga: analiza obejmuje TYLKO podzbiór z `wymagany kontekst=0`.  
> Należy powtórzyć z `--all-rows` żeby zobaczyć pełny obraz.

---

## Kluczowe liczby

| Metryka | Wartość |
|---|---|
| URLi w Qdrant | **211** |
| Unikalnych gold URLi w benchmarku | 44 |
| Pytań w analizie | 165 |
| Pytań z pokryciem (exact + parent) | 112 / 165 (**67.88%**) |
| Pytań bez pokrycia | 53 / 165 (**32.12%**) |
| Pokrycie URLi (exact) | 23/44 (52.27%) |
| Pokrycie URLi (parent) | 1/44 (2.27%) |
| Brakujące URLe | 20/44 (45.45%) |

---

## Priorytety do zescrapowania (wewnętrzne domeny PW/MiNI)

Posortowane wg liczby pytań w benchmarku — to bezpośrednie odcięcie Hit@k.

| # | URL | Pytań | Akcja |
|---|---|---|---|
| 1 | `ww2.mini.pw.edu.pl/studia/plany-zajec-i-procedury/plan-sesji` | **17** | Dodać do scrape list i reingestować |
| 2 | `bss.ca.pw.edu.pl/Stypendia/.../Stypendium-Rektora` | **9** | Dodać domenę `bss.ca.pw.edu.pl` do scrape list |
| 3 | `ww2.mini.pw.edu.pl/wp-content/uploads/Warunki-rejestracji-na-kolejny-semestr_rok-studiów.pdf` | **5** | Pobrać PDF ręcznie → `manual_pdfs/` |
| 4 | `ww2.mini.pw.edu.pl/studia/plany-zajec-i-procedury/plany-zajec` | **3** | Dodać do scrape list |
| 5 | `ww2.mini.pw.edu.pl/wp-content/uploads/zasadyRekrutacjiMINI_20210217.pdf` | **3** | Pobrać PDF ręcznie → `manual_pdfs/` |
| 6 | `www.bip.pw.edu.pl/Sprawy-Studenckie/Regulamin-studiow-w-Politechnice-Warszawskiej2` | **2** | Sprawdzić czy `regulamin_studiow.pdf` z `manual_pdfs/` jest zaingestowany |
| 7 | `www.ca.pw.edu.pl/Kwestor/Dzial-Plac/Ubezpieczenia/Ubezpieczenie-NNW` | **1** | Dodać URL lub pobrać ręcznie |
| 8 | `ww4.mini.pw.edu.pl/for-students/deans-office` | **1** | Angielska wersja strony dziekanatu — sprawdzić czy istnieje |
| 9 | `ww2.mini.pw.edu.pl/wp-content/uploads/pracownicy/rada/uch_06_2017_07_zal1.pdf` | **1** | Pobrać PDF ręcznie → `manual_pdfs/` |
| 10 | `ww2.mini.pw.edu.pl/wp-content/uploads/20250529-iad-plan-studiow-mgr-4sem.pdf` | **1** | Pobrać PDF ręcznie |

**Łączny gain z naprawienia poz. 1–5:** +37 pytań pokrytych → coverage **67.88% → ~90%**

---

## Poza zakresem (zewnętrzne domeny — nie scrapujemy)

| URL | Pytań | Powód |
|---|---|---|
| `erasmus.pw.edu.pl` | 1 | Zewnętrzna subdomena, różny właściciel |
| `pl.wikipedia.org` | 1 | Wikipedia — nie należy do nas |
| `prawastudenta.edu.pl` | 1 | Zewnętrzny serwis |
| `srs.usos.pw.edu.pl` | 1 | System rezerwacji sal — zewnętrzny |
| `www.okno.pw.edu.pl` | 1 | Rekrutacja PW — nie MiNI |
| `www.wim.pw.edu.pl` | 1 | Inny wydział |
| `pages.mini.pw.edu.pl/~porterj/` | 1 | Strona prywatna prowadzącego |

Te 7 pytań (4.2%) to **górny sufit** cch@k — retrieval nie może ich trafić bo danych w ogóle nie ma.

---

## Błąd w danych (gold CSV)

Jeden URL jest zduplikowany i sklejony:
```
https://www.pw.edu.pl/studia/harmonogram-roku-akademickiegohttps://www.pw.edu.pl/studia/harmonogram-roku-akademickiego
```
To 1 pytanie, które nie zostanie zaliczone jako hit nawet jeśli system zwróci poprawny URL. Należy naprawić w `questions_with_links.csv`.

---

## Parent coverage (częściowe pokrycie)

| URL gold | Pokrycie | Pytań | Uwaga |
|---|---|---|---|
| `ww2.mini.pw.edu.pl/laboratorium/system-druku` | parent | 7 | Strona rodzica (`/laboratorium/`) jest w DB, ale nie konkretna podstrona druku |

7 pytań o drukarki ma częściowe pokrycie — system może trafić na ogólną stronę laboratorium, ale nie na dedykowaną stronę systemu druku.

---

## Następne kroki

1. **Teraz:** Puścić ablacje (rewriter + reranker) — poznamy aktualną jakość retrieval
2. **Potem:** Doscrapować priorytety 1–5 (plan sesji, stypendia, warunki rejestracji)
3. **Opcjonalnie:** Naprawić malformed URL w gold CSV
4. **Powtórzyć coverage analysis** z `--all-rows` żeby poznać pełny obraz całego benchmarku
