# Rejestracja self-hosted runnera na laptopie (Ubuntu)

## Bezpieczeństwo — przeczytaj najpierw

- Runner na laptopie obsługuje TYLKO workflow `run_experiment.yml` (label `local-gpu`)
- VMka wydziałowa NIE MA labela `local-gpu` → nie może podjąć eksperymentów
- Laptop NIE MA labela `self-hosted` (nie dodawaj go!) → nie może podjąć jobów produkcyjnych z VMki
- Oba runnery działają w tym samym repo, ale dzięki labelom są od siebie całkowicie izolowane

---

## Krok 1 — Pobierz token rejestracyjny

1. Wejdź na GitHub: `https://github.com/<twoja-org>/<repo>/settings/actions/runners`
2. Kliknij **"New self-hosted runner"**
3. Wybierz: **Linux** → **x64**
4. Skopiuj token z sekcji "Configure" — wygląda tak:
   ```
   --token AXXXXXXXXXXXXXXXXXXXXXXXXX
   ```
   Token jest ważny przez 1 godzinę.

---

## Krok 2 — Zainstaluj runnera na laptopie

Wykonaj na laptopie (Ubuntu):

```bash
# Utwórz dedykowany katalog
mkdir -p ~/actions-runner && cd ~/actions-runner

# Pobierz runnera (sprawdź aktualną wersję na stronie GitHub — podmień X.X.X)
curl -o actions-runner-linux-x64-2.321.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-linux-x64-2.321.0.tar.gz

# Wypakuj
tar xzf ./actions-runner-linux-x64-2.321.0.tar.gz
```

> Aktualną wersję znajdziesz na stronie GitHub w kroku 1 — GitHub podaje gotową komendę curl.

---

## Krok 3 — Zarejestruj runnera z labelem local-gpu

```bash
./config.sh \
  --url https://github.com/<twoja-org>/<repo> \
  --token AXXXXXXXXXXXXXXXXXXXXXXXXX \
  --name laptop-basia \
  --labels local-gpu \
  --work _work \
  --unattended
```

**Krytyczne flagi:**
- `--labels local-gpu` — TYLKO ten label, nie dodawaj `self-hosted`
- `--name laptop-basia` — dowolna nazwa, żebyś wiedziała który runner to laptop
- `--unattended` — nie pyta o nic interaktywnie

Po rejestracji wejdź na GitHub → Settings → Actions → Runners i sprawdź, że widzisz **dwa runnery**:
- runner VMki (label: `self-hosted`) — status: Active
- `laptop-basia` (label: `local-gpu`) — status: Offline (bo jeszcze nie uruchomiony)

---

## Krok 4 — Uruchom runnera jako serwis systemd

Żeby runner startował automatycznie po uruchomieniu laptopa:

```bash
# Zainstaluj serwis
sudo ./svc.sh install

# Uruchom teraz
sudo ./svc.sh start

# Sprawdź status
sudo ./svc.sh status
```

Runner zmieni status na **Idle** w panelu GitHub.

Żeby zatrzymać (np. przed wyłączeniem laptopa):
```bash
sudo ./svc.sh stop
```

---

## Krok 5 — Weryfikacja izolacji (zrób to przed pierwszym eksperymentem)

### Sprawdź runnery w panelu GitHub

Przejdź do `Settings → Actions → Runners` i upewnij się że:

| Runner | Labels | Opis |
|--------|--------|------|
| runner VMki | `self-hosted` | Produkcja — chatbot |
| laptop-basia | `local-gpu` | Eksperymenty |

**Nie powinno być runnera z oboma labelami jednocześnie.**

### Sprawdź konfigurację workflow eksperymentów

Upewnij się że w `.github/workflows/run_experiment.yml` jest:
```yaml
runs-on: local-gpu
```

### Sprawdź konfiguracje workflow produkcyjnych

Upewnij się że WSZYSTKIE poniższe workflow mają `runs-on: self-hosted` (nie `local-gpu`):
- `.github/workflows/deploy.yml` (lub jak się nazywa deploy chatbota)
- `.github/workflows/ingest_schedule_only.yml`
- dowolny inny workflow związany z chatbotem

```bash
# Szybka weryfikacja — uruchom lokalnie:
grep -r "runs-on" .github/workflows/
```

Oczekiwany wynik:
```
run_experiment.yml:    runs-on: local-gpu
ingest_schedule_only.yml:    runs-on: self-hosted
deploy.yml (lub inna nazwa):    runs-on: self-hosted
```

---

## Odpalanie eksperymentu przez GitHub Actions

1. Wejdź na GitHub → zakładka **Actions**
2. Z lewej listy wybierz workflow **"Run Experiment"**
3. Kliknij **"Run workflow"** → pojawi się formularz:
   - **variant**: np. `baseline`
   - **testset**: `human` / `rag` / `both`
   - **judge_model**: np. `openai/gpt-4o-mini`
   - **n_questions**: `20` (najpierw mały test!)
4. Kliknij **"Run workflow"** (zielony przycisk)
5. Job pojawi się w kolejce — podejmie go laptop gdy jest online

---

## Pierwsze uruchomienie — checklist

Zanim odpalisz pełny eksperyment:

- [ ] Runner `laptop-basia` widoczny w panelu GitHub ze statusem **Idle**
- [ ] `grep -r "runs-on" .github/workflows/` pokazuje właściwe labele
- [ ] Chatbot produkcyjny działa (`curl https://chatbotknds.mini.pw.edu.pl/api/health` lub odpytaj go ręcznie)
- [ ] Uruchom najpierw `--variant baseline --testset human --n-questions 5` jako smoke test
- [ ] Po smoke teście sprawdź że chatbot produkcyjny nadal działa

---

## Zatrzymanie runnera / Usunięcie

Żeby tymczasowo zatrzymać (runner nie będzie podejmował jobów):
```bash
sudo ./svc.sh stop
```

Żeby trwale usunąć:
```bash
sudo ./svc.sh uninstall
./config.sh remove --token <nowy-token-z-github>
```
