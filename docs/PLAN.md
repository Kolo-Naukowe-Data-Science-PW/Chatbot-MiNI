# Implementation plan

Last updated: 2026-04-23.
Branch: `new_pipeline`.

Tasks are ordered by priority. **Tasks 1–4: DONE** ✓. Tasks 5–6 come after local setup.

---

## ✅ Task 1 — MRRw: weighted MRR link metric

**Why**: Novel metric for the DSS presentation. The existing `hierarchical_relevance` in `benchmark.py` scores URLs with a fixed `0.5^depth` decay. The MRRw formula (from `Metryka_chatbot.pdf`) is more expressive: it distinguishes *deeper* (more specific) links from *shallower* (more general) ones, with configurable weights α and β.

**Formula**:

```
w(d) = { 1       if d = 0  (exact match)
        { α^d    if d > 0  (returned URL is deeper by d levels)
        { β^|d|  if d < 0  (returned URL is shallower by |d| levels)

S = max over returned links of  w(d_i) / r_i

MRRw = (1/N) * Σ S_j
```

Default parameters: α = 0.8 (deeper is good), β = 0.4 (shallower is penalised more).

Key conceptual difference from existing code: the current scoring treats *child* URLs as worse than *parents* (child = 0.25, parent = 0.50). MRRw says the opposite — a more specific child URL is *better* (α = 0.8 > β = 0.4), because it's a sign the chatbot found the precise page.

**Files to change**:
- `src/evaluation/benchmark.py` — add `depth_difference()`, `mrr_weight()`, `mrr_weighted_single()`, `mrr_weighted()` functions; include MRRw in `main()` output and bar chart

**Acceptance criteria**:
- `mrr_weighted(gold)` returns a float in [0, 1]
- Edge cases handled: unrelated URLs → S = 0; no returned links → S = 0
- α, β configurable as module-level constants
- Appears in the benchmark output table and `metrics_summary.png`

---

## ✅ Task 2 — Golden answers generator

**Why**: Text-level evaluation of chatbot answers requires a reference to compare against. We use a supermodel (GPT-4o) to generate "ideal" answers for each eval question. These are not RAG answers — the supermodel answers from its own knowledge. We then compare the chatbot's answers against these golden answers using BERTScore and cosine similarity of embeddings.

**What to generate**:
- 1 golden answer per question (cost vs. value tradeoff; can be extended to 2-3 later)
- The supermodel answers in Polish without RAG context, using a system prompt that establishes WMiNI faculty context

**Files to create**:
- `src/evaluation/generate_golden_answers.py` — standalone script; reads `data/questions_filtered.csv`, calls GPT-4o for each question, writes `data/golden_answers.csv` with columns: `query`, `gold_url`, `golden_answer`

**Files to change**:
- `src/evaluation/benchmark.py` — add `evaluate_text_quality()` that loads `golden_answers.csv`, runs chatbot on each question, computes BERTScore F1 between chatbot answer and golden answer, outputs mean BERTScore

**Acceptance criteria**:
- `generate_golden_answers.py` can be interrupted and resumed (skips already-answered rows)
- Output CSV has one row per question, UTF-8 encoded
- `--limit N` flag for cheap testing runs
- BERTScore evaluation produces a single float (mean F1) and saves per-question results to CSV

---

## ✅ Task 3 — Evaluation (5): weighted user feedback — backend

**Why**: Different user groups have very different domain expertise. A dean's office administrator's feedback on answer accuracy is far more valuable than a first-year student's. The current system collects feedback but ignores who gave it.

**User types and weights**:

| user_type | Label | Weight |
|-----------|-------|--------|
| `admin` | Pracownik administracji / pani z dziekanatu | 10 |
| `phd` | Doktorant | 5 |
| `master` | Student studiów magisterskich | 3 |
| `student_senior` | Student II/III roku studiów inżynierskich | 2 |
| `student_junior` | Student I roku studiów inżynierskich | 1 |

**What needs to be built**:

1. **API: `user_type` field in feedback** — add `user_type: str | None` to `FeedbackRequest` in `api.py`; the field is saved to `model_feedback.csv`

2. **Aggregation script** (`src/evaluation/weighted_feedback.py`):
   - Reads `model_feedback.csv`
   - Computes weighted average of ratings per `user_type` weight
   - Outputs a summary table (stdout + optional CSV) with: metric name, unweighted mean, weighted mean, breakdown per user_type
   - Also computes separate averages per language (pl/en/ua) and per model

**Files to change**:
- `src/api/api.py` — add `user_type` to `FeedbackRequest`

**Files to create**:
- `src/evaluation/weighted_feedback.py` — aggregation script

**Acceptance criteria**:
- `FeedbackRequest` accepts `user_type` without breaking existing clients (field is optional)
- `weighted_feedback.py` produces weighted means for `rating` and per-criterion ratings from `ratings` dict
- Script handles missing/unknown `user_type` gracefully (falls back to weight = 1)

---

## ✅ Task 4 — Evaluation (5): weighted user feedback — frontend

**Why**: To collect `user_type`, the chatbot needs a role-selection screen at the start of each session. Additionally, the system prompt should adapt to the user's role — the chatbot should explain things differently to a first-year student vs. a dean's office administrator.

**Entry screen changes**:
- Current: student selects field of study + semester
- New: first select role from: `student_junior` / `student_senior` / `master` / `phd` / `admin`
- If role is `student_*` or `master`: also show field of study + semester selectors (current behaviour)
- If role is `phd` or `admin`: skip field of study / semester

**Per-role system prompt variations** (in `prompt_builder.py`):

| Role | Tone and style adjustments |
|------|---------------------------|
| `student_junior` | Patient, step-by-step explanations, avoid jargon, encourage follow-up questions |
| `student_senior` | Normal (current default) |
| `master` | More concise, assumes familiarity with university procedures |
| `phd` | Focus on research infrastructure, grant procedures, publications |
| `admin` | Direct, formal, procedure-focused; can reference specific regulation numbers and deadlines |

**Frontend flow**:
- `user_type` stored in React state after selection
- Sent with every `/chat` request in `modelConfig` or as a top-level field (TBD — simplest is top-level field in `QueryRequest`)
- Sent with every `/feedback` request in `user_type` field

**Files to change**:
- `src/api/api.py` — add `user_type` to `QueryRequest`; pass to `build_messages`
- `src/api/prompt_builder.py` — accept `user_type`, adjust system prompt accordingly
- `src/frontend/src/App.jsx` — add role-selection screen before conversation starts; include `user_type` in all API calls

**Acceptance criteria**:
- Role selection screen appears before the first message can be sent
- Student roles still show field/semester selectors; PhD and admin skip them
- System prompt visibly differs per role (verified by inspecting API logs)
- `user_type` appears in saved feedback CSV rows

---

## Task 5 — Facebook scraping

**Why**: WRS MiNI posts events on their Facebook page (`facebook.com/wrsminipw`). Students frequently ask about upcoming events. Currently the chatbot has no access to this.

**Approach**: `facebook-scraper` Python library (no API key required, scrapes public pages). Fallback: Playwright if the library gets blocked.

**Risks**: Facebook actively fights scrapers. This may require maintenance. Consider whether events from the faculty website (`ww2.mini.pw.edu.pl/events`) already cover enough of the use case.

**Files to change**:
- `src/ingestion/scraper.py` — add `scrape_facebook_page()` function; integrate with `scrap_data()` for v4
- `chatbot_mini.yml` + `environment-linux.yml` — add `facebook-scraper` dependency

**Acceptance criteria**:
- At least the last 20 posts from WRS MiNI Facebook are scraped and saved as `.txt` in `scraped_raw/`
- Posts include date, text content, and source URL
- Failures are logged and do not crash the main pipeline

---

## Task 6 — Streaming responses (SSE)

**Why**: UX improvement — users see the answer appearing word by word instead of waiting for the full response. Makes the chatbot feel much more responsive.

**Approach**:
- Add `/chat/stream` endpoint in `api.py` using FastAPI `StreamingResponse` + `text/event-stream`
- OpenRouter supports streaming via `stream=True` in the OpenAI SDK
- React frontend handles `EventSource` or `fetch` with `ReadableStream`

**Files to change**:
- `src/api/api.py` — add `/chat/stream` endpoint (keep `/chat` as-is for backward compat)
- `src/api/main.py` — add `query_llm_stream()` generator function
- `src/frontend/src/App.jsx` — switch to streaming fetch for the chat call

**Acceptance criteria**:
- Answer appears token by token in the UI
- Sources are sent as a final SSE event after the stream ends
- `/chat` (non-streaming) still works unchanged

---

## Testing tasks (require local Qdrant setup)

These are run-and-analyse tasks, not coding tasks. They require the full local pipeline to be up (bge-m3 embeddings ingested into Qdrant).

| Task | What to run | Output |
|------|-------------|--------|
| Link depth analysis | `benchmark.py` with extra CSV dump of per-query returned URLs + gold URL | Table: query / returned links / gold link / depth diff |
| Model comparison | `testpro_runner.py` with forced model variants (no random), no personality | CSV comparing GPT-4o-mini vs Gemini 2.5 Flash vs Llama 3.1 |
| Personality comparison | Same with fixed best model, different `styleInstruction` variants | CSV comparing personalities |
| Legal vs general links | `benchmark.py` filtered to legal-category questions vs general questions | Two separate metric tables |

---

## Further work (not in scope for now)

- **OpenTelemetry** instrumentation for the API
- **MCP routing** — separate chatbot for legal documents (graph DB) vs general questions (Qdrant)
- **LLM judge calibration** against dean's office "ideal" answers
- **More user types at chatbot entry** — `researcher` role with publication/grant focus
