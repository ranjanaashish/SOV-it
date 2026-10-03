# Agentic SOV Cleansing & Intelligence System — open-source prototype

Turns messy client **Statement of Values (SOV)** spreadsheets into a strictly-typed,
17-column `Cleaned_SOV.xlsx` plus a full audit log. Four collaborating agents
share one typed state object, and a human approves every change before it touches the data.

**Principle:** *the language model proposes, deterministic code verifies and applies.*
The whole stack is open source and can run fully on-prem. No data leaves the machine
unless you point the LLM at a remote endpoint.

---

## What's new in v0.2

| # | Change | Where |
|---|---|---|
| 1 | **Live pipeline bar, always visible.** It sticks to the top across every tab, is shown before upload too, and **re-draws while the agents run**. It has status pills, a pulse on the active stage, an animated hand-off line, per-stage counts, a "↺ re-reasoned" loop marker and a progress bar. | `ui_components.pipeline_html`, `SOVState.set_listener` |
| 2 | **Inline live editing.** Turn on "✏️ Edit inline" in any review card. Each *distinct* value gets one control (e.g. `Partial` → `N`, `100%` → `Y`) that applies to every row holding it. Values are validated as you type, and a live preview highlights the changed cells. After saving, the card, the before/after tab and the output reflect the edit. Mapping edits show live how well the column's values fit the chosen field. | `app.render_value_editor`, `Orchestrator._apply_edit(value_map=…)`, `values.py` |
| 3 | **Reset knowledge graph** from the sidebar: either learned memory only (keeps the seeded schema, synonyms and rules) or a full rebuild, behind a confirmation checkbox. | `KnowledgeGraph.reset()` |
| 4 | **Visual data-quality report.** Score gauge, KPI tiles, completeness bars against the 95% target, an issues-by-field chart stacked by severity, and a row × field heat-map with hover details. Severity colours are colour-blind checked and always labelled. | `ui_components.py` |
| 5 | **Brand palette.** Background `#1A365D`, primary `#3B82F6`, accent `#007A7C`, highlight `#D9A06F`, text `#F0F4F0`. | `.streamlit/config.toml`, `ui_components.CSS` |
| 6a | **Faster LLM.** Pooled keep-alive HTTP session, response cache, parallel batches (`LLM_MAX_WORKERS`), Ollama `keep_alive` and warm-up, circuit breaker, already-precise items skipped, smaller outputs. | `llm.py`, `DataQualityAgent.explain` |
| 6b | **Rejection notes now work reliably and are understood better.** Notes are submitted from a form, so there's no need to press Enter first; an empty note shows an inline hint. The parser handles negation ("Contents, **not** Building Value"), value maps ("partial means N, 100% is Y" / "N for partial"), "set to …", "values are in thousands", keep / blank / remove rows / make positive, and typos. The LLM is only a fallback, and its output is validated. Every revision shows **🧠 Understood as: …**. | `agents/reasoning.py` |

Also fixed: Agent 4 now blanks (and audits) sprinkler values that are not valid codes. Before, they could slip into the output as text.

**Run from the project folder** (`streamlit run app.py`) so the theme in `.streamlit/config.toml` is applied.

---

## 1. What it does

```
Upload ─► Agent 1 ─► Agent 2 ─► Agent 3 ─► Human gate ─► Agent 4 ─► Export
.xlsx/.csv  Sheet &    Column     Quality +    Approve /      Apply only    Cleaned_SOV.xlsx
            header     mapping    reasoning    reject / edit  approved,     Audit_Log.xlsx/.json
                                     ▲          │             log every     Processing_Summary.json
                                     └─re-reason┘             change
                 ▲            ▲
                 └── Knowledge graph + vector memory (evidence only; learns after sign-off)
```

| Agent | What it does | LLM? |
|---|---|---|
| **1 · Sheet Discovery** | Scans every sheet with no header assumed. Scores header candidates in the first 30 rows (text ratio, fill, insurance vocabulary, density below, uniqueness). Ranks sheets as Primary, Secondary or Reject with reasons, handles merged and two-row headers, and spots totals rows. The reviewer can override the sheet and header row. | No |
| **2 · Schema Mapping** | Pass 0: knowledge-graph memory and vector memory (`REJECTED_AS` edges block repeats). Pass 1: exact match, then the curated synonym dictionary, then RapidFuzz ≥ 0.75. Pass 2: Sentence-BERT similarity blended with a **value profile** (e.g. `Y/N/13R` values mean sprinklers, 4-digit years mean Year Built). Pass 3: an open-source LLM sees only the leftover headers, 5 samples each and neighbouring headers. Code then drops invented columns, targets outside the 17 fields and duplicate targets. Anything below 0.50 is flagged. Hard negatives such as TIV and deductibles are never mapped. | Pass 3 only |
| **3 · Data Quality & Reasoning** | Rule checks per field and per row: missing values, wrong types, currency symbols, K/M/B suffixes, negatives, future or implausible Year Built, Storeys/Buildings < 1, non-integers, invalid sprinkler codes, state names vs codes, ZIP+4 and 4-digit ZIPs, country spelling drift, whitespace, duplicate References, duplicate rows, totals rows, rows with no values. Builds the queue (`column_mapping`, `data_correction`, `standardisation`, `flag_for_review`), each with before/after examples, a rationale, an uncertainty note and a confidence. The LLM rewrites rationales in plain English; templates ensure 100% coverage if the LLM is down. **Re-reasoning:** a rejection note goes back to the agent, which returns a revised item or escalates with an explicit question. | Rationale + re-reasoning |
| **4 · Controlled Transformation** | Plain code. Runs only when the review gate is validated. Works on a copy, applies approved or edited items only, reindexes to the 17 fields in fixed order, casts types, keeps blanks blank and audits every change (who, when, why, before/after, confidence). Validates the written file independently. | Never |

**Human-in-the-loop UI (Streamlit):** workflow diagram driven by the event log, sheet manifest and override, mapping JSON, completeness chart and issue table, a review queue grouped by type and sorted by confidence (Accept / Reject-with-note / Edit), **Approve all ≥ 0.90** (high-severity flags are excluded on purpose), a before/after preview, a blocked-until-reviewed export, downloads, the audit log and a knowledge-graph view.

---

## 2. Requirements

**Runtime**
- Python **3.10+** (3.11 recommended)
- 4 GB RAM minimum (8 GB+ if you run a 7B LLM on the same box)
- Optional: [Ollama](https://ollama.com) or any OpenAI-compatible server (vLLM, llama.cpp, LocalAI, TGI) for the LLM passes. A GPU helps but is not required.

**Python packages** (`requirements.txt`, all permissive licences)

| Package | Purpose | Licence |
|---|---|---|
| pandas ≥ 2.1, openpyxl ≥ 3.1, numpy | data handling, xlsx read/write | BSD / MIT |
| pydantic ≥ 2.5 | typed shared state | MIT |
| rapidfuzz ≥ 3 | fuzzy header matching | MIT |
| sentence-transformers ≥ 2.7 | semantic embeddings (MiniLM / BGE / E5) | Apache-2.0 |
| networkx ≥ 3.4 | knowledge graph | BSD |
| requests | LLM HTTP client (no vendor SDK) | Apache-2.0 |
| streamlit ≥ 1.40 | approval UI | Apache-2.0 |

**Default models** (swap via env vars)
- LLM: `qwen2.5:7b-instruct` (Apache-2.0). Alternatives: `llama3.1:8b-instruct`, `mistral-nemo`, or `qwen2.5:14b-instruct` for better reasoning.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0), ~90 MB, CPU-friendly.

**Graceful degradation:** with no LLM the pipeline still runs end to end in rules-only mode. Without `sentence-transformers` it falls back to tri-gram vectors, and without `rapidfuzz` to `difflib`.

---

## 3. Run it

### Option A — local Python
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional LLM (separate terminal)
ollama pull qwen2.5:7b-instruct
ollama serve

python samples/make_samples.py      # creates 3 synthetic SOVs + ground_truth.json + robustness files
streamlit run app.py                # http://localhost:8501
```

### Option B — Docker (app + Ollama)
```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:7b-instruct
# open http://localhost:8501
```

### Headless evaluation (scores against the ground truth)
```bash
python evaluate.py                         # mapping accuracy, anomaly recall, output validation, robustness
python evaluate.py path/to/your_sov.xlsx   # one file, auto-approve, outputs written to outputs/<name>/
```
`evaluate.py` auto-approves only to measure the pipeline. The product itself never auto-applies.

### Configuration
Everything is set through environment variables; see `.env.example`. Nothing is hard-coded and no API keys are in the code. `LLM_SEND_SAMPLES=0` sends only headers (never cell values) to the LLM.

---

## 4. Project layout
```
app.py                       Streamlit approval UI
evaluate.py                  headless runner + metrics (NFR-1..5)
samples/make_samples.py      synthetic SOVs with planted anomalies + ground truth
sov_agent/
  schema.py                  17-field data dictionary, synonyms, value sets, rules
  state.py                   Pydantic shared state, stage gates, recommendation/audit models
  orchestrator.py            stage-gated hand-offs, human gate, approve-all, export gate
  io_utils.py                safe xlsx/csv ingestion (merged cells, any header position)
  text.py                    parsing helpers (currency, placeholders, fuzzy)
  llm.py                     OpenAI-compatible client for open-source LLM servers
  embeddings.py              sentence-transformers wrapper (+ fallback)
  knowledge_graph.py         NetworkX KG + vector memory, JSON persistence, learning
  agents/
    sheet_discovery.py       Agent 1
    schema_mapping.py        Agent 2
    data_quality.py          Agent 3 (checks, queue, LLM rationales)
    reasoning.py             Agent 3 re-reasoning on rejection
    transformation.py        Agent 4 + export + output validator
```

---

## 5. How it meets the brief

| Requirement | Where |
|---|---|
| FR-1 ingest xlsx/csv, multi-sheet, any header row, 5k rows | `io_utils.py`, Agent 1. `robust_5000_rows.xlsx` is in the samples |
| FR-2 two-pass mapping, confidence + method, < 0.50 flagged, JSON | Agent 2 → `state.mapping_json` (Section 09 format) |
| FR-3 completeness, type checks, logical errors, format issues, per-row/per-field flags | Agent 3 → `state.quality` |
| FR-4 four action types, before/after, rationale, uncertainty, never auto-apply, re-reason | Agent 3 + `reasoning.py` |
| FR-5 accept/reject/edit, side-by-side preview, Approve All ≥ 0.90, export blocked | `app.py`, `Orchestrator.can_export` |
| FR-6 approved only, exact names, cast types, blanks kept, file/sheet names, audit fields | Agent 4, `export_files`, `validate_xlsx` |
| C-01 no auto-transform | review gate + Agent 4 `require("review")` |
| C-02 no hallucinated data | blanks stay blank. Values not castable after review are left blank **and logged**, never guessed. The LLM never writes cell values |
| C-03 exactly 17 columns, fixed order | `TARGET_FIELDS` reindex + validator |
| C-04 ≥ 4 distinct agents, shared state | 4 agent classes, one `SOVState`, stage flags |
| C-05 rationale for every decision | template rationale on every item. The LLM only enhances it |
| C-07 file names | `Cleaned_SOV.xlsx`, `Audit_Log.xlsx` + `Audit_Log.json` |
| NFR-4 no crashes | `IngestError` messages and orchestrator catch-alls. Robustness files are in the samples |
| Bonus | vector memory, live workflow diagram, re-reasoning loop, DQ score at intake, KG learning |

---

## 6. Review of Team Astra's proposed solution

**Overall: strong, and it aligns with the brief.** The "LLM proposes, code verifies" principle, the stage-gated typed state, the knowledge graph with `REJECTED_AS` edges, and learning only after sign-off are all the right calls. This prototype implements that design, with the changes below.

| # | Gap / risk in the proposal | Why it matters | What this prototype does |
|---|---|---|---|
| 1 | **GPT-4o (closed, external API)** | The brief prefers secure handling of commercial data. Sending SOV values to a third party is a privacy and cost risk and adds vendor lock-in. | Open-source LLM through any OpenAI-compatible server (Ollama/vLLM), self-hosted. `LLM_SEND_SAMPLES=0` sends headers only. |
| 2 | **Pass 2 goes straight to the LLM.** SBERT is listed in the stack but not in the flow. | FR-2 explicitly asks for semantic similarity with sentence embeddings in pass 2. The judges will check this. | Embeddings form pass 2, blended with a value profile. The LLM becomes pass 3, only for what is left. |
| 3 | **TIV listed as a synonym** | TIV is the *sum* of building, contents, BI and other. Mapping it to Building Value double counts exposure. | TIV, totals, deductibles and flood zone are hard negatives, excluded with a rationale. |
| 4 | Header detection = "most text cells" | Title rows, notes rows and merged group headers ("Values (USD)") can win. | Weighted score that includes insurance-vocabulary hits and density below. Merged spans are copied, there is a two-row header fallback, and the reviewer can override. |
| 5 | No handling of **totals rows** | A "Total" row in the schedule doubles TIV downstream. | Detected and offered as an *approvable* exclusion (never removed silently). |
| 6 | **Approve All ≥ 0.90** covers everything | Negative TIVs or duplicate References could be bulk-approved without a look. | Approve All skips high-severity flags. |
| 7 | Cast-failure behaviour undefined | Strict typing plus "no fabricated data" collide when a value like `abc` is left in a float column. | The reviewer acknowledges the flag; the value is left blank, logged as `cast_failed_blank`, and can be fixed with Edit. |
| 8 | Duplicate-target rule = "drop" | Two columns can both legitimately look like "Building". | Losers are flagged as `conflict` for review instead of being dropped silently. |
| 9 | Mapping change does not re-run quality checks | If a reviewer remaps a column, the issues for the old target are stale. | Agent 3 re-runs automatically and keeps decisions on unchanged issues. |
| 10 | Zip as integer | `02108` becomes `2108`, losing the leading zero. The schema requires integer, but this should be visible. | Flagged as `zip_short` with an explanation. ZIP+4 is standardised to 5 digits on approval. |
| 11 | Per-finding LLM calls | Thousands of rows → thousands of calls → misses the < 60 s target. | Findings are aggregated per issue type and field and explained in batches of 15. |
| 12 | LangGraph vs custom left open | — | Custom orchestrator with explicit gates. Each method maps one-to-one to a LangGraph node if you want it later. |

---

## 7. Scaling path (all open source)

| Concern | Prototype | Scale-out |
|---|---|---|
| UI | Streamlit | Keep for reviewers. Add a FastAPI service wrapping `Orchestrator` for system-to-system intake |
| Orchestration | in-process, stage-gated | LangGraph (nodes = agents) or Celery/RQ + Redis workers; the state is serialisable (`SOVState.model_dump()`) |
| LLM | Ollama | vLLM / TGI behind a load balancer (OpenAI-compatible, so no code change) |
| Knowledge graph | NetworkX + JSON | Neo4j Community / Memgraph (same node/edge types) |
| Vector memory | NumPy in the KG file | Qdrant / Chroma / pgvector |
| Storage | in-memory + downloads | MinIO (S3-compatible) for files, PostgreSQL for state, audit and decisions |
| Auth / audit | reviewer name field | Keycloak SSO; the audit log is already structured for an append-only store |
| Observability | event log | OpenTelemetry + Prometheus/Grafana; Langfuse (OSS) for LLM traces |

---

## 8. Known limitations
- This was syntax-checked only; it has not been executed in the authoring environment. Run `python samples/make_samples.py && python evaluate.py` first on your machine.
- The real hackathon samples are not included. The synthetic samples mimic the described structure (clean / abbreviated / multi-sheet). Re-score on the real files and extend `SYNONYMS` from the misses.
- Header confidence is a heuristic score, not a calibrated probability. Calibrate the thresholds against labelled files.
- `.xls` (legacy) needs `xlrd`. The app tells the user to save as `.xlsx`.
- Merged cells in data rows are reported only through the missing-value checks; values are not propagated down, by design (C-02).
