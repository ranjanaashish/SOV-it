<div align="center">

# SOV-it
### Agentic Statement of Values Cleansing and Intelligence System
*An autonomous multi-agent platform for commercial property underwriting data transformation, schema mapping, and validation.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-7BDCB5.svg?style=flat)](LICENSE)
[![Architecture: Multi-Agent](https://img.shields.io/badge/Architecture-Autonomous_Multi--Agent-1A2C47.svg?style=flat)](#high-level-system-architecture)
[![Team Astra](https://img.shields.io/badge/Developed_by-Team_Astra-7BDCB5.svg?style=flat)](#contributors-and-institutional-partners)

---

**Developed by Team Astra**  
*As part of the ADROSONIC Build Hackathon*  
*In collaboration with **Adrosonic** and **Birla Institute of Technology, Mesra (BIT Mesra)***

</div>

---

## Executive Summary

Underwriting commercial property insurance requires processing unstructured, non-standardised client Statements of Values (SOV) spreadsheets. Real-world client submissions frequently contain unaligned column headers, mixed alphanumeric notations, compound value errors, missing geographic attributes, and inconsistent building occupancy classifications. These defects lead to operational delays, accumulation blindspots, and manual data-entry overhead.

**SOV-it** is an enterprise-grade agentic platform that automates the ingestion, mapping, validation, and transformation of complex client SOVs into standardised, verified schemas.

### Core Architectural Principle
> **The language model proposes; deterministic logic and human underwriters verify and commit.**

The platform enforces **100 per cent human-in-the-loop governance**, persistent graph-based institutional memory across submissions, and a certified transformation gate guaranteeing zero schema drift.

---

## High-Level System Architecture

<div align="center">
  <img src="assets/architecture_diagram.svg" alt="SOV-it High-Level System Architecture" width="100%" />
</div>

---

## Autonomous Agent Architecture

| Agent | Responsibility | Core Methodology |
| :--- | :--- | :--- |
| **Agent 1: Sheet and Header Discovery** | Analyzes multi-tab workbooks without assumed structure. Scans candidates across rows 1–30 based on text ratio, density, insurance domain vocabulary, and value distribution below. Identifies totals rows, title banners, and merged cells. | Density-distribution heuristics, vocabulary scoring, OpenPyXL grid parsing. |
| **Agent 2: Hierarchical Schema Mapper** | Maps source columns to the 17 standard insurance target fields via a tiered resolution strategy: <br>• **Tier 0:** Knowledge Graph historical memory and vector search.<br>• **Tier 1:** Curated synonym dictionaries and RapidFuzz token matching ($\ge 0.75$).<br>• **Tier 2:** Semantic Sentence-BERT embeddings blended with value profiling.<br>• **Tier 3:** Zero-shot constrained LLM reasoning for remaining ambiguous headers. | NetworkX Graph, Sentence-BERT, RapidFuzz, Pluggable LLM. |
| **Agent 3: Data Quality and Reasoning Engine** | Validates over 20 underwriting rules including currency parsing, multiplier expansion (K, M, B), negative value detection, construction code normalization, ZIP/State consistency, and Year Built plausibility. Generates formal explanations and executes **adaptive re-reasoning loops** upon underwriter rejection. | Rulebook engine, regular expressions, contextual re-reasoning. |
| **Agent 4: Controlled Transformation** | Deterministic transformation engine with zero generative hallucination risk. Applies only underwriter-approved modifications, retains raw data on an immutable source copy, enforces strict schema typing, and generates cell-level provenance audit logs. | Pandas, OpenPyXL, cryptographic provenance hashing. |

---

## Target Schema Specification (17 Standard Fields)

SOV-it standardizes all incoming data into the 17-field commercial insurance property schema:

| Target Field | Data Type | Permissible Range / Formats | Validation and Cleansing Logic |
| :--- | :--- | :--- | :--- |
| `Location ID` | String / Alphanumeric | Unique property identifier | Trimmed, deduplicated, preserves leading zeros |
| `Street Address` | String | Postal street address | Cleaned punctuation, casing standardized |
| `City` | String | City name | Standardized casing, whitespace stripped |
| `State` | String | 2-Letter US Postal Code (`CA`, `NY`) | Resolves full state names to standard 2-letter codes |
| `Zip Code` | String | 5-Digit or 9-Digit (`12345`, `12345-6789`) | Padded leading zeros, standardizes hyphenation |
| `County` | String | County jurisdiction | Normalized naming conventions |
| `Country` | String | ISO country code or standard name | Resolves spelling drift (`USA`, `US`, `United States`) |
| `Building Value` | Float | $\ge 0$ | Strips currency symbols, expands numeric multipliers |
| `Contents Value` | Float | $\ge 0$ | Numeric parsing, negative values flagged |
| `Business Interruption Value` | Float | $\ge 0$ | Standardized financial evaluation |
| `Total Insurable Value (TIV)` | Float | $\ge 0$ | Cross-checked against sum of component values |
| `Square Footage` | Float | $\ge 0$ | Parses commas and area units (`sq ft`, `sf`) |
| `Number of Stories` | Integer | $\ge 1$ | Cast to whole integers; fractional stories flagged |
| `Year Built` | Integer | $1700 \le \text{Year} \le \text{Current Year}$ | Flags future years; resolves two-digit years |
| `Construction Type` | Categorical | ISO Construction Classes (1–6, `MFR`, `NC`) | Normalized to standard structural classifications |
| `Occupancy Type` | Categorical | Commercial occupancy codes (`Office`, `Retail`) | Curated insurance categorization |
| `Sprinkler Flag` | Categorical | `Y` / `N` / Valid sprinkler class (`13`, `13R`) | Standardized binary and sprinkler class indicators |

---

## Installation and Quickstart

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/ranjanaashish/SOV-it.git
cd SOV-it

# 2. Configure virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```
*The local interface will be accessible at `http://localhost:8501`.*

### Docker Deployment

```bash
docker compose up -d --build
```

---

## LLM Configuration and Provider Support

SOV-it supports multiple inference backends configurable directly via the sidebar:

- **Groq Cloud** (`llama-3.3-70b-versatile` — high-throughput inference)
- **Ollama Local** (`qwen2.5:7b-instruct` / `qwen2.5:3b-instruct` — fully on-premise, offline execution)
- **OpenAI / Azure OpenAI** (`gpt-4o-mini`, `gpt-4o`)
- **Google Gemini** (`gemini-1.5-flash` via OpenAI-compatible endpoint)
- **Custom OpenAI Endpoints** (vLLM, LM Studio, LocalAI)
- **Rules-Only Deterministic Mode** (Instant execution with zero network dependency)

---
Agent 1 evaluates candidate header rows across the first $R_{\max} = 30$ rows of every sheet to detect table bounds without assuming fixed layouts.
#### 1. Header Candidate Scoring
For each candidate row $r \in \{1, \dots, \min(30, N_{\text{rows}})\}$, the heuristic score $S_{\text{header}}(r) \in [0, 1]$ is a weighted linear combination:
$$S_{\text{header}}(r) = w_{\text{text}} R_{\text{text}}(r) + w_{\text{fill}} F_{\text{fill}}(r) + w_{\text{vocab}} V_{\text{vocab}}(r) + w_{\text{density}} D_{\text{below}}(r) + w_{\text{uniq}} U_{\text{unique}}(r)$$
Where the empirical weights are:
$$w_{\text{text}} = 0.25, \quad w_{\text{fill}} = 0.15, \quad w_{\text{vocab}} = 0.30, \quad w_{\text{density}} = 0.20, \quad w_{\text{uniq}} = 0.10$$
Component definitions:
- **Text Ratio $R_{\text{text}}(r)$:** Proportion of non-empty cells that are non-numeric string labels:
  $$R_{\text{text}}(r) = \frac{|\{v \in \mathbf{v}_r \mid v \in \text{String} \wedge v \notin \mathbb{R}\}|}{|\mathbf{v}_r|}$$
- **Fill Ratio $F_{\text{fill}}(r)$:** Horizontal completeness against the active sheet width $W_{\text{active}}$:
  $$F_{\text{fill}}(r) = \frac{|\mathbf{v}_r|}{W_{\text{active}}}$$
- **Uniqueness $U_{\text{unique}}(r)$:** Distinctness of normalized cell values to penalize repeated data rows:
  $$U_{\text{unique}}(r) = \frac{|\{\text{norm}(v) \mid v \in \mathbf{v}_r\}|}{|\mathbf{v}_r|}$$
- **Insurance Vocabulary Hits $V_{\text{vocab}}(r)$:** Proportion matching the domain alias index $\mathcal{A}$:
  $$V_{\text{vocab}}(r) = \frac{1}{|\mathbf{v}_r|} \sum_{v \in \mathbf{v}_r} \mathbf{1}\left[ \text{norm}(v) \in \mathcal{A} \lor \max_{a \in \mathcal{A}} \text{Sim}_{\text{RapidFuzz}}(\text{norm}(v), a) \ge 0.85 \right]$$
- **Look-Ahead Body Density $D_{\text{below}}(r)$:** Mean cell presence across the succeeding 10 data rows:
  $$D_{\text{below}}(r) = \frac{1}{10} \sum_{k=r+1}^{r+10} \left( \frac{1}{|\text{cols}(r)|} \sum_{c \in \text{cols}(r)} \mathbf{1}[M_{k, c} \ne \emptyset] \right)$$
#### 2. Sheet Ranking & Classification Score
For a sheet $s$ with optimal candidate header row $r^* = \arg\max_r S_{\text{header}}(r)$, the sheet confidence $C_{\text{sheet}} \in [0, 1]$ is:
$$S_{\text{sheet}} = 0.35 \cdot S_{\text{header}}(r^*) + 0.20 \cdot (1 - R_{\text{null}}) + 0.20 \cdot \min\left(\frac{N_{\text{rows}}}{20}, 1\right) + 0.15 \cdot \min\left(\frac{N_{\text{cols}}}{8}, 1\right) + 0.10 \cdot V_{\text{vocab}}(r^*) - \sum P_i$$
Penalties:
- $P_{\text{name}} = 0.25$ if sheet name matches non-SOV patterns (`summary|cover|note|glossary|...`)
- $P_{\text{sparse}} = 0.20$ if total data rows $N_{\text{rows}} < 3$
- Clamped confidence: $C_{\text{sheet}} = \text{clip}(S_{\text{sheet}}, 0.0, 1.0)$
Classification decision boundary:
$$\text{Class}(s) = \begin{cases} \text{Primary} & \text{if } C_{\text{sheet}} \ge 0.45 \text{ and is } \max_{s'} C_{\text{sheet}}(s') \\ \text{Secondary} & \text{if } C_{\text{sheet}} \ge 0.25 \\ \text{Reject} & \text{if } C_{\text{sheet}} < 0.25 \end{cases}$$
---
#### 2. Pass 1: String Metrology & Fuzzy Matching
- **Exact Match:** $C = 1.0$ if $\text{norm}(h) \equiv \text{norm}(t)$.
- **Curated Synonym Match:** $C = 0.96$ if $\text{norm}(h) \in \text{Synonyms}(t)$.
- **RapidFuzz Token Sort / Ratio:**
  For similarity score $S_{\text{fuzzy}} \ge 0.75$:
  $$C_{\text{fuzzy}} = \min\left(0.93, 0.75 + (S_{\text{fuzzy}} - 0.75) \times 0.80\right)$$
#### 3. Pass 2: Semantic Embeddings & Value Profiling
When string matching yields no definitive hit, semantic cosine similarity is evaluated against target field definitions and curated exemplars:
$$S_{\text{semantic}}(h, t) = \max_{d \in \mathcal{D}_t} \frac{\mathbf{e}_h \cdot \mathbf{e}_d}{\|\mathbf{e}_h\|_2 \|\mathbf{e}_d\|_2}$$
*Tri-gram Vector Fallback:* If PyTorch is unavailable, embeddings fall back to $D = 4096$-dimensional CRC32-hashed character tri-gram vectors:
$$\mathbf{v}[k] = \sum_{\tau \in \text{trigrams}(h)} \mathbf{1}[\text{crc32}(\tau) \pmod{4096} = k], \quad \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
*Empirical Value Profile Compatibility $F_{\text{comp}}(t, \mathcal{P})$:*
Data rows are sampled ($N \le 300$) to construct an empirical feature vector:
$$\mathcal{P} = \left[ f_{\text{num}}, f_{\text{year}}, f_{\text{zip}}, f_{\text{small\_int}}, f_{\text{money}}, f_{\text{state}}, f_{\text{sprinkler}}, f_{\text{country}}, f_{\text{text}}, f_{\text{addr}}, f_{\text{unique}} \right]$$
Compatibility scoring examples:
- **Year Built:** $F_{\text{comp}} = f_{\text{year}} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[1700 \le x_i \le \text{Year}_{\text{current}} + 5]$
- **Monetary Fields:** $F_{\text{comp}} = f_{\text{money}} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[|x_i| \ge 1000 \lor \text{format} = \text{"currency"}]$
- **Fire Sprinklers:** $F_{\text{comp}} = f_{\text{spr}} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[\text{norm}(s_i) \in \{\text{Y}, \text{N}, \text{13}, \text{13R}, \dots\}]$
- **Storeys / Buildings:** $F_{\text{comp}} = f_{\text{small\_int}} \cdot (1 - f_{\text{year}})$ where $f_{\text{small\_int}} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[0 \le x_i \le 200]$
*Blended Semantic Confidence:*
$$S_{\text{combined}} = 0.75 \cdot S_{\text{semantic}} + 0.25 \cdot F_{\text{comp}}$$
*Profile Verification Penalties & Boosts:*
- If $F_{\text{comp}} < 0.20$: $C \leftarrow \max(0.0, C - \Delta_{\text{penalty}})$ where $\Delta_{\text{penalty}} \in \{0.10, 0.20\}$.
- If $F_{\text{comp}} \ge 0.80$: $C \leftarrow \min(C_{\max}, C + 0.05)$.
#### 4. Competitive Conflict Resolution
When multiple source columns $\{c_1, c_2, \dots\}$ compete for the same target field $t$:
$$c^* = \arg\max_{c_j} C(c_j \to t)$$
Column $c^*$ retains assignment $t$. Competitors $c_k \ne c^*$ are demoted to conflicts:
$$C(c_k) \leftarrow \min(C(c_k), 0.45), \quad \text{flag} \leftarrow \text{"conflict"}$$
Aggregate mapping confidence:
$$\bar{C}_{\text{mapping}} = \frac{1}{|\mathcal{M}_{\text{mapped}}|} \sum_{m \in \mathcal{M}_{\text{mapped}}} C(m)$$
---
### 2.3 Agent 3 — Data Quality, Anomaly Detection & Scoring
Agent 3 scans row vectors against deterministic data-dictionary constraints and computes statistical health metrics.
#### 1. Per-Field Completeness Metric
For each target field $f \in \mathcal{F}_{17}$:
$$\text{Completeness}(f) = \begin{cases} 0.0 & \text{if } f \text{ is unmapped} \\ \frac{1}{N} \sum_{i=1}^N \mathbf{1}[v_{i, f} \notin \emptyset \wedge \text{norm}(v_{i, f}) \notin \text{Placeholders}] & \text{if } f \text{ is mapped} \end{cases}$$
#### 2. Composite Data Quality Score (DQ Score)
The overall data quality gauge $DQ \in [0, 100]$ combines completeness, row defect purity, and mapping confidence:
$$DQ = 100 \times \left( 0.50 \cdot \bar{C}_{\text{completeness}} + 0.30 \cdot \left(1 - \frac{|\mathcal{R}_{\text{defect}}|}{N}\right) + 0.20 \cdot \bar{C}_{\text{mapping}} \right)$$
Where:
- $\bar{C}_{\text{completeness}} = \frac{1}{17} \sum_{f \in \mathcal{F}_{17}} \text{Completeness}(f)$
- $|\mathcal{R}_{\text{defect}}|$ is the count of distinct rows containing substantive anomalies (excluding trivial whitespace or casing issues)
- $\bar{C}_{\text{mapping}}$ is the average confidence of mapped columns
#### 3. High-Confidence Auto-Approval Gate
The "Approve All $\ge 0.90$" operator evaluates the set of recommendations $\mathcal{R}$:
$$\text{AutoApprove}(r) \iff C(r) \ge 0.90 \wedge \text{Severity}(r) \ne \text{"High"}$$
High-severity defects (e.g. negative sums, future construction years, unparseable numeric types, duplicate primary references) are mathematically barred from autonomous sign-off.
---

### 2.4 Agent 4 — Controlled Transformation & Invariant Verification
Agent 4 performs deterministic schema projection and invariant enforcement without non-deterministic LLM mutation.
#### 1. Dimension & Type Invariance
Let $\mathbf{X}_{\text{source}} \in \mathbb{R}^{N \times K}$ be the input table. The transformation operator $\mathcal{T}$ guarantees:
$$\mathcal{T}(\mathbf{X}_{\text{source}}) = \mathbf{X}_{\text{clean}} \in \mathcal{D}^{M \times 17}$$
Where:
- Number of columns is strictly $17$ in canonical schema order.
- $M = N - |\mathcal{R}_{\text{dropped}}|$ (where totals rows and user-rejected rows are excised).
- Strict Type Projection:
  $$\forall i \in \{1, \dots, M\}, \quad X_{i, f} \in \mathcal{T}_f \cup \{\text{NaN}\}$$
  where $\mathcal{T}_f \in \{\text{Int64}, \text{Float64}, \text{String}\}$.
#### 2. Conservation of Truth (No Hallucination)
Values that fail strictly defined type casting after human review are safely converted to typed blanks with a logged audit entry:
$$v_{\text{output}} = \begin{cases} \text{Coerce}(v, \text{dtype}_f) & \text{if } \text{Valid}(v, \text{dtype}_f) \\ \text{NaN} & \text{if } v = \emptyset \lor \text{CastFailed}(v) \end{cases}$$
$$\text{Fabricated}(v) = \text{False}, \quad \forall v \in \mathbf{X}_{\text{clean}}$$
#### 3. Audit Log Bijection
Every atomic transformation is bijective to an immutable audit record:
$$\Delta_{i, f} \longleftrightarrow \langle \text{row}_i, \text{col}_{\text{src}}, \text{col}_{\text{target}}, v_{\text{before}}, v_{\text{after}}, \text{Rule}_{\text{ID}}, C, \text{Reviewer}, t_{\text{timestamp}} \rangle$$
---

## Contributors and Institutional Partners

Developed by **Team Astra** as part of **ADROSONIC Build Hackathon** in collaboration with **Adrosonic** and **Birla Institute of Technology, Mesra (BIT Mesra)**:

| Contributor | Profile and Role |
| :--- | :--- |
| **Aashish Ranjan** | Final year IMSc. QEDS Student ([@ranjanaashish](https://github.com/ranjanaashish)) |
| **Aastha Chhabra** | Final year IMSc. QEDS Student ([@aasthaaachhabra](https://github.com/aasthaaachhabra)) |
| **ADROSONIC Hackathon** | Hackathon Host ([@ADROSONICHackathon](https://github.com/ADROSONICHackathon)) |

---

<div align="center">
  <sub>Built for precision underwriting. MIT Licensed. © 2026 Team Astra.</sub>
</div>
