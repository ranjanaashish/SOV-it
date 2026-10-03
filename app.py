"""Streamlit human-approval interface for the Agentic SOV Cleansing system.

Run from this folder (so .streamlit/config.toml theme is picked up):
    streamlit run app.py
"""
from __future__ import annotations

from collections import OrderedDict

import pandas as pd
import streamlit as st

import ui_components as ui
from sov_agent import schema
from sov_agent.agents.data_quality import DROP, RULEBOOK
from sov_agent.agents.schema_mapping import compatibility, profile
from sov_agent.orchestrator import APPROVE_ALL_THRESHOLD, Orchestrator
from sov_agent.state import StageGateError
from sov_agent.text import to_display
from sov_agent.values import allowed_choices, coerce_value, norm_value

st.set_page_config(page_title="SOV-it — Agentic SOV Cleansing & Intelligence System", page_icon="🛡️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models and knowledge graph…")
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


orc = get_orchestrator()
ss = st.session_state
for k, v in {"state": None, "files": None, "upload_key": None, "flash": [], "open_rec": None}.items():
    ss.setdefault(k, v)

KEEP, BLANK = "(keep original)", "(blank)"


def flash(msg: str, icon: str = "✅") -> None:
    ss.flash.append((msg, icon))


def reviewer() -> str:
    return (ss.get("reviewer") or "").strip() or "reviewer"


# ---------------------------------------------------------------- callbacks (run before the next render)
def cb_accept(rid: str) -> None:
    ss.open_rec = None
    try:
        orc.decide(ss.state, rid, "approve", reviewer())
        ss.files = None
        flash("Accepted")
    except ValueError as exc:
        flash(str(exc), "⚠️")


def cb_reject(rid: str) -> None:
    note = (ss.get(f"note_{rid}") or "").strip()
    if not note:
        ss[f"note_err_{rid}"] = "Please write a short reason, e.g. “This is Contents, not Building Value”."
        ss.open_rec = rid
        return
    ss.pop(f"note_err_{rid}", None)
    try:
        before = {r.id for r in ss.state.recommendations}
        orc.decide(ss.state, rid, "reject", reviewer(), note=note)
        ss.files = None
        new = next((r for r in ss.state.recommendations if r.id not in before), None)
        if new is not None:
            ss.open_rec = new.id
            flash(f"Re-reasoned · {new.interpretation or new.title}", "🧠" if new.status == "pending" else "❓")
    except ValueError as exc:
        flash(str(exc), "⚠️")


def cb_edit_mapping(rid: str) -> None:
    choice = ss.get(f"tgt_{rid}")
    try:
        orc.decide(ss.state, rid, "edit", reviewer(), edit={"target": None if choice == "(exclude column)" else choice})
        ss.files = None
        ss[f"editing_{rid}"] = False
        flash(f"Mapping saved: {choice}")
    except ValueError as exc:
        flash(str(exc), "⚠️")


def cb_edit_values(rid: str, groups: list[tuple[str, str]]) -> None:
    vmap = {}
    for i, (key, display) in enumerate(groups):
        val = ss.get(f"val_{rid}_{i}")
        if val is None or val == KEEP or (isinstance(val, str) and val.strip() == display):
            continue
        vmap[display] = None if val == BLANK else val
    if not vmap:
        flash("Nothing changed — pick at least one new value.", "ℹ️")
        return
    try:
        orc.decide(ss.state, rid, "edit", reviewer(), edit={"value_map": vmap})
        ss.files = None
        ss[f"editing_{rid}"] = False
        ss.open_rec = rid
        flash("Edit saved: " + ", ".join(f"{k} → {v if v is not None else '(blank)'}" for k, v in vmap.items()))
    except ValueError as exc:
        flash(str(exc), "⚠️")


def cb_approve_all() -> None:
    n = orc.approve_all_high(ss.state, reviewer())
    ss.files = None
    flash(f"Approved {n} high-confidence item(s)")


def cb_reset_kg() -> None:
    res = orc.kg.reset(learned_only=ss.get("kg_scope", "Learned memory only") == "Learned memory only")
    ss["kg_confirm"] = False
    b, a = res["before"], res["after"]
    flash(f"Knowledge graph reset: {sum(b['nodes'].values())} → {sum(a['nodes'].values())} nodes, "
          f"{b['edges'].get('MAPS_TO', 0)} → 0 learned mappings, {b['vector_memory']} → 0 vectors", "🧹")


# ---------------------------------------------------------------- top header banner & pipeline
st.markdown(ui.header_banner(), unsafe_allow_html=True)

try:
    pipe = st.container(key="pipeline")
except TypeError:                     # Streamlit < 1.43 has no container keys (bar is then not sticky)
    pipe = st.container()
pipe_slot = pipe.empty()


def live_pipeline(s) -> None:
    pipe_slot.markdown(ui.pipeline_html(s), unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(ui.sidebar_header(), unsafe_allow_html=True)
    ss["reviewer"] = st.text_input("Reviewer name", value=ss.get("reviewer", ""), placeholder="e.g. A. Analyst")
    up = st.file_uploader("Upload SOV (.xlsx or .csv)", type=["xlsx", "xlsm", "csv", "xls"])
    if up is not None and ss.upload_key != (up.name, up.size):
        with st.spinner("Agents 1–3 analysing the file…"):
            ss.state = orc.intake(up.getvalue(), up.name, listener=live_pipeline)
            ss.state.set_listener(None)
            ss.files, ss.upload_key, ss.open_rec = None, (up.name, up.size), None
    st.divider()

    # -------------------------------------------------------- LLM Provider Settings
    with st.expander("⚙️ LLM & Provider Settings", expanded=False):
        presets = {
            "Groq (Fast Cloud)": {
                "base_url": "https://api.groq.com/openai/v1",
                "model": "llama-3.3-70b-versatile",
                "needs_key": True,
                "help": "Get a free key at console.groq.com. Extremely fast cloud inference (~500 t/s).",
            },
            "OpenAI (Cloud)": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "needs_key": True,
                "help": "Requires an OpenAI API key from platform.openai.com.",
            },
            "Google Gemini (OpenAI API)": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "model": "gemini-1.5-flash",
                "needs_key": True,
                "help": "Get an API key at aistudio.google.com.",
            },
            "OpenRouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "model": "meta-llama/llama-3.3-70b-instruct",
                "needs_key": True,
                "help": "Unified gateway for any open-source or commercial model.",
            },
            "Ollama (Local)": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen2.5:3b-instruct",
                "needs_key": False,
                "help": "Runs 100% on your machine. No API key needed.",
            },
            "Custom (Any Endpoint)": {
                "base_url": orc.llm.base_url,
                "model": orc.llm.model,
                "needs_key": True,
                "help": "Connect to any OpenAI-compatible server (vLLM, LocalAI, LM Studio, etc.).",
            },
            "Disabled (Rules-Only)": {
                "base_url": "",
                "model": "",
                "needs_key": False,
                "help": "Disable LLM completely; runs deterministic rules in ~1 second.",
            },
        }

        # Detect default provider from active settings
        curr_base = orc.llm.base_url.lower()
        if orc.llm.disabled:
            default_prov = "Disabled (Rules-Only)"
        elif "groq.com" in curr_base:
            default_prov = "Groq (Fast Cloud)"
        elif "api.openai.com" in curr_base:
            default_prov = "OpenAI (Cloud)"
        elif "generativelanguage.googleapis.com" in curr_base:
            default_prov = "Google Gemini (OpenAI API)"
        elif "openrouter.ai" in curr_base:
            default_prov = "OpenRouter"
        elif "11434" in curr_base or "ollama" in curr_base or "localhost" in curr_base:
            default_prov = "Ollama (Local)"
        else:
            default_prov = "Custom (Any Endpoint)"

        chosen_prov = st.selectbox("Provider", list(presets.keys()),
                                   index=list(presets.keys()).index(ss.get("llm_provider", default_prov)),
                                   key="cfg_provider")
        ss["llm_provider"] = chosen_prov
        pinfo = presets[chosen_prov]

        if chosen_prov == "Disabled (Rules-Only)":
            orc.llm.configure(disabled=True)
            st.caption("LLM disabled. Pipeline runs on deterministic rules.")
        else:
            if chosen_prov == "Custom (Any Endpoint)":
                cfg_url = st.text_input("Base URL", value=ss.get("cfg_base_url", orc.llm.base_url),
                                        placeholder="http://localhost:8000/v1")
                cfg_model = st.text_input("Model Name", value=ss.get("cfg_model", orc.llm.model),
                                          placeholder="e.g. mistral-7b-instruct")
            else:
                cfg_url = pinfo["base_url"]
                cfg_model = st.text_input("Model Name", value=ss.get("cfg_model", pinfo["model"]),
                                          help=pinfo["help"])

            if pinfo["needs_key"]:
                cfg_key = st.text_input("API Key", type="password",
                                        value=ss.get("cfg_key", orc.llm.api_key),
                                        placeholder="Paste API key here…", help=pinfo["help"])
            else:
                cfg_key = ""

            ss["cfg_base_url"] = cfg_url
            ss["cfg_model"] = cfg_model
            ss["cfg_key"] = cfg_key

            orc.llm.configure(base_url=cfg_url, model=cfg_model, api_key=cfg_key, disabled=False)

            send_samples = st.checkbox("Send cell samples to LLM",
                                       value=orc.agent2.send_samples,
                                       help="Uncheck to send only header names (never cell values) for privacy.")
            orc.agent2.send_samples = send_samples

            if st.button("🔌 Test Connection", use_container_width=True):
                with st.spinner("Pinging endpoint…"):
                    orc.llm._available = None
                    if orc.llm.available():
                        st.success("Connected successfully!")
                    else:
                        st.error("Connection failed. Check URL, model name, and API key.")

    llm_ok = orc.llm.available()
    st.markdown(f"**LLM** {'🟢 online' if llm_ok else '⚪ offline'}  \n`{orc.llm.label}`")
    if not llm_ok:
        if orc.llm.disabled:
            st.caption("LLM disabled → running in rules-only deterministic mode.")
        elif pinfo.get("needs_key") and not orc.llm.api_key:
            st.caption("Enter API key in ⚙️ LLM Settings above to connect.")
        else:
            st.caption("Rules-only mode: deterministic rationales, no LLM pass. Start Ollama to enable.")
    else:
        s = orc.llm.stats()
        st.caption(f"{s['calls']} calls · {s['cache_hits']} cache hits · avg {s['avg_seconds']}s · "
                   f"{orc.llm.max_workers} parallel workers")
    st.markdown(f"**Embeddings** `{orc.embedder.backend}`")
    kgs = orc.kg.stats()
    st.markdown(f"**Knowledge graph:** {sum(kgs['nodes'].values())} nodes · "
                f"{kgs['edges'].get('MAPS_TO', 0)} learned mappings · {kgs['vector_memory']} vectors")
    with st.popover("🧹 Reset knowledge graph", use_container_width=True):
        st.radio("What to reset", ["Learned memory only", "Everything (rebuild from seed)"], key="kg_scope",
                 help="Learned memory = approved/rejected mappings, co-occurrence edges and vectors. "
                      "The 17-field schema, synonyms, value sets and rules are always re-seeded.")
        st.checkbox(f"I understand this permanently deletes {kgs['edges'].get('MAPS_TO', 0)} learned mappings, "
                    f"{kgs['edges'].get('REJECTED_AS', 0)} rejections and {kgs['vector_memory']} vectors.",
                    key="kg_confirm")
        st.button("Reset now", type="primary", disabled=not ss.get("kg_confirm"), on_click=cb_reset_kg,
                  use_container_width=True)

for msg, icon in ss.flash:
    st.toast(msg, icon=icon)
ss.flash = []
state = ss.state

# ---------------------------------------------------------------- always-visible pipeline (re-drawn live)
pipe_slot.markdown(ui.pipeline_html(state), unsafe_allow_html=True)

if state is None:
    st.markdown(
        """
        <div class="card" style="margin-top:14px; border-left:4px solid var(--accent);">
            <h3 style="margin:0 0 8px 0; border-left:none; padding-left:0; color:#FFFFFF; font-size:18px;">
                Getting Started — Pipeline Overview
            </h3>
            <p style="color:var(--muted); font-size:14px; margin-bottom:14px; max-width:850px;">
                Upload a raw client Statement of Values (SOV) spreadsheet in the sidebar. 
                Four collaborating AI agents execute automated sheet discovery, schema mapping, and deep anomaly reasoning—with 
                <b>100% human-in-the-loop governance</b> and a strictly verified transformation gate.
            </p>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:12px; margin-top:10px;">
                <div class="kpi">
                    <b style="color:var(--accent);">1 · Sheets & Header</b>
                    <div style="font-size:12px; color:var(--muted); margin-top:4px;">Discovers active property sheet and header candidate rows with no assumed schema.</div>
                </div>
                <div class="kpi">
                    <b style="color:var(--accent);">2 · Mappers</b>
                    <div style="font-size:12px; color:var(--muted); margin-top:4px;">Hierarchical multi-pass matching: Graph memory → Curated synonyms → RapidFuzz → Semantic SBERT → LLM.</div>
                </div>
                <div class="kpi">
                    <b style="color:var(--accent);">3 · Quality Report</b>
                    <div style="font-size:12px; color:var(--muted); margin-top:4px;">20+ insurance validation rules with adaptive neural reasoning on rejections.</div>
                </div>
                <div class="kpi">
                    <b style="color:var(--accent);">4 · Export & Audit</b>
                    <div style="font-size:12px; color:var(--muted); margin-top:4px;">Strictly typed Cleaned_SOV.xlsx output with complete cell-level audit trail.</div>
                </div>
            </div>
            <div style="margin-top:16px; font-size:13px; color:#FFFFFF; font-weight:600; display:flex; align-items:center; gap:8px;">
                👈 Upload an SOV file (.xlsx, .csv) in the sidebar to begin
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if state.stages["ingest"].status == "error":
    st.error(f"⚠️ {state.stages['ingest'].message}")
    st.stop()

badge = {"pending": "🟡 pending", "approved": "🟢 approved", "edited": "🔵 edited",
         "rejected": "🔴 rejected", "escalated": "🟠 needs you", "superseded": "⚪ superseded"}
open_n = sum(1 for r in state.active_recs() if r.is_open)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sheet", state.selected_sheet or "—", f"header row {state.header_row}" if state.header_row else None,
          delta_color="off")
c2.metric("Rows", f"{state.quality.get('rows', 0):,}")
c3.metric("Mapping confidence", f"{state.mapping_json.get('overall_confidence', 0):.2f}")
c4.metric("Data quality score", state.quality.get("dq_score", "—"))
c5.metric("Open review items", open_n)

tabs = st.tabs([
    "📑 Sheets & header",
    "🧭 Mappers",
    "🩺 Quality report",
    "✅ Review",
    "👀 Before / after",
    "📦 Export & audit",
    "🕸️ Knowledge graph",
    "📜 Event log",
])
T_SHEETS, T_MAP, T_QUALITY, T_REVIEW, T_PREVIEW, T_EXPORT, T_KG, T_LOG = tabs

# ---------------------------------------------------------------- sheets
with T_SHEETS:
    st.markdown("### Agent 1 — ranked sheet manifest")
    man = pd.DataFrame([{"sheet": s.sheet_name, "class": s.classification, "confidence": s.confidence,
                         "header_row": s.header_row, "rows": s.n_rows, "cols": s.n_cols, "null_ratio": s.null_ratio,
                         "reasoning": " · ".join(s.reasoning)} for s in state.sheet_manifest])
    st.dataframe(man, use_container_width=True, hide_index=True,
                 column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    with st.expander("Manifest JSON"):
        st.json([s.model_dump() for s in state.sheet_manifest])
    st.markdown("**Override** — every later stage depends on this; re-runs Agents 1–3 and clears decisions.")
    names = [s.sheet_name for s in state.sheet_manifest]
    oc1, oc2, oc3 = st.columns([2, 1, 1])
    sel = oc1.selectbox("Data sheet", names, index=names.index(state.selected_sheet) if state.selected_sheet in names else 0)
    hdr = oc2.number_input("Header row", min_value=1, max_value=500, value=int(state.header_row or 1))
    oc3.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    if oc3.button("Re-run with override", use_container_width=True):
        ss.state = orc.analyse(state, sheet=sel, header_row=int(hdr))
        st.rerun()
    if sel in state.raw_sheets:
        st.caption("Raw grid preview (first 15 rows, no header assumed)")
        st.dataframe(state.raw_sheets[sel].head(15).astype(str).replace("None", ""), use_container_width=True)

if not state.stages["discovery"].validated:
    with T_SHEETS:
        st.warning(state.stages["discovery"].message or "Sheet discovery did not complete. See 'Sheets & header' above.")
    st.stop()

# ---------------------------------------------------------------- mapping
with T_MAP:
    st.markdown("### Agent 2 — schema mapping")
    rows = [{"source_column": m.source_column, "target": m.target, "confidence": m.confidence, "method": m.method,
             "flag": m.flag, "samples": ", ".join(m.sample_values[:3]), "rationale": m.rationale}
            for m in state.mappings]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    with st.expander("Mapping JSON", expanded=False):
        st.json(state.mapping_json)

# ---------------------------------------------------------------- quality report
with T_QUALITY:
    st.markdown("### Agent 3 — data quality report")
    g1, g2 = st.columns([1, 2])
    g1.markdown(ui.score_gauge(state.quality.get("dq_score", 0)), unsafe_allow_html=True)
    g2.markdown(ui.kpi_tiles(state), unsafe_allow_html=True)
    st.markdown(ui.score_interpretation(state), unsafe_allow_html=True)
    st.write("")

    q1, q2 = st.columns(2)
    with q1:
        st.markdown(ui.chart_header_html("Field Completeness Profile", "Share of rows populated vs. 95% target threshold", "17 Target Fields"), unsafe_allow_html=True)
        st.altair_chart(ui.completeness_chart(state), use_container_width=True)
        st.markdown(ui.completeness_interpretation(state), unsafe_allow_html=True)
    
    with q2:
        st.markdown(ui.chart_header_html("Issue Severity & Volume", "Detected validation flags and syntax anomalies by field", "Risk Ranked"), unsafe_allow_html=True)
        ic = ui.issues_chart(state)
        if ic is not None:
            st.altair_chart(ic, use_container_width=True)
        else:
            st.success("No data issues found.")
        st.markdown(ui.issues_interpretation(state), unsafe_allow_html=True)

    hm = ui.row_heatmap(state, RULEBOOK)
    if hm is not None:
        st.write("")
        st.markdown(ui.chart_header_html("Property Anomaly Hotspots", "Focused compound risk matrix across top flagged records", "Top 12 Records"), unsafe_allow_html=True)
        st.altair_chart(hm, use_container_width=True)
        st.markdown(ui.heatmap_interpretation(state), unsafe_allow_html=True)

    st.write("")
    st.markdown("### Granular Issue Registry")
    issue_df = pd.DataFrame([{"severity": i.severity, "issue": i.issue_type.replace("_", " "), "field": i.field,
                              "source column": i.source_column or "—", "rows": i.count,
                              "example rows": ", ".join(map(str, i.affected_rows[:10])) + (" …" if i.count > 10 else ""),
                              "examples": ", ".join(str(e.get("value")) for e in i.examples[:3])}
                             for i in sorted(state.issues, key=lambda i: ({"High": 0, "Medium": 1, "Low": 2}[i.severity], -i.count))])

    def _sev_style(v):
        c = ui.SEV_COLORS.get(v)
        return f"background-color:{c};color:{ui.BG};font-weight:700" if c else ""

    if not issue_df.empty:
        st.dataframe(issue_df.style.map(_sev_style, subset=["severity"]), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- review queue
def render_mapping_editor(r) -> None:
    opts = ["(exclude column)"] + schema.TARGET_FIELDS
    cur = r.proposed_target if r.proposed_target in schema.TARGET_FIELDS else "(exclude column)"
    choice = st.selectbox("Target field", opts, index=opts.index(cur), key=f"tgt_{r.id}")
    series = state.source_df[r.source_column]
    samples = [str(to_display(v)) for v in series.dropna().unique()[:6]]
    if choice != "(exclude column)":
        fit = compatibility(choice, profile(series))
        if fit is not None:
            cls = "live-ok" if fit >= 0.5 else "live-bad"
            st.markdown(f'<span class="{cls}">{fit:.0%}</span> of the values in <code>{r.source_column}</code> look like '
                        f'<b>{choice}</b>', unsafe_allow_html=True)
            st.progress(min(1.0, fit))
        taken = [s for s, t in state.effective_mapping(include_pending=False).items() if t == choice and s != r.source_column]
        if taken:
            st.markdown(f'<span class="live-bad">⚠ {choice} is already approved for “{taken[0]}”.</span>',
                        unsafe_allow_html=True)
    preview = pd.DataFrame({"source value": samples, "goes to": [choice] * len(samples)})
    st.dataframe(preview, hide_index=True, use_container_width=True)
    st.button("💾 Save mapping", key=f"savemap_{r.id}", type="primary", on_click=cb_edit_mapping, args=(r.id,))


def render_value_editor(r) -> None:
    col, field = r.source_column, r.field
    rows_ = [c.row for c in r.changes] or next((i.affected_rows for i in state.issues if i.id == r.issue_id), [])
    if not col or col not in state.source_df.columns or not rows_:
        st.caption("This item has no editable cell values.")
        return
    proposed = {c.row: c.after for c in r.changes}
    groups: OrderedDict[str, dict] = OrderedDict()
    for row in rows_:
        v = state.source_df.at[row, col]
        g = groups.setdefault(norm_value(v), {"display": str(to_display(v)) if v is not None else "", "rows": [],
                                              "proposed": proposed.get(row, v)})
        g["rows"].append(row)
    choices = allowed_choices(field)
    st.markdown(f"**Change each distinct value once — it applies to every row that holds it** "
                f"({len(rows_)} row(s), {len(groups)} distinct value(s)).")
    keys = list(groups.items())[:40]
    live: dict[str, object] = {}
    for i, (key, g) in enumerate(keys):
        a, b, c = st.columns([2.2, 2.0, 2.2])
        disp_text = html.escape(g['display'] or '(empty)')
        a.markdown(f"<div style='padding-top:6px; overflow:hidden; text-overflow:ellipsis;'><code>{disp_text}</code> &nbsp;·&nbsp; <span style='font-size:12px;color:var(--muted);'>{len(g['rows'])} row(s)</span></div>", unsafe_allow_html=True)
        prop = g["proposed"]
        if choices:
            opts = [KEEP, BLANK] + choices
            default = to_display(prop) if to_display(prop) in choices else (BLANK if prop is None and r.changes else KEEP)
            val = b.selectbox("new value", opts, index=opts.index(default), key=f"val_{r.id}_{i}",
                              label_visibility="collapsed")
        else:
            default = "" if prop is None or prop == DROP else str(to_display(prop))
            val = b.text_input("new value", value=default, key=f"val_{r.id}_{i}", label_visibility="collapsed",
                              placeholder="new value (empty = blank)")
        if val == KEEP or (isinstance(val, str) and val.strip() == g["display"]):
            live[key] = ("keep", None)
            c.markdown('<div style="padding-top:6px;"><span class="meta">unchanged</span></div>', unsafe_allow_html=True)
        else:
            out, err = coerce_value(field, None if val == BLANK else val)
            live[key] = ("err", err) if err else ("ok", out)
            if err:
                c.markdown(f'<div style="padding-top:6px;"><span class="live-bad" style="word-break:break-word;">⚠ {html.escape(str(err))}</span></div>', unsafe_allow_html=True)
            else:
                c.markdown(f'<div style="padding-top:6px;"><span class="live-ok">→ {html.escape("(blank)" if out is None else str(out))}</span></div>', unsafe_allow_html=True)
    # live preview of the affected rows
    prev = []
    for key, g in keys:
        kind, out = live[key]
        for row in g["rows"][:5]:
            prev.append({"row": row, "before": g["display"],
                         "after (live)": g["display"] if kind == "keep" else ("⚠ invalid" if kind == "err"
                                                                              else ("(blank)" if out is None else str(out)))})
    pdf = pd.DataFrame(prev[:25])

    def _hl(s):
        return [f"background-color: rgba(217,160,111,.28); font-weight:700" if (a != b and not str(a).startswith("⚠"))
                else ("background-color: rgba(244,114,182,.25)" if str(a).startswith("⚠") else "")
                for a, b in zip(pdf["after (live)"], pdf["before"])] if s.name == "after (live)" else [""] * len(s)

    st.caption("Live preview — updates as you change the values above")
    st.dataframe(pdf.style.apply(_hl), hide_index=True, use_container_width=True)
    bad = any(k == "err" for k, _ in live.values())
    st.button("💾 Save values", key=f"savev_{r.id}", type="primary", disabled=bad,
              on_click=cb_edit_values, args=(r.id, [(k, g["display"]) for k, g in keys]))


def render_rec(r) -> None:
    head = f"{badge[r.status]} · {r.title} · conf {r.confidence:.2f}" + (f" · rev {r.revision}" if r.revision else "")
    with st.expander(head, expanded=(r.status == "escalated" or ss.open_rec == r.id)):
        if r.question:
            st.markdown(f'<div class="question">❓ {r.question}</div>', unsafe_allow_html=True)
        if r.interpretation:
            st.markdown(f'<div class="understood">🧠 <b>Understood as:</b> {r.interpretation}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="why"><b>Why:</b> {r.rationale}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="unc"><b>Uncertainty:</b> {r.uncertainty or "—"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta">{ui.sev_pill(r.severity)} &nbsp; Raw column <code>{r.source_column or "—"}</code> '
                    f'· Target <code>{r.field or "—"}</code> · Method <code>{r.method}</code></div>',
                    unsafe_allow_html=True)
        if r.kg_evidence:
            st.caption("Knowledge graph: " + " | ".join(r.kg_evidence))
        if r.examples:
            st.dataframe(pd.DataFrame(r.examples).astype(str), hide_index=True, use_container_width=True)
        if len(r.changes) > 3:
            st.caption(f"{len(r.changes)} row-level changes proposed")
        if r.history:
            st.caption("History: " + " → ".join(f"{h.get('decision')}{' (“' + h['note'] + '”)' if h.get('note') else ''}"
                                               for h in r.history if h.get("decision")))
        if not r.is_open:
            return

        a1, a2 = st.columns([1, 2])
        with a1:
            st.button("✅ Accept", key=f"acc_{r.id}", type="primary", use_container_width=True,
                      on_click=cb_accept, args=(r.id,))
            st.write("")
            editing = st.toggle("✏️ Edit inline", key=f"editing_{r.id}")
        with a2:
            ph = ("e.g. “This is Contents, not Building Value” · “ignore this column”"
                  if r.action_type == "column_mapping" else
                  "e.g. “Partial means N, 100% is Y” · “keep as is” · “set to Y” · “remove rows” · “make positive”")
            with st.form(key=f"rejform_{r.id}", clear_on_submit=False, border=False):
                st.text_area("Reject with a note — the agent will re-reason", key=f"note_{r.id}",
                             placeholder=ph, height=80)
                st.form_submit_button("❌ Reject & re-reason", use_container_width=True,
                                      on_click=cb_reject, args=(r.id,))
            if ss.get(f"note_err_{r.id}"):
                st.error(ss[f"note_err_{r.id}"])
        if editing:
            st.markdown("---")
            if r.action_type == "column_mapping":
                render_mapping_editor(r)
            else:
                render_value_editor(r)


with T_REVIEW:
    if not ss.get("reviewer"):
        st.info("Enter your reviewer name in the sidebar — it is written to the audit log.")
    eligible = [r for r in state.active_recs() if r.status == "pending" and r.confidence >= APPROVE_ALL_THRESHOLD
                and not (r.action_type == "flag_for_review" and r.severity == "High")]
    b1, b2, b3 = st.columns([1.5, 1.6, 2.2])
    b1.button(f"Approve all ≥ {APPROVE_ALL_THRESHOLD:.2f} ({len(eligible)})", type="primary", disabled=not eligible,
              use_container_width=True, on_click=cb_approve_all)
    show = b2.radio("Show", ["Open", "All", "Decided"], horizontal=True, label_visibility="collapsed")
    b3.caption("High-severity flags (negative values, future years, duplicates) always need an individual decision. "
               "Rejections go back to the agent, which re-reasons from your note.")
    groups = {"column_mapping": "🧭 Column mappings", "data_correction": "🛠️ Data corrections",
              "standardisation": "📏 Standardisation", "flag_for_review": "🚩 Flags for review"}
    any_shown = False
    for g, title in groups.items():
        items = [r for r in state.active_recs() if r.action_type == g]
        if show == "Open":
            items = [r for r in items if r.is_open]
        elif show == "Decided":
            items = [r for r in items if not r.is_open]
        if not items:
            continue
        any_shown = True
        items.sort(key=lambda r: (r.status != "escalated", -r.confidence))
        st.markdown(f"### {title} ({len(items)})")
        for r in items:
            render_rec(r)
    if not any_shown:
        st.success("🎉 Nothing left to review. Go to **Export & audit** to apply the approved changes.")

# ---------------------------------------------------------------- before / after
with T_PREVIEW:
    st.markdown("### Before / after (first 30 rows · approved, edited and pending changes)")
    df = state.source_df
    mapping = state.effective_mapping(include_pending=True)
    before = df[[c for c in mapping if c in df.columns]].head(30).rename(columns=mapping)
    after = before.copy().astype(object)
    for r in state.active_recs():
        if r.action_type == "column_mapping" or r.status in ("rejected", "superseded") or r.field not in after.columns:
            continue
        for c in r.changes:
            if c.row in after.index and c.after != DROP:
                after.at[c.row, r.field] = c.after
    b_disp = before.map(lambda v: "" if v is None else str(to_display(v)))
    a_disp = after.map(lambda v: "" if v is None else str(to_display(v)))
    changed = b_disp != a_disp
    l, rr = st.columns(2)
    l.caption("Before (source values, target names)")
    l.dataframe(b_disp, use_container_width=True)
    rr.caption(f"After (proposed) — {int(changed.values.sum())} changed cell(s) highlighted")
    rr.dataframe(a_disp.style.apply(lambda _: changed.map(
        lambda x: "background-color: rgba(217,160,111,.30); font-weight:700" if x else ""), axis=None),
        use_container_width=True)

# ---------------------------------------------------------------- export
with T_EXPORT:
    st.markdown("### Agent 4 — controlled transformation")
    ok, reasons = orc.can_export(state)
    if not ok:
        st.error("Export blocked: " + "; ".join(reasons))
    if st.button("▶️ Apply approved transformations", type="primary", disabled=not ok):
        try:
            with st.spinner("Applying approved items…"):
                state.set_listener(live_pipeline)
                try:
                    ss.files = orc.transform(state)
                finally:
                    state.set_listener(None)
            st.success("Done — output passed schema validation.")
            st.rerun()
        except (StageGateError, RuntimeError) as exc:
            st.error(str(exc))
    if ss.files:
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("⬇️ Cleaned_SOV.xlsx", ss.files["Cleaned_SOV.xlsx"], "Cleaned_SOV.xlsx", use_container_width=True)
        d2.download_button("⬇️ Audit_Log.xlsx", ss.files["Audit_Log.xlsx"], "Audit_Log.xlsx", use_container_width=True)
        d3.download_button("⬇️ Audit_Log.json", ss.files["Audit_Log.json"], "Audit_Log.json", use_container_width=True)
        d4.download_button("⬇️ Summary.json", ss.files["Processing_Summary.json"], "Processing_Summary.json",
                           use_container_width=True)
        st.markdown(f"**Schema validation:** {state.summary.get('output_validation')}")
        st.dataframe(state.output_df.head(50), use_container_width=True)
        st.markdown("### Audit log")
        st.dataframe(pd.DataFrame([a.model_dump() for a in state.audit_log]).astype(str), use_container_width=True,
                     hide_index=True)
        with st.expander("Processing summary"):
            st.json({k: v for k, v in state.summary.items() if k != "explained_ids"})

# ---------------------------------------------------------------- knowledge graph
with T_KG:
    st.markdown("### Knowledge graph — memory across submissions")
    kgs = orc.kg.stats()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Nodes", sum(kgs["nodes"].values()))
    k2.metric("Learned mappings", kgs["edges"].get("MAPS_TO", 0))
    k3.metric("Remembered rejections", kgs["edges"].get("REJECTED_AS", 0))
    k4.metric("Vectors", kgs["vector_memory"])
    st.caption("Reset it from the sidebar (🧹 Reset knowledge graph).")
    learned = orc.kg.learned_edges()
    if learned:
        st.graphviz_chart(orc.kg.to_dot(), use_container_width=True)
        st.dataframe(pd.DataFrame(learned), use_container_width=True, hide_index=True)
    else:
        st.caption("No learned edges yet. They are written only after a reviewer signs off an export.")

# ---------------------------------------------------------------- event log
with T_LOG:
    st.markdown("### Event log (shared state, drives the pipeline bar)")
    for s_name, s in state.stages.items():
        if s.message:
            st.caption(f"**{s_name}** — {s.message}")
    st.dataframe(pd.DataFrame([e.model_dump() for e in state.events][::-1]), use_container_width=True, hide_index=True)
