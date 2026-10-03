"""Theme, live pipeline bar and data-quality charts for the Streamlit UI.

Palette:
  Background: #1a2c47 · Surface: #223859 · Accent: #7bdcb5 · Text: #FFFFFF · Muted: #B8C8DA
Font:
  Montserrat (weights 300, 400, 500, 600, 700, 800)
Status colours:
  High: #EF4444 · Medium: #F59E0B · Low: #10B981
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import altair as alt
import pandas as pd

from sov_agent import schema
from sov_agent.state import STAGES, SOVState

BG, SURFACE, SURFACE_2 = "#1a2c47", "#223859", "#142338"
PRIMARY, ACCENT, HIGHLIGHT, TEXT, MUTED = "#7bdcb5", "#7bdcb5", "#7bdcb5", "#FFFFFF", "#B8C8DA"
HIGH, MEDIUM, LOW = "#F87171", "#FBBF24", "#7BDCB5"
SEV_COLORS = {"High": HIGH, "Medium": MEDIUM, "Low": LOW}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Montserrat:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root {{
  --bg:{BG}; --surface:{SURFACE}; --surface-2:{SURFACE_2}; --primary:{PRIMARY}; --accent:{ACCENT};
  --highlight:{HIGHLIGHT}; --text:{TEXT}; --muted:{MUTED}; --high:{HIGH}; --med:{MEDIUM}; --low:{LOW};
  --line: rgba(123, 220, 181, 0.22);
}}

html, body, .stApp {{
  font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}

/* Typography: Dithered technical hierarchy */
p, label, input, textarea, select, .stMarkdown,
[data-testid="stMarkdownContainer"] p {{
  font-family: 'Montserrat', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}

h1, h2, h3, h4, h5, h6,
.stage .name, .chart-title {{
  font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  letter-spacing: -0.015em;
}}

/* Monospace tokens for precision labels, badges, pills, metrics */
code, .pill, .sev, .chart-badge, .agent, .meta,
[data-testid="stMetricLabel"] p, .kpi .l {{
  font-family: 'IBM Plex Mono', monospace !important;
}}

body, [data-testid="stMarkdownContainer"] > p {{
  font-size: 14px;
  line-height: 1.5;
  color: var(--text);
}}

/* CRITICAL: Explicitly preserve Streamlit Material Symbols and Icon fonts */
[data-testid="stIconMaterial"],
[class*="material-symbols"],
[class*="material-icons"],
[data-testid="stExpanderToggleIcon"],
[data-testid="stFileUploader"] span[data-testid="stIconMaterial"],
span[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-symbols-outlined {{
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
  font-weight: normal !important;
  font-style: normal !important;
  line-height: 1 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  display: inline-block !important;
  white-space: nowrap !important;
  word-wrap: normal !important;
  direction: ltr !important;
  -webkit-font-smoothing: antialiased !important;
}}

/* Dithered Dot-Pattern Texture on Viewport */
.stApp {{
  background-color: var(--bg) !important;
  background-image: 
    radial-gradient(circle at 1px 1px, rgba(123, 220, 181, 0.08) 1px, transparent 0),
    radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.03) 1px, transparent 0) !important;
  background-size: 8px 8px, 4px 4px !important;
  background-position: 0 0, 2px 2px !important;
  color: var(--text);
}}

.main .block-container {{
  padding-top: 1.5rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 100% !important;
}}

/* Sidebar with Dithered Stipple Grid */
[data-testid="stSidebar"] {{
  background-color: #142338 !important;
  background-image: radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.035) 1px, transparent 0) !important;
  background-size: 6px 6px !important;
  border-right: 1px solid rgba(123, 220, 181, 0.22) !important;
}}

h1 {{
  color: #FFFFFF !important;
  font-size: 24px !important;
  font-weight: 700 !important;
  margin-bottom: 0.5rem !important;
}}

h2 {{
  color: #FFFFFF !important;
  font-size: 20px !important;
  font-weight: 700 !important;
  margin-bottom: 0.5rem !important;
}}

h3 {{
  color: #FFFFFF !important;
  font-size: 16px !important;
  font-weight: 600 !important;
  border-left: 3px solid var(--accent);
  padding-left: 0.6rem;
  margin-top: 1rem !important;
  margin-bottom: 0.5rem !important;
}}

[data-testid="stMetric"] {{
  background-color: rgba(34, 56, 89, 0.90) !important;
  background-image: radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.025) 1px, transparent 0) !important;
  background-size: 4px 4px !important;
  border: 1px solid var(--line) !important;
  border-left: 3px solid var(--accent) !important;
  border-radius: 6px !important;
  padding: 10px 14px;
  min-width: 0;
  overflow: hidden;
}}

[data-testid="stMetricLabel"] {{
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

[data-testid="stMetricLabel"] p {{
  color: var(--muted) !important;
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin: 0 !important;
  padding: 0 !important;
}}

[data-testid="stMetricValue"] {{
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 22px !important;
  font-weight: 700 !important;
  color: #FFFFFF !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
}}

[data-testid="stMetricValue"] > div {{
  font-family: 'Space Grotesk', sans-serif !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
}}

[data-testid="stMetricDelta"] {{
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
  margin-top: 2px !important;
}}

[data-baseweb="tab-list"] {{
  gap: 4px;
  border-bottom: 1px solid var(--line);
  overflow-x: auto;
  flex-wrap: nowrap;
}}

[data-baseweb="tab"] {{
  background: rgba(255, 255, 255, 0.04);
  border-radius: 4px 4px 0 0;
  padding: 8px 14px;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 13px;
  font-weight: 600;
  color: var(--muted) !important;
  white-space: nowrap !important;
  border: 1px solid transparent !important;
  border-bottom: none !important;
  transition: all 0.15s ease;
}}

[data-baseweb="tab"][aria-selected="true"] {{
  background: var(--surface) !important;
  background-image: radial-gradient(circle at 1px 1px, rgba(123, 220, 181, 0.12) 1px, transparent 0) !important;
  background-size: 4px 4px !important;
  color: var(--accent) !important;
  font-weight: 700;
  border: 1px solid rgba(123, 220, 181, 0.3) !important;
  border-bottom: 2px solid var(--accent) !important;
}}

[data-baseweb="tab-highlight"] {{
  display: none !important;
}}

[data-testid="stExpander"] {{
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(34, 56, 89, 0.75);
}}

.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"], .stDownloadButton button {{
  background: var(--accent) !important;
  border: 1px solid #7bdcb5 !important;
  color: #1a2c47 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  letter-spacing: 0.02em !important;
  border-radius: 4px !important;
  box-shadow: 0 2px 8px rgba(123, 220, 181, 0.25) !important;
}}

.stButton button[kind="primary"]:hover, .stDownloadButton button:hover {{
  filter: brightness(1.1);
  box-shadow: 0 4px 12px rgba(123, 220, 181, 0.35);
}}

.stButton button[kind="secondary"] {{
  border: 1px solid rgba(255, 255, 255, 0.18) !important;
  color: var(--text) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  border-radius: 4px !important;
}}

.stButton button:hover {{
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}}

code {{
  color: var(--accent) !important;
  background: rgba(123, 220, 181, 0.12) !important;
  border: 1px solid rgba(123, 220, 181, 0.25) !important;
  padding: 1px 6px;
  border-radius: 4px;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 12px;
  word-break: break-word;
  overflow-wrap: anywhere;
}}

/* Agent Workflow Pipeline: Dithered styling */
.st-key-pipeline {{
  margin-bottom: 12px;
}}

.pipe {{
  display: flex;
  align-items: stretch;
  overflow-x: auto;
  padding: 10px 12px;
  border-radius: 8px;
  background-color: var(--surface);
  background-image: radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.025) 1px, transparent 0);
  background-size: 4px 4px;
  border: 1px solid rgba(123, 220, 181, 0.25);
  box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}}

.stage {{
  flex: 1 1 0;
  min-width: 140px;
  border-radius: 6px;
  padding: 9px 11px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  gap: 3px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  transition: all .2s ease;
  min-height: 84px;
}}

.stage .stage-head {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  margin-bottom: 2px;
}}

.stage .stage-tag {{
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}}

.stage .ico {{
  font-size: 14px;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.08);
  flex-shrink: 0;
}}

.stage .agent {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9.5px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .5px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.stage .pill {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.10);
  color: var(--text);
  white-space: nowrap;
  flex-shrink: 0;
  text-transform: uppercase;
}}

.stage .name {{
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 12.5px;
  color: var(--text);
  line-height: 1.3;
  margin-top: 1px;
  word-break: break-word;
}}

.stage .sub {{
  font-size: 10.5px;
  color: var(--muted);
  margin-top: 1px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.3;
}}

.stage.done {{
  border-color: var(--accent);
  background: rgba(123, 220, 181, 0.10);
}}

.stage.done .pill {{
  background: var(--accent);
  color: #1a2c47;
  font-weight: 700;
}}

.stage.done .ico {{
  background: rgba(123, 220, 181, 0.25);
  color: var(--accent);
}}

.stage.running {{
  border-color: var(--accent);
  background: rgba(123, 220, 181, 0.16);
}}

.stage.running .pill {{
  background: var(--accent);
  color: #1a2c47;
  font-weight: 700;
}}

.stage.blocked {{
  border-color: var(--med);
}}

.stage.blocked .pill {{
  background: var(--med);
  color: #1a2c47;
}}

.stage.error {{
  border-color: var(--high);
}}

.stage.error .pill {{
  background: var(--high);
  color: #FFFFFF;
}}

.stage.idle {{
  opacity: .6;
}}

.stage .loop {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: var(--accent);
  margin-top: 4px;
  font-weight: 600;
}}

.link {{
  flex: 0 0 20px;
  display: flex;
  align-items: center;
  padding: 0 3px;
}}

.link span {{
  display: block;
  height: 2px;
  width: 100%;
  border-radius: 1px;
  background: rgba(255, 255, 255, 0.15);
}}

.link.done span {{
  background-color: var(--accent);
  background-image: repeating-linear-gradient(90deg, var(--accent) 0, var(--accent) 2px, transparent 2px, transparent 4px);
}}

.link.active span {{
  background-color: var(--accent);
}}

.progress {{
  height: 5px;
  border-radius: 2px;
  background: rgba(255, 255, 255, 0.10);
  margin-top: 8px;
  overflow: hidden;
}}

.progress > div {{
  height: 100%;
  background-color: var(--accent);
  background-image: repeating-linear-gradient(45deg, rgba(26, 44, 71, 0.35) 0, rgba(26, 44, 71, 0.35) 2px, transparent 2px, transparent 5px);
  transition: width .4s ease;
}}

/* Cards & Badges: Dithered styling */
.card {{
  background-color: var(--surface);
  background-image: radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.025) 1px, transparent 0);
  background-size: 4px 4px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
}}

.kpis {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px;
}}

.kpi {{
  background-color: var(--surface);
  background-image: radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.025) 1px, transparent 0);
  background-size: 4px 4px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px 12px;
  min-width: 0;
  overflow: hidden;
}}

.kpi .v {{
  font-family: 'Space Grotesk', sans-serif;
  font-size: 22px;
  font-weight: 700;
  color: #FFFFFF;
  line-height: 1.2;
}}

.kpi .l {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  margin-top: 3px;
  line-height: 1.3;
  word-break: break-word;
}}

.dot {{
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
}}

.sev {{
  font-family: 'IBM Plex Mono', monospace;
  display: inline-block;
  font-size: 10px;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 4px;
  color: #1a2c47;
  text-transform: uppercase;
}}

.why {{
  border-left: 3px solid var(--accent);
  padding: 4px 10px;
  margin: 6px 0;
}}

.unc {{
  border-left: 3px solid var(--med);
  padding: 4px 10px;
  margin: 6px 0;
  color: var(--muted);
}}

.understood {{
  border-left: 3px solid var(--accent);
  padding: 6px 12px;
  margin: 8px 0;
  background: rgba(123, 220, 181, 0.08);
  border-radius: 0 6px 6px 0;
}}

.question {{
  border-left: 3px solid var(--high);
  padding: 6px 12px;
  margin: 8px 0;
  background: rgba(239, 68, 68, 0.10);
  border-radius: 0 6px 6px 0;
}}

.meta {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  color: var(--muted);
}}

.live-ok {{ color: var(--low); font-weight: 700; }}
.live-bad {{ color: var(--high); font-weight: 700; }}

div[data-testid="stRadio"] > div {{
  flex-wrap: nowrap !important;
  gap: 12px !important;
}}

div[data-testid="stRadio"] label {{
  white-space: nowrap !important;
}}

div[data-testid="stForm"] {{
  padding-bottom: 4px;
}}

textarea {{
  font-family: 'Montserrat', sans-serif !important;
  font-size: 13px !important;
  line-height: 1.4 !important;
}}

/* Quality Report Card Containers & Interpretations: Dithered styling */
.chart-card {{
  background-color: var(--surface);
  background-image: radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.025) 1px, transparent 0);
  background-size: 4px 4px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px 18px;
  margin-bottom: 14px;
}}

.chart-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  padding-bottom: 8px;
}}

.chart-title {{
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 15px;
  font-weight: 700;
  color: #FFFFFF;
  line-height: 1.25;
}}

.chart-subtitle {{
  font-size: 11.5px;
  color: var(--muted);
  margin-top: 3px;
}}

.chart-badge {{
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 10px;
  font-weight: 600;
  padding: 2px 7px;
  border-radius: 4px;
  background: rgba(123, 220, 181, 0.15);
  color: var(--accent);
  border: 1px solid rgba(123, 220, 181, 0.3);
  white-space: nowrap;
  text-transform: uppercase;
}}

.chart-interpretation {{
  background-color: rgba(123, 220, 181, 0.07);
  background-image: radial-gradient(circle at 1px 1px, rgba(123, 220, 181, 0.08) 1px, transparent 0);
  background-size: 4px 4px;
  border-left: 3px solid var(--accent);
  border-radius: 0 6px 6px 0;
  padding: 10px 14px;
  margin-top: 10px;
  font-size: 12.5px;
  color: #FFFFFF;
  line-height: 1.5;
}}

.chart-interpretation b {{
  color: #7bdcb5;
}}

.chart-interpretation code {{
  color: #7bdcb5 !important;
  background: rgba(123, 220, 181, 0.15) !important;
  font-size: 11px;
  padding: 1px 5px;
  font-family: 'IBM Plex Mono', monospace !important;
}}
</style>
"""

STAGE_META = {
    "ingest": ("01", "Upload", "Intake"),
    "discovery": ("02", "Sheets & header", "Agent 1"),
    "mapping": ("03", "Mappers", "Agent 2"),
    "quality": ("04", "Quality report", "Agent 3"),
    "review": ("05", "Review gate", "Human Review"),
    "transform": ("06", "Export & audit", "Agent 4"),
}
PILL = {"idle": "waiting", "running": "working", "done": "Done", "blocked": "blocked", "error": "error"}


def _load_b64(path: Path) -> str:
    if path.exists():
        mime = "image/png" if path.suffix == ".png" else ("image/svg+xml" if path.suffix == ".svg" else "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    return ""


def header_banner() -> str:
    root = Path(__file__).parent / "assets"
    adro_b64 = _load_b64(root / "adrosonic_linear.png")
    bit_b64 = _load_b64(root / "bit_logo.png")

    adro_img = (
        f'<img src="{adro_b64}" style="height:32px; max-width:140px; object-fit:contain; display:block;" alt="Adrosonic" />'
        if adro_b64 else ''
    )
    bit_img = (
        f'<img src="{bit_b64}" style="height:32px; max-width:140px; object-fit:contain; display:block;" alt="BIT Mesra" />'
        if bit_b64 else ''
    )

    return f"""
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px;
                padding:12px 20px; margin-bottom:12px; background-color:#223859;
                background-image: radial-gradient(circle at 1px 1px, rgba(123, 220, 181, 0.12) 1px, transparent 0);
                background-size: 6px 6px;
                border:1px solid rgba(123,220,181,0.28);
                border-radius:8px; box-shadow:0 4px 14px rgba(0,0,0,0.25);">
        <div style="min-width:260px;">
            <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                <span style="background:var(--accent); color:#1a2c47; font-family:'Space Grotesk',sans-serif; font-weight:800; font-size:16px; padding:2px 10px; border-radius:4px; letter-spacing:0.5px; box-shadow:0 2px 6px rgba(123,220,181,0.3);">SOV-it</span>
                <span style="margin:0; font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:700; color:#FFFFFF; letter-spacing:-0.02em; line-height:1.25;">
                    Agentic SOV Cleansing and Intelligence System
                </span>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; font-weight:600; color:#7bdcb5; margin-top:4px; letter-spacing:0.04em;">
                BY TEAM ASTRA
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:12px; background:rgba(26,44,71,0.65); padding:6px 14px; border-radius:6px; border:1px solid rgba(255,255,255,0.12); flex-shrink:0;">
            {adro_img}
            <div style="width:1px; height:22px; background:rgba(255,255,255,0.2);"></div>
            {bit_img}
        </div>
    </div>
    """


def sidebar_header() -> str:
    return """
    <div style="margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid rgba(123,220,181,0.2);">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:13px; font-weight:700; color:#7bdcb5; text-transform:uppercase; letter-spacing:0.06em;">
            Intake & Controls
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#B8C8DA; margin-top:3px;">
            Upload workbook to run multi-agent pipeline
        </div>
    </div>
    """


def _stage_sub(stage: str, state: SOVState | None) -> str:
    if state is None:
        return "—"
    try:
        if stage == "ingest":
            return f"{len(state.raw_sheets)} sheet(s) · {state.file_name}" if state.raw_sheets else state.file_name
        if stage == "discovery":
            return f"{state.selected_sheet} · row {state.header_row}" if state.selected_sheet else "—"
        if stage == "mapping":
            if not state.mappings:
                return "—"
            mapped = sum(1 for m in state.mappings if m.target)
            return f"{mapped}/{len(state.mappings)} mapped"
        if stage == "quality":
            if not state.quality:
                return "—"
            return f"{len(state.issues)} issues · DQ {state.quality.get('dq_score')}"
        if stage == "review":
            recs = state.active_recs()
            if not recs:
                return "—"
            open_n = sum(1 for r in recs if r.is_open)
            return f"{open_n} open · {len(recs) - open_n} decided"
        if stage == "transform":
            if state.output_df is None:
                return "—"
            return f"{len(state.output_df)} rows · {len(state.audit_log)} audits"
    except Exception:
        return "—"
    return "—"


def pipeline_html(state: SOVState | None) -> str:
    parts = ['<div class="pipe">']
    done = 0
    for i, s in enumerate(STAGES):
        status = state.stages[s].status if state is not None else "idle"
        if state is not None and s == "review" and status == "running":
            pill = "action needed"
        else:
            pill = PILL[status]
        done += status == "done"
        ico, name, agent = STAGE_META[s]
        loop = ""
        if state is not None and s == "review":
            n = sum(1 for r in state.recommendations if r.revision > 0)
            if n:
                loop = f'<div class="loop">Re-reasoned: {n}</div>'
        msg = html.escape(state.stages[s].message) if state is not None else ""
        sub_text = html.escape(_stage_sub(s, state))
        parts.append(
            f'<div class="stage {status}" title="{msg}">'
            f'<div class="stage-head">'
            f'<div class="stage-tag"><div class="ico">{ico}</div><div class="agent">{agent}</div></div>'
            f'<span class="pill">{html.escape(pill)}</span>'
            f'</div>'
            f'<div class="name">{name}</div>'
            f'<div class="sub" title="{sub_text}">{sub_text}</div>'
            f'{loop}'
            f'</div>')
        if i < len(STAGES) - 1:
            nxt = state.stages[STAGES[i + 1]].status if state is not None else "idle"
            cls = "done" if status == "done" and nxt == "done" else ("active" if status == "done" else "")
            parts.append(f'<div class="link {cls}"><span></span></div>')
    parts.append("</div>")
    pct = int(100 * done / len(STAGES))
    parts.append(f'<div class="progress"><div style="width:{pct}%"></div></div>')
    return "".join(parts)


# ---------------------------------------------------------------------- data-quality visuals
def _theme(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(background="transparent", font="IBM Plex Mono")
            .configure_title(font="Space Grotesk", fontSize=13, color="#FFFFFF")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#FFFFFF", titleColor=MUTED, gridColor="rgba(255,255,255,0.07)",
                            gridDash=[2, 2],
                            domainColor="rgba(123,220,181,0.25)", tickColor="rgba(123,220,181,0.25)",
                            labelFont="IBM Plex Mono", titleFont="Space Grotesk",
                            labelFontSize=11, titleFontSize=12, labelLimit=220)
            .configure_legend(labelColor="#FFFFFF", titleColor=MUTED, orient="top", symbolType="square",
                              labelFont="IBM Plex Mono", titleFont="Space Grotesk",
                              labelLimit=200, labelFontSize=11, titleFontSize=11))


def chart_header_html(title: str, subtitle: str, badge: str = "") -> str:
    badge_html = f'<span class="chart-badge">{html.escape(badge)}</span>' if badge else ''
    return f"""
    <div class="chart-header">
        <div>
            <div class="chart-title">{html.escape(title)}</div>
            <div class="chart-subtitle">{html.escape(subtitle)}</div>
        </div>
        {badge_html}
    </div>
    """


def score_gauge(score: float) -> str:
    score = float(score or 0)
    color, label = (LOW, "Good") if score >= 80 else ((MEDIUM, "Needs attention") if score >= 60 else (HIGH, "Poor"))
    r, c = 54, 2 * 3.14159 * 54
    dash = c * max(0.0, min(1.0, score / 100))
    return (f'<div class="card" style="display:flex;align-items:center;gap:16px;height:100%">'
            f'<svg width="120" height="120" viewBox="0 0 132 132" style="flex-shrink:0;" role="img" aria-label="Data quality score {score:.0f} of 100">'
            f'<circle cx="66" cy="66" r="{r}" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="12"/>'
            f'<circle cx="66" cy="66" r="{r}" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" '
            f'stroke-dasharray="{dash:.1f} {c:.1f}" transform="rotate(-90 66 66)"/>'
            f'<text x="66" y="64" text-anchor="middle" font-family="Space Grotesk" font-size="28" font-weight="700" fill="#FFFFFF">{score:.0f}</text>'
            f'<text x="66" y="84" text-anchor="middle" font-family="IBM Plex Mono" font-size="11" fill="{MUTED}">of 100</text></svg>'
            f'<div style="min-width:0;">'
            f'<div style="font-size:11px;color:{MUTED};text-transform:uppercase;letter-spacing:.5px;font-weight:600;">Data quality score</div>'
            f'<div style="font-size:18px;font-weight:700;color:#FFFFFF;margin-top:2px;">{label}</div>'
            f'<div style="font-size:11.5px;color:{MUTED};margin-top:4px;line-height:1.4;">50% completeness · 30% clean rows · 20% mapping confidence</div>'
            f'</div></div>')


def score_interpretation(state: SOVState) -> str:
    score = float(state.quality.get("dq_score", 0))
    q = state.quality
    label = "Strong Underwriting Quality" if score >= 80 else ("Needs Attention" if score >= 60 else "Critical Risk Items")
    color = "#7bdcb5" if score >= 80 else ("#fbbf24" if score >= 60 else "#f87171")
    rows = q.get('rows', 0)
    flagged = q.get('rows_flagged', 0)
    flagged_pct = flagged / max(1, rows)
    clean_pct = 1 - flagged_pct
    
    return f"""
    <div class="chart-interpretation" style="border-left-color:{color};">
        <span style="font-weight:700; color:{color};">Executive Health Summary ({score:.0f}/100 · {label}):</span>
        The dataset contains <b>{rows:,} property rows</b>, with <b>{clean_pct:.0%} passing all validation checks cleanly</b>. 
        <b>{flagged_pct:.0%} of rows ({flagged:,} properties)</b> trigger one or more anomaly flags. 
        The composite score weights 50% field completeness, 30% row-level correctness, and 20% schema mapping confidence.
    </div>
    """


def kpi_tiles(state: SOVState) -> str:
    q = state.quality
    sev = {"High": 0, "Medium": 0, "Low": 0}
    for i in state.issues:
        if i.issue_type != "missing_field":
            sev[i.severity] += 1
    full = sum(1 for v in q["completeness"].values() if v >= 0.95)
    tiles = [
        (f"{q['rows']:,}", "Data rows"),
        (f"{q['rows_flagged']:,}", f"Rows flagged ({q['rows_flagged'] / max(1, q['rows']):.0%})"),
        (f"{full}/17", "Fields ≥ 95% complete"),
    ]
    out = ['<div class="kpis">']
    for v, l in tiles:
        out.append(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>')
    for k, col in SEV_COLORS.items():
        out.append(f'<div class="kpi"><div class="v">{sev[k]}</div>'
                   f'<div class="l"><span class="dot" style="background:{col}"></span>{k}-severity issues</div></div>')
    out.append("</div>")
    return "".join(out)


def completeness_chart(state: SOVState) -> alt.Chart:
    mapping = {t: s for s, t in state.effective_mapping(include_pending=True).items()}
    rows = []
    for f, v in state.quality["completeness"].items():
        band = "Complete (≥95%)" if v >= 0.95 else ("Partial (70–95%)" if v >= 0.70 else "Gap (<70%)")
        rows.append({"field": f, "pct": v, "band": band, "label": f"{v:.0%}",
                     "source": mapping.get(f, "(no source column)")})
    df = pd.DataFrame(rows)
    domain = ["Complete (≥95%)", "Partial (70–95%)", "Gap (<70%)"]
    
    # Sort by completeness percentage descending for clean, intuitive progression
    base = alt.Chart(df).encode(
        y=alt.Y("field:N", sort=alt.EncodingSortField(field="pct", order="descending"), title=None,
                axis=alt.Axis(labelColor="#FFFFFF", labelFontSize=11, labelLimit=180))
    )
    bars = base.mark_bar(cornerRadiusEnd=5, height=14).encode(
        x=alt.X("pct:Q", scale=alt.Scale(domain=[0, 1.15]), axis=alt.Axis(format="%", values=[0, .25, .5, .75, 1]),
                title="Populated Rows (%)"),
        color=alt.Color("band:N", scale=alt.Scale(domain=domain, range=[LOW, MEDIUM, HIGH]),
                        legend=alt.Legend(title=None, orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip("field:N", title="Field"), alt.Tooltip("source:N", title="Source Column"),
                 alt.Tooltip("pct:Q", title="Completeness", format=".1%"), alt.Tooltip("band:N", title="Status")]
    )
    labels = base.mark_text(align="left", dx=6, fontSize=11, font="IBM Plex Mono", fontWeight="bold").encode(
        x="pct:Q", text="label:N", color=alt.value(TEXT)
    )
    rule = alt.Chart(pd.DataFrame({"x": [0.95]})).mark_rule(
        strokeDash=[4, 4], color=PRIMARY, strokeWidth=1.5, opacity=0.75
    ).encode(x="x:Q")
    
    return _theme((bars + labels + rule).properties(height=17 * 22))


def completeness_interpretation(state: SOVState) -> str:
    comp = state.quality.get("completeness", {})
    full = [f for f, v in comp.items() if v >= 0.95]
    partial = [f for f, v in comp.items() if 0.70 <= v < 0.95]
    gaps = [f for f, v in comp.items() if v < 0.70]
    
    parts = [f"<b>{len(full)} of 17 schema fields</b> meet or exceed the target completeness threshold (≥95%)."]
    if partial:
        p_names = ", ".join(f"<code>{p}</code> ({comp[p]:.0%})" for p in partial[:4])
        parts.append(f"Partial coverage observed in {p_names}.")
    if gaps:
        g_names = ", ".join(f"<code>{g}</code> ({comp[g]:.0%})" for g in gaps[:3])
        parts.append(f"<b>Critical gaps (&lt;70%):</b> {g_names} — consider manual column mapping in the Mappers tab.")
    else:
        parts.append("No critical missing fields (&lt;70%) detected across primary risk attributes.")
        
    return f"""
    <div class="chart-interpretation">
        <div style="font-weight:700; color:#7bdcb5; margin-bottom:4px;">Completeness Interpretation</div>
        {' '.join(parts)}
    </div>
    """


def issues_chart(state: SOVState) -> alt.Chart | None:
    rows = [{"field": i.field if i.field != "(row)" else "Row-level", "severity": i.severity,
             "issue": i.issue_type.replace("_", " "), "rows": max(1, i.count)}
            for i in state.issues if i.issue_type != "missing_field"]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    order = df.groupby("field")["rows"].sum().sort_values(ascending=False).index.tolist()
    
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=14, stroke=BG, strokeWidth=1).encode(
        y=alt.Y("field:N", sort=order, title=None, axis=alt.Axis(labelColor="#FFFFFF", labelFontSize=11, labelLimit=180)),
        x=alt.X("sum(rows):Q", title="Flagged Rows"),
        color=alt.Color("severity:N", scale=alt.Scale(domain=["High", "Medium", "Low"], range=[HIGH, MEDIUM, LOW]),
                        legend=alt.Legend(title=None, orient="top", direction="horizontal")),
        order=alt.Order("severity:N", sort="ascending"),
        tooltip=[alt.Tooltip("field:N", title="Field"), alt.Tooltip("issue:N", title="Rule Violated"),
                 alt.Tooltip("severity:N", title="Severity"), alt.Tooltip("rows:Q", title="Affected Rows")]
    )
    return _theme(chart.properties(height=max(140, 24 * len(order))))


def issues_interpretation(state: SOVState) -> str:
    issues = [i for i in state.issues if i.issue_type != "missing_field"]
    if not issues:
        return """
        <div class="chart-interpretation">
            <div style="font-weight:700; color:#7bdcb5; margin-bottom:4px;">Issue Distribution Interpretation</div>
            Zero rule violations or syntax anomalies detected. The dataset conforms to all underwriting validation checks.
        </div>
        """
    sev_counts = {"High": 0, "Medium": 0, "Low": 0}
    for i in issues:
        sev_counts[i.severity] += 1
    
    field_counts = {}
    for i in issues:
        f = i.field if i.field != "(row)" else "Row-level"
        field_counts[f] = field_counts.get(f, 0) + i.count
    top_field = max(field_counts.items(), key=lambda x: x[1])[0] if field_counts else "—"
    
    parts = [f"Detected <b>{len(issues)} distinct issue types</b> across {len(field_counts)} fields."]
    if sev_counts["High"] > 0:
        parts.append(f"<b>{sev_counts['High']} High-severity issue(s)</b> require explicit human sign-off before export.")
    if top_field != "—":
        parts.append(f"Highest flag frequency occurs in <code>{top_field}</code> ({field_counts[top_field]} affected rows), predominantly resolvable via bulk approval.")
        
    return f"""
    <div class="chart-interpretation">
        <div style="font-weight:700; color:#7bdcb5; margin-bottom:4px;">Issue Distribution Interpretation</div>
        {' '.join(parts)}
    </div>
    """


def row_heatmap(state: SOVState, rulebook: dict, max_rows: int = 12) -> alt.Chart | None:
    rank = {"High": 3, "Medium": 2, "Low": 1}
    sev_of = {k: v[1] for k, v in rulebook.items()}
    cells: dict[tuple[int, str], dict] = {}
    row_severity_weight = {}
    for i in state.issues:
        if i.issue_type in ("missing_field",):
            continue
        field = i.field if i.field != "(row)" else "Row-level"
        sev = sev_of.get(i.issue_type, i.severity)
        weight = rank.get(sev, 1)
        for r in i.affected_rows:
            key = (r, field)
            row_severity_weight[r] = row_severity_weight.get(r, 0) + weight
            cur = cells.get(key)
            if cur is None or rank[sev] > rank[cur["severity"]]:
                cells[key] = {"row": f"Row {r}", "row_num": r, "field": field, "severity": sev,
                              "issues": i.issue_type.replace("_", " ") if cur is None
                              else cur["issues"] + ", " + i.issue_type.replace("_", " ")}
            else:
                cur["issues"] += ", " + i.issue_type.replace("_", " ")
    if not cells:
        return None
    df = pd.DataFrame(cells.values())
    top_rows = sorted(row_severity_weight.keys(), key=lambda r: row_severity_weight[r], reverse=True)[:max_rows]
    df = df[df["row_num"].isin(top_rows)]
    
    row_order = [f"Row {r}" for r in sorted(top_rows)]
    fields = [f for f in schema.TARGET_FIELDS + ["Row-level"] if f in set(df["field"])]
    
    chart = alt.Chart(df).mark_rect(cornerRadius=4, stroke=SURFACE, strokeWidth=2).encode(
        x=alt.X("field:N", sort=fields, title=None, axis=alt.Axis(labelAngle=0, orient="top", labelFontSize=11, labelColor="#FFFFFF")),
        y=alt.Y("row:O", sort=row_order, title="Excel Row", axis=alt.Axis(labelFontSize=11, labelColor="#FFFFFF")),
        color=alt.Color("severity:N", scale=alt.Scale(domain=["High", "Medium", "Low"], range=[HIGH, MEDIUM, LOW]),
                        legend=alt.Legend(title=None, orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip("row:O", title="Property"), alt.Tooltip("field:N", title="Field"),
                 alt.Tooltip("severity:N", title="Worst Severity"), alt.Tooltip("issues:N", title="Detected Issues")]
    )
    return _theme(chart.properties(height=max(130, 20 * len(top_rows))))


def heatmap_interpretation(state: SOVState) -> str:
    issues = [i for i in state.issues if i.issue_type != "missing_field"]
    if not issues:
        return ""
    
    row_counts = {}
    for i in issues:
        for r in i.affected_rows:
            row_counts[r] = row_counts.get(r, 0) + 1
    
    multi = sum(1 for c in row_counts.values() if c > 1)
    
    parts = [f"Focusing on the top {min(12, len(row_counts))} anomalous property records."]
    if multi > 0:
        worst_row = max(row_counts.items(), key=lambda x: x[1])[0]
        worst_count = row_counts[worst_row]
        parts.append(f"<b>{multi} row(s)</b> exhibit compound multi-field errors (e.g., Row {worst_row} triggers {worst_count} flags simultaneously).")
    else:
        parts.append("Anomalies are isolated single-field errors rather than compound multi-attribute corruption.")
    parts.append("Prioritize compound rows in the Review tab to eliminate systemic risk early.")
    
    return f"""
    <div class="chart-interpretation">
        <div style="font-weight:700; color:#7bdcb5; margin-bottom:4px;">Hotspot Matrix Interpretation</div>
        {' '.join(parts)}
    </div>
    """


def sev_pill(sev: str) -> str:
    color = SEV_COLORS.get(sev, MUTED)
    return f'<span class="sev" style="background:{color}; color:#1a2c47;">{sev}</span>'

