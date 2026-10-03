"""Agent 1 - Sheet Intelligence & Discovery.

Scores every sheet, detects the header row (any position, first 30 rows),
ranks sheets as Primary / Secondary / Reject with reasoning, and builds the
structured DataFrame of the chosen sheet. Deterministic: no LLM needed.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .. import schema
from ..io_utils import merged_spans
from ..state import SheetInfo, SOVState
from ..text import best_match, is_number

NAME = "Agent 1 · Sheet Discovery"
REJECT_NAME_HINTS = re.compile(r"summary|cover|note|instruction|readme|read me|info|glossary|lookup|"
                               r"dropdown|list|legend|definition|contact|change log|version", re.I)
TOTAL_ROW = re.compile(r"^\s*(grand\s+)?(sub\s*)?total|^\s*sum\b", re.I)
HEADER_SCAN_ROWS = 30


class SheetDiscoveryAgent:
    name = NAME

    def __init__(self, kg=None):
        self.kg = kg
        self.alias = schema.alias_index()

    # ------------------------------------------------------------------ public
    def run(self, state: SOVState, sheet_override: Optional[str] = None,
            header_override: Optional[int] = None) -> SOVState:
        state.require("ingest")
        state.start("discovery", NAME)
        manifest = [self._score_sheet(n, df, state.merged_ranges.get(n, []))
                    for n, df in state.raw_sheets.items()]
        manifest.sort(key=lambda s: -s.confidence)
        self._classify(manifest)
        state.sheet_manifest = manifest

        chosen = sheet_override or next((s.sheet_name for s in manifest if s.classification == "Primary"), None)
        if chosen is None:
            state.fail("discovery", NAME, "No sheet looks like a property schedule. "
                                          "Choose a sheet manually in 'Sheets & header'.")
            return state
        info = next(s for s in manifest if s.sheet_name == chosen)
        header_row = header_override or info.header_row
        if sheet_override:
            state.log(NAME, "override", f"Reviewer selected sheet '{chosen}'")
        if header_override:
            state.log(NAME, "override", f"Reviewer set header row {header_override}")
        if header_row is None:
            state.fail("discovery", NAME, f"Could not detect a header row in '{chosen}'.")
            return state

        df, total_rows = self._build_frame(state.raw_sheets[chosen], header_row - 1,
                                           state.merged_ranges.get(chosen, []))
        state.selected_sheet, state.header_row, state.source_df = chosen, header_row, df
        state.summary["total_rows_detected"] = total_rows

        ok = df is not None and len(df.columns) >= 2 and len(df) >= 1
        msg = (f"Selected '{chosen}' (confidence {info.confidence:.2f}), header row {header_row}, "
               f"{len(df)} data rows x {len(df.columns)} columns") if ok else \
              f"Sheet '{chosen}' has no usable data under header row {header_row}."
        state.finish("discovery", NAME, ok, msg)
        return state

    # ------------------------------------------------------------------ scoring
    def _header_vocab(self, values: list) -> float:
        if not values:
            return 0.0
        hits = 0
        for v in values:
            n = schema.normalize_header(v)
            if n in self.alias or n in schema.NON_TARGET_HEADERS:
                hits += 1
            elif best_match(n, self.alias.keys())[1] >= 0.85:
                hits += 1
        return hits / len(values)

    def detect_header(self, df: pd.DataFrame) -> list[dict]:
        cands = []
        if df.empty:
            return cands
        width = max(1, int(df.notna().any(axis=0).sum()))
        for r in range(min(HEADER_SCAN_ROWS, len(df))):
            vals = [v for v in df.iloc[r].tolist() if v is not None]
            if len(vals) < 2:
                continue
            text = [v for v in vals if isinstance(v, str) and not is_number(v)]
            text_ratio = len(text) / len(vals)
            fill = len(vals) / width
            uniq = len({str(v).strip().lower() for v in vals}) / len(vals)
            vocab = self._header_vocab(text[:40])
            below = df.iloc[r + 1:r + 11]
            if below.empty:
                density = 0.0
            else:
                cols = [c for c, v in zip(df.columns, df.iloc[r].tolist()) if v is not None]
                density = float(below[cols].notna().mean(axis=1).mean())
            score = 0.25 * text_ratio + 0.15 * fill + 0.30 * vocab + 0.20 * density + 0.10 * uniq
            cands.append({"row": r + 1, "score": round(score, 3), "text_ratio": round(text_ratio, 2),
                          "fill": round(fill, 2), "vocab": round(vocab, 2), "density_below": round(density, 2)})
        cands.sort(key=lambda c: -c["score"])
        return cands[:5]

    def _score_sheet(self, name: str, df: pd.DataFrame, merged: list[str]) -> SheetInfo:
        if df is None or df.empty:
            return SheetInfo(sheet_name=name, classification="Reject", confidence=0.0,
                             reasoning=["Sheet is empty"])
        cands = self.detect_header(df)
        if not cands:
            return SheetInfo(sheet_name=name, classification="Reject", confidence=0.05,
                             n_rows=len(df), n_cols=df.shape[1],
                             reasoning=["No row with at least two labelled columns in the first 30 rows"])
        best = cands[0]
        hr0 = best["row"] - 1
        header_cols = [c for c, v in zip(df.columns, df.iloc[hr0].tolist()) if v is not None]
        body = df.iloc[hr0 + 1:][header_cols].dropna(how="all")
        n_rows, n_cols = len(body), len(header_cols)
        null_ratio = float(body.isna().mean().mean()) if n_rows else 1.0

        reasons = []
        score = (0.35 * best["score"] + 0.20 * (1 - null_ratio) + 0.20 * min(n_rows / 20, 1)
                 + 0.15 * min(n_cols / 8, 1) + 0.10 * best["vocab"])
        reasons.append(f"Header detected on row {best['row']} (score {best['score']:.2f}, "
                       f"{int(best['vocab'] * 100)}% insurance vocabulary)")
        reasons.append("Consistent tabular structure" if best["density_below"] > 0.6
                       else "Sparse rows under the header")
        reasons.append(f"{n_rows} data rows x {n_cols} columns, null ratio {null_ratio:.0%}")
        if REJECT_NAME_HINTS.search(name):
            score -= 0.25
            reasons.append("Sheet name suggests notes/summary/reference content")
        if n_rows < 3:
            score -= 0.2
            reasons.append("Fewer than 3 data rows")
        if merged:
            reasons.append(f"{len(merged)} merged cell range(s) found")
        return SheetInfo(sheet_name=name, classification="Reject", confidence=round(max(0.0, min(1.0, score)), 2),
                         header_row=best["row"], n_rows=n_rows, n_cols=n_cols,
                         null_ratio=round(null_ratio, 3), reasoning=reasons, header_candidates=cands)

    @staticmethod
    def _classify(manifest: list[SheetInfo]) -> None:
        primary_set = False
        for s in manifest:  # already sorted by confidence
            if s.confidence >= 0.45 and not primary_set:
                s.classification, primary_set = "Primary", True
                s.reasoning.append("Highest-ranked tabular sheet: selected as authoritative data source")
            elif s.confidence >= 0.45:
                s.classification = "Secondary"
                s.reasoning.append("Tabular but ranked below the Primary sheet (supporting data)")
            elif s.confidence >= 0.25:
                s.classification = "Secondary"
                s.reasoning.append("Some structure, low relevance")
            else:
                s.classification = "Reject"
                s.reasoning.append("Low structural relevance (metadata, notes or summary)")

    # ------------------------------------------------------------------ frame build
    @staticmethod
    def _build_frame(raw: pd.DataFrame, hr0: int, merged: list[str]):
        header = list(raw.iloc[hr0].tolist())
        above = list(raw.iloc[hr0 - 1].tolist()) if hr0 > 0 else [None] * len(header)
        # Labels spanning merged header cells are copied to every column in the span.
        for a, b in merged_spans(merged, hr0):
            for c in range(a + 1, min(b, len(header) - 1) + 1):
                if header[c] is None:
                    header[c] = header[a]
        names, seen = [], {}
        for i, h in enumerate(header):
            if h is None and above[i] is not None and isinstance(above[i], str):
                h = above[i]          # two-row header: fall back to the group label
            name = str(h).strip() if h is not None else f"Unnamed_Col_{i + 1}"
            name = re.sub(r"\s+", " ", name)
            if name in seen:
                seen[name] += 1
                name = f"{name} ({seen[name]})"
            else:
                seen[name] = 1
            names.append(name)
        body = raw.iloc[hr0 + 1:].copy()
        body.columns = names
        body.index = [i + 1 for i in body.index]       # Excel row numbers
        body = body.dropna(how="all")
        keep = [c for c in body.columns if not (c.startswith("Unnamed_Col_") and body[c].isna().all())]
        body = body[keep]
        total_rows = []
        for idx, row in body.iterrows():
            first3 = [v for v in row.tolist() if v is not None][:3]
            if any(isinstance(v, str) and TOTAL_ROW.search(v) for v in first3):
                total_rows.append(int(idx))
        return body, total_rows
