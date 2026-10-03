"""Agent 2 - Schema Mapping.

Pass 0  knowledge-graph memory (approved edges, vector memory; REJECTED_AS blocks)
Pass 1  exact -> curated synonym dictionary -> RapidFuzz (>= 0.75)
Pass 2  semantic similarity with sentence embeddings, blended with a value profile
Pass 3  open-source LLM for still-unresolved headers (headers + 5 samples + neighbours)

The LLM proposes; code verifies: invented source columns, targets outside the
17 fields and duplicate targets are dropped. Confidence < 0.50 is flagged.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from .. import schema
from ..state import Mapping, SOVState
from ..text import best_match, is_blank, parse_number, sample_values

NAME = "Agent 2 · Schema Mapping"
REVIEW_THRESHOLD = 0.50
FUZZY_THRESHOLD = 0.75
_ADDR = re.compile(r"^\d+[A-Za-z]?\s+\w+")
_ZIP = re.compile(r"^\d{5}(-\d{4})?$")


# ---------------------------------------------------------------------- value profile
def profile(series: pd.Series) -> Optional[dict]:
    vals = [v for v in series.tolist() if not is_blank(v)][:300]
    n = len(vals)
    if n == 0:
        return None
    nums = [parse_number(v) for v in vals]
    parsed = [x for x, k in nums if x is not None]
    ints = [x for x in parsed if float(x).is_integer()]
    strs = [str(v).strip() for v in vals]
    p = {
        "n": n,
        "num_frac": len(parsed) / n,
        "year_frac": sum(1 for x in ints if 1700 <= x <= schema.CURRENT_YEAR + 5) / n,
        "zip_frac": sum(1 for s in strs if _ZIP.match(s.split(".")[0]) or
                        (s.isdigit() and len(s) == 4)) / n,
        "small_int_frac": sum(1 for x in ints if 0 <= x <= 200) / n,
        "money_frac": sum(1 for (x, k) in nums if x is not None and (abs(x) >= 1000 or k == "currency")) / n,
        "state_frac": sum(1 for s in strs if s.upper() in schema.STATE_ABBRS or s.lower() in schema.US_STATES) / n,
        "spr_frac": sum(1 for s in strs if s.upper().replace(" ", "") in schema.SPRINKLER_MAP) / n,
        "country_frac": sum(1 for s in strs if any(s.lower() in g for g in schema.COUNTRY_GROUPS.values())) / n,
        "text_frac": sum(1 for (x, _) in nums if x is None) / n,
        "addr_frac": sum(1 for s in strs if _ADDR.match(s)) / n,
        "unique_frac": len(set(strs)) / n,
    }
    return p


def compatibility(target: str, p: Optional[dict]) -> Optional[float]:
    if p is None:
        return None
    t = p["text_frac"] * (1 - p["addr_frac"]) * (1 - 0.8 * p["state_frac"]) * (1 - 0.8 * p["spr_frac"])
    table = {
        "Reference": 0.3 + 0.7 * p["unique_frac"] * (1 - p["addr_frac"]),
        "Address": p["addr_frac"],
        "City": t * (1 - p["country_frac"]),
        "State": p["state_frac"],
        "Zip": p["zip_frac"],
        "County": t * (1 - p["country_frac"]),
        "Country": p["country_frac"],
        "Occupancy": t,
        "Construction": t,
        "Storeys": p["small_int_frac"] * (1 - p["year_frac"]),
        "Number of Buildings": p["small_int_frac"] * (1 - p["year_frac"]),
        "Year Built": p["year_frac"],
        "Fire Sprinklers (Y/N)": p["spr_frac"],
    }
    for f in schema.MONEY_FIELDS:
        table[f] = p["money_frac"]
    return float(max(0.0, min(1.0, table[target])))


# ---------------------------------------------------------------------- agent
class SchemaMappingAgent:
    name = NAME

    def __init__(self, kg=None, embedder=None, llm=None):
        self.kg, self.embedder, self.llm = kg, embedder, llm
        self.alias = schema.alias_index()
        self._docs: list[tuple[str, str]] = []
        for f in schema.TARGET_FIELDS:
            self._docs.append((f, f"{f}: {schema.FIELD_DESCRIPTIONS[f]}"))
            self._docs += [(f, a) for a in schema.SYNONYMS.get(f, [])]
        self._doc_vecs = embedder.encode([d for _, d in self._docs]) if embedder else None
        self.send_samples = os.getenv("LLM_SEND_SAMPLES", "1") == "1"

    # ------------------------------------------------------------------ public
    def run(self, state: SOVState) -> SOVState:
        state.require("discovery")
        state.start("mapping", NAME)
        df: pd.DataFrame = state.source_df
        headers = list(df.columns)
        mappings = [self.suggest(h, df[h], neighbours=self._neigh(headers, i)) for i, h in enumerate(headers)]
        mappings = self._llm_pass(mappings, df, headers)
        mappings = self._resolve_conflicts(mappings)
        state.mappings = mappings
        state.mapping_json = self.to_json(state)
        ok = self._validate(mappings, headers)
        mapped = sum(1 for m in mappings if m.target)
        state.finish("mapping", NAME, ok,
                     f"{mapped}/{len(headers)} columns mapped, {state.mapping_json['unresolved_count']} "
                     f"need review, overall confidence {state.mapping_json['overall_confidence']:.2f}")
        return state

    @staticmethod
    def _neigh(headers: list[str], i: int) -> list[str]:
        return [h for h in headers[max(0, i - 2):i] + headers[i + 1:i + 3]]

    def suggest(self, header: str, series: pd.Series, exclude: set[str] | None = None,
                neighbours: list[str] | None = None, note: str | None = None) -> Mapping:
        """Deterministic passes 0-2 for one header. `exclude` = targets ruled out."""
        exclude = set(exclude or ())
        norm = schema.normalize_header(header)
        samples = sample_values(series)
        prof = profile(series)
        evidence: list[str] = []
        kg = self.kg.lookup(header) if self.kg else {"approved": [], "rejected": set(), "evidence": []}
        exclude |= kg["rejected"]
        evidence += kg["evidence"]
        m = Mapping(source_column=header, sample_values=samples, kg_evidence=evidence)

        # Hard negatives (TIV etc.) are never mapped automatically.
        if norm in schema.NON_TARGET_HEADERS:
            m.method, m.confidence, m.flag = "rule", 0.92, "not_in_schema"
            m.rationale = f"'{header}' is not one of the 17 target fields: {schema.NON_TARGET_HEADERS[norm]} It will be excluded from the output."
            m.uncertainty = "If this column actually holds a target field, edit the mapping."
            return m

        # Pass 0: memory
        for a in kg["approved"]:
            if a["target"] not in exclude:
                return self._finish(m, a["target"], min(0.99, 0.92 + 0.02 * a["approvals"]), "memory", prof,
                                    f"Knowledge graph: '{header}' was mapped to {a['target']} and approved "
                                    f"in {a['approvals']} earlier submission(s).")
        if self.kg is not None:
            vm = self.kg.vector_lookup(header)
            if vm and vm["target"] not in exclude:
                return self._finish(m, vm["target"], round(0.9 * vm["similarity"], 2), "memory", prof,
                                    f"Vector memory: very similar to previously approved header "
                                    f"'{vm['similar_to']}' -> {vm['target']} (similarity {vm['similarity']:.2f}).")

        # Pass 1a: exact
        for f in schema.TARGET_FIELDS:
            if f not in exclude and norm == schema.normalize_header(f):
                return self._finish(m, f, 1.0, "exact", prof, f"Header '{header}' exactly matches target field '{f}'.")
        # Pass 1b: synonym dictionary
        if norm in self.alias and self.alias[norm] not in exclude:
            f = self.alias[norm]
            return self._finish(m, f, 0.96, "synonym", prof,
                                f"'{header}' is a known insurance synonym of '{f}' in the curated dictionary.")
        # Pass 1c: fuzzy
        keys = [k for k, f in self.alias.items() if f not in exclude]
        best, score = best_match(norm, keys)
        if best and score >= FUZZY_THRESHOLD:
            f = self.alias[best]
            conf = min(0.93, 0.75 + (score - FUZZY_THRESHOLD) * 0.8)
            return self._finish(m, f, round(conf, 2), "fuzzy", prof,
                                f"Normalised header '{norm}' is {score:.0%} similar to synonym '{best}' of '{f}' (RapidFuzz).")

        # Pass 2: semantic embeddings + value profile
        sem = self._semantic(header, norm, prof, exclude)
        m.alternatives = sem[:3]
        if sem and sem[0]["similarity"] >= (self.embedder.threshold if self.embedder else 1.1):
            top = sem[0]
            conf = min(0.89, top["combined"])
            return self._finish(m, top["target"], round(conf, 2), "semantic", prof,
                                f"Semantic similarity {top['similarity']:.2f} between '{header}' and the meaning of "
                                f"'{top['target']}' ({self.embedder.backend}); sample values agree "
                                f"{(top['value_fit'] or 0):.0%}.")

        # Value profile only (e.g. headerless column)
        if prof:
            fits = sorted(((compatibility(f, prof), f) for f in schema.TARGET_FIELDS if f not in exclude), reverse=True)
            if fits and fits[0][0] >= 0.8 and fits[0][1] not in ("Reference", "City", "County", "Occupancy", "Construction"):
                m.target, m.method, m.confidence = fits[0][1], "value_profile", 0.45
                m.rationale = (f"Header '{header}' is not recognised, but {fits[0][0]:.0%} of its values look like "
                               f"{fits[0][1]} (e.g. {', '.join(samples[:3])}).")
                m.uncertainty = "Proposed from values only; the header gives no support."
                m.flag = "human_review_required"
                return m

        m.method, m.confidence, m.flag = "none", round(sem[0]["combined"], 2) if sem else 0.0, "human_review_required"
        m.rationale = f"No reliable match for '{header}' in memory, synonyms, fuzzy or semantic passes."
        m.uncertainty = ("Closest candidates: " + ", ".join(f"{a['target']} ({a['combined']:.2f})" for a in sem[:3])) if sem else "No candidates."
        if note:
            m.uncertainty += f" Reviewer note considered: '{note}'."
        return m

    def _semantic(self, header: str, norm: str, prof, exclude: set[str]) -> list[dict]:
        if self.embedder is None or self._doc_vecs is None:
            return []
        q = self.embedder.encode([f"{header} ({norm})"])
        sims = self.embedder.cosine(q, self._doc_vecs)[0]
        best: dict[str, float] = {}
        for (f, _), s in zip(self._docs, sims):
            if f not in exclude:
                best[f] = max(best.get(f, -1.0), float(s))
        out = []
        for f, s in best.items():
            fit = compatibility(f, prof)
            combined = s if fit is None else 0.75 * s + 0.25 * fit
            out.append({"target": f, "similarity": round(s, 3), "value_fit": None if fit is None else round(fit, 2),
                        "combined": round(combined, 3)})
        out.sort(key=lambda x: -x["combined"])
        return out

    @staticmethod
    def _finish(m: Mapping, target: str, conf: float, method: str, prof, why: str) -> Mapping:
        m.target, m.method, m.confidence, m.rationale = target, method, conf, why
        fit = compatibility(target, prof)
        if fit is not None:
            if fit < 0.2 and method not in ("memory",):
                penalty = 0.1 if method in ("exact", "synonym") else 0.2
                m.confidence = round(max(0.0, conf - penalty), 2)
                m.uncertainty = (f"Only {fit:.0%} of sample values look like {target} "
                                 f"(samples: {', '.join(m.sample_values[:3])}); please confirm.")
            elif fit >= 0.8 and method in ("fuzzy", "semantic"):
                m.confidence = round(min(0.95 if method == "fuzzy" else 0.89, conf + 0.05), 2)
        else:
            m.uncertainty = "Column is empty, so values could not confirm the mapping."
        if m.confidence < REVIEW_THRESHOLD:
            m.flag = "human_review_required"
        return m

    # ------------------------------------------------------------------ LLM pass
    def _llm_pass(self, mappings: list[Mapping], df: pd.DataFrame, headers: list[str]) -> list[Mapping]:
        todo = [m for m in mappings if (m.target is None and m.flag != "not_in_schema") or m.confidence < 0.6]
        if not todo or self.llm is None or not self.llm.available():
            return mappings
        taken = {m.target for m in mappings if m.target and m.confidence >= 0.6}
        cols = []
        for m in todo:
            i = headers.index(m.source_column)
            cols.append({"source_column": m.source_column,
                         "samples": m.sample_values if self.send_samples else [],
                         "neighbours": self._neigh(headers, i)})
        user = json.dumps({
            "targets": {f: schema.FIELD_DESCRIPTIONS[f] for f in schema.TARGET_FIELDS},
            "already_mapped_targets": sorted(taken),
            "columns": cols,
            "output_format": {"mappings": [{"source_column": "str", "target": "one of targets or null",
                                            "confidence": "0..1", "reason": "short plain English"}]},
        })
        system = ("You are an insurance exposure-data analyst. Map Statement of Values column headers to the "
                  "fixed target schema. Use sample values as evidence. Use only target names given, or null "
                  "if none fits (totals such as TIV, deductibles, flood zones are null). Never invent columns.")
        resp = self.llm.chat_json(system, user)
        if not resp or not isinstance(resp.get("mappings"), list):
            return mappings
        by_col = {m.source_column: m for m in todo}
        for item in resp["mappings"]:
            if not isinstance(item, dict):
                continue
            src, tgt = item.get("source_column"), item.get("target")
            if src not in by_col:                           # invented source column -> drop
                continue
            if tgt is not None and tgt not in schema.TARGET_FIELDS:   # invented target -> drop
                continue
            try:
                conf = float(item.get("confidence", 0))
            except (TypeError, ValueError):
                continue
            conf = max(0.0, min(0.88, conf))                # LLM never auto-qualifies for Approve All
            m = by_col[src]
            reason = str(item.get("reason", ""))[:300]
            if tgt is None:
                m.kg_evidence.append(f"LLM: no target fits ({reason})")
                continue
            if conf > m.confidence and tgt not in self.kg_rejected(src):
                prof = profile(df[src])
                self._finish(m, tgt, round(conf, 2), "llm", prof, f"LLM reasoning ({self.llm.model}): {reason}")
                m.flag = "human_review_required" if m.confidence < REVIEW_THRESHOLD else None
        return mappings

    def kg_rejected(self, header: str) -> set[str]:
        return self.kg.lookup(header)["rejected"] if self.kg else set()

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _resolve_conflicts(mappings: list[Mapping]) -> list[Mapping]:
        by_target: dict[str, list[Mapping]] = {}
        for m in mappings:
            if m.target:
                by_target.setdefault(m.target, []).append(m)
        for target, ms in by_target.items():
            if len(ms) < 2:
                continue
            ms.sort(key=lambda m: -m.confidence)
            winner = ms[0]
            for loser in ms[1:]:
                loser.alternatives = [{"target": target, "combined": loser.confidence}] + loser.alternatives
                loser.rationale = (f"Also resembles '{target}' ({loser.method}, {loser.confidence:.2f}) but "
                                   f"'{winner.source_column}' is a stronger match for that field.")
                loser.uncertainty = "Two columns compete for one field; confirm which should be used."
                loser.target, loser.method, loser.flag = None, "conflict", "human_review_required"
                loser.confidence = round(min(loser.confidence, 0.45), 2)
        return mappings

    @staticmethod
    def _validate(mappings: list[Mapping], headers: list[str]) -> bool:
        targets = [m.target for m in mappings if m.target]
        return (all(m.source_column in headers for m in mappings)
                and all(t in schema.TARGET_FIELDS for t in targets)
                and len(targets) == len(set(targets))
                and all(m.rationale for m in mappings))

    @staticmethod
    def to_json(state: SOVState) -> dict:
        maps = {}
        for m in state.mappings:
            d = {"target": m.target, "confidence": m.confidence, "method": m.method}
            if m.flag:
                d["flag"] = m.flag
            maps[m.source_column] = d
        mapped = [m.confidence for m in state.mappings if m.target]
        return {
            "sheet_identified": state.selected_sheet,
            "header_row": state.header_row,
            "mappings": maps,
            "unmapped_target_fields": [f for f in schema.TARGET_FIELDS
                                       if f not in {m.target for m in state.mappings}],
            "unresolved_count": sum(1 for m in state.mappings if m.flag == "human_review_required"),
            "overall_confidence": round(float(np.mean(mapped)), 2) if mapped else 0.0,
        }
