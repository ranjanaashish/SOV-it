"""Agent 3 - Data Quality & Reasoning.

1. Rule checks (deterministic, vectorisable) per mapped field and per row.
2. Builds the recommendation queue: column_mapping, data_correction,
   standardisation, flag_for_review. Only safe rule-based fixes become
   row-level changes; anything ambiguous is a flag for a human.
3. Reasoning: the LLM rewrites each template rationale in plain English with
   an uncertainty statement. Templates guarantee 100% rationale coverage even
   when the LLM is down. Nothing is applied here.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from typing import Optional

import pandas as pd

from .. import schema
from ..state import Change, Issue, Recommendation, SOVState, stable_id
from ..text import is_blank, parse_number, to_display

NAME = "Agent 3 · Data Quality & Reasoning"
DROP = "<<drop row>>"           # sentinel 'after' value meaning: exclude this row
_INT_IN_TEXT = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
_ZIP5 = re.compile(r"^(\d{5})(?:-\d{4})?$")
_ZIP4 = re.compile(r"^(\d{5})-(\d{4})$")

# issue_type -> (action_type, severity, confidence, title, rationale template, uncertainty template)
RULEBOOK: dict[str, tuple] = {
    "missing": ("flag_for_review", "Medium", 1.0, "{n} blank value(s) in {field}",
                "{n} row(s) have no {field}. They will stay blank in the output; the system never invents values.",
                "None about the detection. The business impact depends on whether {field} is needed by the exposure model."),
    "placeholder": ("data_correction", "Low", 0.95, "Replace placeholder text in {field} with blank",
                    "{n} value(s) in '{src}' are placeholders such as {ex}, not real data. Replacing them with a blank cell keeps "
                    "the column typed correctly without inventing data.",
                    "Low: these tokens are on a fixed placeholder list."),
    "currency_format": ("data_correction", "Medium", 0.97, "Strip currency symbols / separators in {field}",
                        "{n} value(s) in '{src}' are stored as text with currency symbols or thousands separators (e.g. {ex}). "
                        "Removing the formatting turns them into numbers the exposure model can read; the amount is unchanged.",
                        "Low: the numeric part is unambiguous. Accounting brackets are read as negatives."),
    "magnitude_suffix": ("data_correction", "Medium", 0.70, "Expand K/M/B abbreviations in {field}",
                         "{n} value(s) in '{src}' use abbreviations like {ex}. Expanding K=thousand, M=million, B=billion gives a numeric value.",
                         "Medium: confirm the submission uses K/M/B in the usual sense (not e.g. 'M' for thousand)."),
    "type_error": ("flag_for_review", "High", 0.90, "{n} value(s) in {field} are not valid {dtype}",
                   "{n} value(s) in '{src}' cannot be read as {dtype} (e.g. {ex}). If you approve, they will be left blank in the "
                   "output because the target column is strictly typed. Reject with the correct value, or use Edit, to supply it.",
                   "None about the detection; the correct value is unknown to the system."),
    "negative_value": ("flag_for_review", "High", 0.90, "{n} negative value(s) in {field}",
                       "{n} row(s) have a negative {field} (e.g. {ex}). Insured values cannot be negative, so this is probably a sign "
                       "error or a credit line. Approving keeps the values unchanged; reject with a note (e.g. 'make positive' or 'blank') to change them.",
                       "High: the system cannot tell whether the sign or the whole value is wrong."),
    "future_year": ("flag_for_review", "High", 0.90, "{n} Year Built value(s) in the future",
                    "{n} row(s) have Year Built after {year} (e.g. {ex}), which is impossible for an existing building. "
                    "Approving keeps them; reject with 'blank' or edit with the correct year.",
                    "High: could be a typo (e.g. 2091 vs 1991) or a planned construction date."),
    "implausible_year": ("flag_for_review", "Medium", 0.85, "{n} implausible Year Built value(s)",
                         "{n} row(s) have Year Built before 1700 (e.g. {ex}).",
                         "Medium: historic buildings exist but values like 0 or 19 are usually data-entry errors."),
    "non_integer": ("flag_for_review", "Medium", 0.85, "{n} non-integer value(s) in {field}",
                    "{n} value(s) in '{src}' have decimals (e.g. {ex}) but {field} must be an integer. If you approve they will be "
                    "left blank; edit to supply the intended whole number.",
                    "Medium: rounding could be right (2.0) or wrong (2.5 storeys = mezzanine)."),
    "text_number": ("data_correction", "Medium", 0.80, "Extract the number from text in {field}",
                    "{n} value(s) in '{src}' contain one number inside text (e.g. {ex}). Extracting it gives the integer {field}.",
                    "Medium: the text may carry meaning that is lost (e.g. 'approx')."),
    "below_min": ("flag_for_review", "High", 0.90, "{n} value(s) in {field} below 1",
                  "{n} row(s) have {field} below 1 (e.g. {ex}). A building has at least one storey and a location at least one "
                  "building. Approving keeps the values; reject with 'blank' or edit the correct number.",
                  "Medium: 0 may be a placeholder for 'unknown'."),
    "zip_plus4": ("standardisation", "Low", 0.93, "Convert ZIP+4 to 5-digit ZIP in {field}",
                  "{n} ZIP code(s) use the ZIP+4 format (e.g. {ex}). The schema stores Zip as an integer, so the 5-digit "
                  "part is kept.", "Low: the +4 suffix only refines delivery route."),
    "zip_short": ("flag_for_review", "Low", 0.85, "{n} ZIP code(s) with 4 digits",
                  "{n} ZIP code(s) have 4 digits (e.g. {ex}); a leading zero was probably dropped by Excel (New England / NJ). "
                  "The integer schema stores them as-is.", "Medium: could also be a truncated code."),
    "zip_invalid": ("flag_for_review", "Medium", 0.90, "{n} invalid ZIP code(s)",
                    "{n} value(s) in '{src}' are not valid ZIP codes (e.g. {ex}). Approving leaves them blank in the integer column.",
                    "None about detection; the correct code is unknown."),
    "sprinkler_variant": ("standardisation", "Low", 0.95, "Standardise sprinkler codes in {field}",
                          "{n} value(s) in '{src}' use free text (e.g. {ex}). They are converted to the allowed codes Y, N, Y13, Y(13R) "
                          "using the knowledge-graph value set.",
                          "Low for yes/no; check NFPA 13 vs 13R conversions."),
    "sprinkler_invalid": ("flag_for_review", "Medium", 0.85, "{n} unrecognised sprinkler value(s)",
                          "{n} value(s) in '{src}' are not a recognised sprinkler code (e.g. {ex}). Approving leaves them blank; "
                          "edit to choose Y, N, Y13 or Y(13R).",
                          "High: values like 'Partial' do not map to a single code."),
    "state_full_name": ("standardisation", "Low", 0.97, "Convert state names to 2-letter codes",
                        "{n} value(s) in '{src}' are full state names (e.g. {ex}). The schema prefers 2-letter abbreviations.",
                        "Low: state names map one-to-one to USPS codes."),
    "state_case": ("standardisation", "Low", 0.98, "Upper-case state codes", "{n} state code(s) are not upper case (e.g. {ex}).",
                   "None."),
    "state_invalid": ("flag_for_review", "Medium", 0.80, "{n} unrecognised state value(s)",
                      "{n} value(s) in '{src}' are not a US state name or code (e.g. {ex}). They are kept as-is if approved.",
                      "Medium: could be a non-US region or a typo."),
    "country_inconsistent": ("standardisation", "Low", 0.85, "Use one spelling for each country",
                             "The country column mixes spellings of the same country (e.g. {ex}). Using the most frequent spelling "
                             "makes the column consistent.", "Low: only spellings of the same country are merged."),
    "whitespace": ("standardisation", "Low", 0.99, "Trim extra spaces in {field}",
                   "{n} value(s) in '{src}' have leading, trailing or repeated spaces (e.g. {ex}).", "None."),
    "duplicate_reference": ("flag_for_review", "High", 0.90, "{n} rows share a Reference",
                            "{n} row(s) reuse the same Reference (e.g. {ex}). Reference must be unique, otherwise exposure is "
                            "double counted or overwritten downstream. Approving keeps them; reject with 'remove row' to drop duplicates.",
                            "Medium: they may be separate buildings at one location that need a suffix."),
    "duplicate_row": ("flag_for_review", "Medium", 0.90, "{n} fully duplicated row(s)",
                      "{n} row(s) are exact copies of an earlier row (rows {ex}). Approving keeps them; reject with 'remove row' to drop them.",
                      "Low about detection."),
    "total_row": ("data_correction", "Medium", 0.90, "Exclude {n} totals row(s)",
                  "{n} row(s) look like summary/totals rows (rows {ex}), not properties. Excluding them avoids double counting TIV.",
                  "Low: detected by a 'Total' label in the first cells."),
    "no_values": ("flag_for_review", "Medium", 0.85, "{n} row(s) with no insured values",
                  "{n} row(s) have no Building Value, Contents, BI or Other (rows {ex}). They add no exposure.",
                  "Medium: values may be in an unmapped column."),
    "missing_field": ("flag_for_review", "Medium", 1.0, "No source column for {field}",
                      "No column in the sheet maps to '{field}', so it will be an empty column in Cleaned_SOV.xlsx. If a column "
                      "holds this data, edit its mapping instead.", "None."),
}


def _ex(values: list, k: int = 3) -> str:
    return ", ".join(repr(to_display(v)) for v in values[:k]) or "-"


class DataQualityAgent:
    name = NAME

    def __init__(self, kg=None, llm=None):
        self.kg, self.llm = kg, llm

    # ------------------------------------------------------------------ public
    def run(self, state: SOVState) -> SOVState:
        state.require("mapping")
        state.start("quality", NAME)
        if not any(r.action_type == "column_mapping" for r in state.recommendations):
            state.recommendations = self._mapping_recs(state)

        mapping = state.effective_mapping(include_pending=True)
        issues, data_recs = self.check(state.source_df, mapping, state.summary.get("total_rows_detected", []))
        state.issues = issues
        self._merge(state, data_recs, {i.id for i in issues})
        state.quality = self._quality_report(state, mapping)
        done = state.summary.setdefault("explained_ids", [])
        todo = [r for r in state.recommendations if r.status == "pending" and r.id not in done]
        self.explain(state, todo)
        done.extend(r.id for r in todo)
        ok = all(r.rationale for r in state.active_recs())
        state.finish("quality", NAME, ok, f"{len(issues)} issue(s); {len(state.active_recs())} recommendation(s) "
                                          f"queued; data quality score {state.quality['dq_score']}")
        return state

    # ------------------------------------------------------------------ mapping recs
    def _mapping_recs(self, state: SOVState) -> list[Recommendation]:
        recs = []
        for m in state.mappings:
            target = m.target
            if target:
                title = f"Map '{m.source_column}' → {target}"
            elif m.flag == "not_in_schema":
                title = f"Exclude '{m.source_column}' (not in target schema)"
            else:
                title = f"Unresolved column '{m.source_column}'"
            ex = [{"before": m.source_column, "after": target or "(excluded)"}]
            ev = list(m.kg_evidence)
            recs.append(Recommendation(
                id=stable_id("map", m.source_column), action_type="column_mapping", title=title,
                field=target, source_column=m.source_column, proposed_target=target, examples=ex,
                rationale=m.rationale, confidence=m.confidence, method=m.method, kg_evidence=ev,
                uncertainty=m.uncertainty or ("Low: strong match." if m.confidence >= 0.9 else
                                              "Please confirm; confidence below 0.90."),
                severity="High" if m.flag == "human_review_required" else ("Low" if m.confidence >= 0.9 else "Medium"),
                question=(f"Which target field does '{m.source_column}' hold (samples: {', '.join(m.sample_values[:3])})?"
                          if m.flag == "human_review_required" else None),
            ))
        return recs

    # ------------------------------------------------------------------ checks
    def check(self, df: pd.DataFrame, mapping: dict[str, str], total_rows: list[int]):
        issues: list[Issue] = []
        recs: list[Recommendation] = []
        rows_by_field: dict[str, set[int]] = defaultdict(set)

        def add(itype: str, field: str, src: Optional[str], rows: list[int], before: list,
                changes: Optional[list[Change]] = None, conf: Optional[float] = None, extra: dict | None = None):
            if not rows:
                return
            action, sev, base_conf, title, why, unc = RULEBOOK[itype]
            fmt = {"n": len(rows), "field": field, "src": src or field, "ex": _ex(before),
                   "dtype": schema.FIELD_TYPES.get(field, ""), "year": schema.CURRENT_YEAR, **(extra or {})}
            iid = stable_id(itype, field, src, tuple(rows))
            issue = Issue(id=iid, issue_type=itype, field=field, source_column=src, severity=sev,
                          affected_rows=rows, count=len(rows), description=why.format(**fmt),
                          examples=[{"row": r, "value": to_display(b)} for r, b in zip(rows[:5], before[:5])])
            issues.append(issue)
            rows_by_field[field].update(rows)
            exs = ([{"row": c.row, "before": to_display(c.before), "after": None if c.after == DROP else to_display(c.after)}
                    for c in (changes or [])[:3]]
                   or [{"row": r, "before": to_display(b), "after": to_display(b)} for r, b in zip(rows[:3], before[:3])])
            ev = []
            if self.kg is not None and field in schema.FIELD_TYPES:
                ev = [f"Graph rule ({field} -CONSTRAINED_BY-> rule): {t}" for t in self.kg.rules_for(field)][:2]
            recs.append(Recommendation(
                id=iid, issue_id=iid, issue_type=itype, action_type=action, title=title.format(**fmt), field=field,
                source_column=src, changes=changes or [], examples=exs, rationale=why.format(**fmt),
                uncertainty=unc.format(**fmt), confidence=conf if conf is not None else base_conf, severity=sev,
                method="rule", kg_evidence=ev))

        for src, field in mapping.items():
            if src not in df.columns:
                continue
            s = df[src]
            items = list(s.items())
            dtype = schema.FIELD_TYPES[field]
            blank = [r for r, v in items if is_blank(v)]
            add("missing", field, src, blank, [None] * len(blank))
            nonblank = [(r, v) for r, v in items if not is_blank(v)]

            if dtype == "float":
                self._check_money(add, field, src, nonblank)
            elif field == "Zip":
                self._check_zip(add, field, src, nonblank)
            elif dtype == "int":
                self._check_int(add, field, src, nonblank)
            else:
                self._check_string(add, field, src, nonblank, s)

        unmapped = [f for f in schema.TARGET_FIELDS if f not in set(mapping.values())]
        for f in unmapped:
            iid = stable_id("missing_field", f)
            action, sev, conf, title, why, unc = RULEBOOK["missing_field"]
            issues.append(Issue(id=iid, issue_type="missing_field", field=f, severity=sev,
                                description=why.format(field=f)))
            recs.append(Recommendation(id=iid, issue_id=iid, issue_type="missing_field", action_type=action,
                                       title=title.format(field=f), field=f, rationale=why.format(field=f),
                                       uncertainty=unc, confidence=conf, severity=sev, method="rule",
                                       examples=[{"before": "(no column)", "after": "(blank column)"}]))

        # cross-row / cross-field checks
        ref_src = next((s for s, f in mapping.items() if f == "Reference"), None)
        if ref_src:
            vals = df[ref_src].map(lambda v: None if is_blank(v) else str(to_display(v)).strip())
            dup = vals[vals.notna() & vals.duplicated(keep=False)]
            add("duplicate_reference", "Reference", ref_src, [int(r) for r in dup.index], dup.tolist())
        mapped_cols = [s for s in mapping if s in df.columns]
        if mapped_cols:
            dupe_rows = df[mapped_cols].astype(str).duplicated(keep="first")
            rows = [int(r) for r in dupe_rows[dupe_rows].index]
            add("duplicate_row", "(row)", None, rows, rows,
                changes=None)
        money_src = [s for s, f in mapping.items() if f in schema.MONEY_FIELDS and s in df.columns]
        if money_src:
            empty = df[money_src].apply(lambda col: col.map(lambda v: parse_number(v)[0] in (None, 0.0))).all(axis=1)
            rows = [int(r) for r in empty[empty].index if r not in total_rows]
            add("no_values", "(row)", None, rows, rows)
        if total_rows:
            add("total_row", "(row)", None, total_rows, total_rows,
                changes=[Change(row=r, before="(row)", after=DROP) for r in total_rows])
        return issues, recs

    @staticmethod
    def _check_money(add, field, src, nonblank):
        buckets = defaultdict(list)
        for r, v in nonblank:
            x, kind = parse_number(v)
            buckets[kind].append((r, v, x))
            if x is not None and x < 0:
                buckets["negative"].append((r, v, x))
        p = buckets["placeholder"]
        add("placeholder", field, src, [r for r, *_ in p], [v for _, v, _ in p],
            [Change(row=r, before=v, after=None) for r, v, _ in p])
        c = buckets["currency"]
        add("currency_format", field, src, [r for r, *_ in c], [v for _, v, _ in c],
            [Change(row=r, before=v, after=x) for r, v, x in c])
        m = buckets["magnitude"]
        add("magnitude_suffix", field, src, [r for r, *_ in m], [v for _, v, _ in m],
            [Change(row=r, before=v, after=x) for r, v, x in m])
        bad = buckets["invalid"]
        add("type_error", field, src, [r for r, *_ in bad], [v for _, v, _ in bad])
        neg = buckets["negative"]
        add("negative_value", field, src, [r for r, *_ in neg], [v for _, v, _ in neg])

    @staticmethod
    def _check_int(add, field, src, nonblank):
        ph, txt, bad, frac, low, future, old = [], [], [], [], [], [], []
        for r, v in nonblank:
            x, kind = parse_number(v)
            if kind == "placeholder":
                ph.append((r, v)); continue
            if x is None:
                nums = _INT_IN_TEXT.findall(str(v))
                (txt if len(nums) == 1 else bad).append((r, v, int(nums[0]) if len(nums) == 1 else None))
                continue
            if not float(x).is_integer():
                frac.append((r, v)); continue
            x = int(x)
            if field == "Year Built":
                if x > schema.CURRENT_YEAR:
                    future.append((r, v))
                elif x < 1700:
                    old.append((r, v))
            elif x < 1:
                low.append((r, v))
        add("placeholder", field, src, [r for r, _ in ph], [v for _, v in ph], [Change(row=r, before=v, after=None) for r, v in ph])
        add("text_number", field, src, [r for r, *_ in txt], [v for _, v, _ in txt],
            [Change(row=r, before=v, after=n) for r, v, n in txt])
        add("type_error", field, src, [r for r, *_ in bad], [v for _, v, _ in bad])
        add("non_integer", field, src, [r for r, _ in frac], [v for _, v in frac])
        add("below_min", field, src, [r for r, _ in low], [v for _, v in low])
        add("future_year", field, src, [r for r, _ in future], [v for _, v in future])
        add("implausible_year", field, src, [r for r, _ in old], [v for _, v in old])

    @staticmethod
    def _check_zip(add, field, src, nonblank):
        p4, short, bad, ph = [], [], [], []
        for r, v in nonblank:
            s = str(to_display(v)).strip()
            if s.lower() in schema.PLACEHOLDERS:
                ph.append((r, v))
            elif _ZIP4.match(s):
                p4.append((r, v, int(_ZIP4.match(s).group(1))))
            elif _ZIP5.match(s):
                continue
            elif s.isdigit() and len(s) == 4:
                short.append((r, v))
            else:
                bad.append((r, v))
        add("placeholder", field, src, [r for r, _ in ph], [v for _, v in ph], [Change(row=r, before=v, after=None) for r, v in ph])
        add("zip_plus4", field, src, [r for r, *_ in p4], [v for _, v, _ in p4], [Change(row=r, before=v, after=z) for r, v, z in p4])
        add("zip_short", field, src, [r for r, _ in short], [v for _, v in short])
        add("zip_invalid", field, src, [r for r, _ in bad], [v for _, v in bad])

    @staticmethod
    def _check_string(add, field, src, nonblank, series):
        ws, ph = [], []
        for r, v in nonblank:
            if isinstance(v, str):
                if v.strip().lower() in schema.PLACEHOLDERS:
                    ph.append((r, v))
                elif v != re.sub(r"\s+", " ", v).strip():
                    ws.append((r, v, re.sub(r"\s+", " ", v).strip()))
        add("placeholder", field, src, [r for r, _ in ph], [v for _, v in ph], [Change(row=r, before=v, after=None) for r, v in ph])
        ph_rows = {r for r, _ in ph}
        clean = [(r, v) for r, v in nonblank if r not in ph_rows]

        if field == "Fire Sprinklers (Y/N)":
            var, bad = [], []
            for r, v in clean:
                s = str(to_display(v)).strip()
                key = s.upper().replace(" ", "")
                if s in schema.SPRINKLER_CODES:
                    continue
                if key in schema.SPRINKLER_MAP:
                    var.append((r, v, schema.SPRINKLER_MAP[key]))
                else:
                    bad.append((r, v))
            conf = 0.95 if all(c in ("Y", "N") for *_, c in var) else 0.85
            add("sprinkler_variant", field, src, [r for r, *_ in var], [v for _, v, _ in var],
                [Change(row=r, before=v, after=c) for r, v, c in var], conf=conf)
            add("sprinkler_invalid", field, src, [r for r, _ in bad], [v for _, v in bad])
            return
        if field == "State":
            full, case, bad = [], [], []
            for r, v in clean:
                s = str(v).strip()
                if s in schema.STATE_ABBRS:
                    continue
                if s.upper() in schema.STATE_ABBRS:
                    case.append((r, v, s.upper()))
                elif re.sub(r"\s+", " ", s.lower()) in schema.US_STATES:
                    full.append((r, v, schema.US_STATES[re.sub(r"\s+", " ", s.lower())]))
                else:
                    bad.append((r, v))
            add("state_full_name", field, src, [r for r, *_ in full], [v for _, v, _ in full],
                [Change(row=r, before=v, after=a) for r, v, a in full])
            add("state_case", field, src, [r for r, *_ in case], [v for _, v, _ in case],
                [Change(row=r, before=v, after=a) for r, v, a in case])
            add("state_invalid", field, src, [r for r, _ in bad], [v for _, v in bad])
            done = {r for r, *_ in full + case}
            ws = [w for w in ws if w[0] not in done]
        if field == "Country":
            groups = defaultdict(Counter)
            for r, v in clean:
                s = str(v).strip()
                for g, spellings in schema.COUNTRY_GROUPS.items():
                    if s.lower() in spellings:
                        groups[g][s] += 1
            changes = []
            for g, cnt in groups.items():
                if len(cnt) > 1:
                    canon = cnt.most_common(1)[0][0]
                    for r, v in clean:
                        s = str(v).strip()
                        if s.lower() in schema.COUNTRY_GROUPS[g] and s != canon:
                            changes.append(Change(row=r, before=v, after=canon))
            add("country_inconsistent", field, src, [c.row for c in changes], [c.before for c in changes], changes)
            done = {c.row for c in changes}
            ws = [w for w in ws if w[0] not in done]
        add("whitespace", field, src, [r for r, *_ in ws], [v for _, v, _ in ws],
            [Change(row=r, before=v, after=a) for r, v, a in ws])

    # ------------------------------------------------------------------ merge & report
    @staticmethod
    def _merge(state: SOVState, new: list[Recommendation], issue_ids: set[str]) -> None:
        existing = {r.id: r for r in state.recommendations}
        kept: list[Recommendation] = [r for r in state.recommendations if r.action_type == "column_mapping"]
        for r in state.recommendations:
            if r.action_type == "column_mapping":
                continue
            if r.issue_id in issue_ids and r.revision > 0:
                kept.append(r)                  # re-reasoned revision of a live issue
        new_ids = {r.id for r in new}
        for r in new:
            old = existing.get(r.id)
            kept.append(old if old is not None else r)
        for r in state.recommendations:        # decisions on vanished issues are superseded
            if r.action_type != "column_mapping" and r.id not in new_ids and r.revision == 0:
                if r.status != "superseded":
                    r.status = "superseded"
                    r.history.append({"decision": "superseded", "note": "mapping changed; issue no longer applies"})
                kept.append(r)
        seen, out = set(), []
        for r in kept:
            if r.id not in seen:
                seen.add(r.id)
                out.append(r)
        state.recommendations = out

    @staticmethod
    def _quality_report(state: SOVState, mapping: dict[str, str]) -> dict:
        df = state.source_df
        n = max(1, len(df))
        completeness = {}
        inv = {f: s for s, f in mapping.items()}
        for f in schema.TARGET_FIELDS:
            src = inv.get(f)
            if src is None or src not in df.columns:
                completeness[f] = 0.0
            else:
                completeness[f] = round(float(df[src].map(lambda v: not is_blank(v)
                                                           and str(v).strip().lower() not in schema.PLACEHOLDERS).mean()), 3)
        flagged_rows = set()
        for i in state.issues:
            if i.issue_type not in ("missing", "missing_field", "whitespace", "state_case"):
                flagged_rows.update(i.affected_rows)
        conf = [m.confidence for m in state.mappings if m.target] or [0.0]
        dq = 100 * (0.5 * sum(completeness.values()) / len(completeness)
                    + 0.3 * (1 - len(flagged_rows) / n) + 0.2 * (sum(conf) / len(conf)))
        by_type = Counter(i.issue_type for i in state.issues)
        per_row = defaultdict(list)
        for i in state.issues:
            for r in i.affected_rows:
                per_row[r].append(f"{i.issue_type}:{i.field}")
        return {"completeness": completeness, "dq_score": round(dq, 1), "rows": len(df),
                "issues_by_type": dict(by_type), "rows_flagged": len(flagged_rows),
                "row_flags": {int(k): v for k, v in sorted(per_row.items())}}

    # ------------------------------------------------------------------ LLM reasoning
    # Items whose template rationale is already precise are not sent to the LLM (saves calls).
    _SKIP_METHODS = {"exact", "synonym", "memory", "rule_exact"}
    _SKIP_TYPES = {"missing_field", "whitespace", "state_case", "missing"}

    def explain(self, state: SOVState, recs: list[Recommendation], chunk: int = 8) -> None:
        """Rewrite template rationales with the LLM (plain English + uncertainty).

        Small batches are sent in parallel (LLMClient.map_json), so a queue of 40
        items costs ~one round-trip instead of three sequential ones."""
        if not recs or self.llm is None or not self.llm.available():
            return
        recs = [r for r in recs if not (r.action_type == "column_mapping" and r.method in self._SKIP_METHODS)
                and r.issue_type not in self._SKIP_TYPES]
        if not recs:
            return
        system = ("You explain data-cleansing recommendations for an insurance Statement of Values to a non-technical "
                  "reviewer. For each item write: rationale (max 2 short sentences: what was found, why it matters for "
                  "exposure pricing, what approving does), uncertainty (1 short sentence). Do not change the proposed "
                  "action or invent values.")
        batches = [recs[i:i + chunk] for i in range(0, len(recs), chunk)]
        prompts = []
        for batch in batches:
            items = [{"id": r.id, "type": r.action_type, "field": r.field, "column": r.source_column,
                      "issue": r.issue_type, "conf": r.confidence, "examples": r.examples[:2],
                      "draft": r.rationale} for r in batch]
            prompts.append((system, json.dumps({"items": items, "output_format":
                            {"items": [{"id": "str", "rationale": "str", "uncertainty": "str"}]}}, default=str)))
        t0 = time.time()
        results = self.llm.map_json(prompts, max_tokens=180 * chunk)
        improved = 0
        for batch, resp in zip(batches, results):
            if not resp or not isinstance(resp.get("items"), list):
                continue
            by_id = {r.id: r for r in batch}
            for it in resp["items"]:
                if not isinstance(it, dict) or it.get("id") not in by_id:
                    continue
                rat, unc = it.get("rationale"), it.get("uncertainty")
                r = by_id[it["id"]]
                if isinstance(rat, str) and 20 <= len(rat) <= 800:
                    r.rationale = rat.strip()
                    improved += 1
                if isinstance(unc, str) and 3 <= len(unc) <= 400:
                    r.uncertainty = unc.strip()
        state.log(NAME, "llm_explained", f"{improved}/{len(recs)} rationale(s) in {time.time() - t0:.1f}s "
                                         f"({len(batches)} parallel batch(es))")