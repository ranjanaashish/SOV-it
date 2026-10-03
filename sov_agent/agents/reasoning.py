"""Re-reasoning on rejection (part of Agent 3, FR-4 / bonus 'iterative refinement').

A rejection must carry a note. The note + original context go back to the
reasoning step, which returns a revised recommendation (revision+1, pending)
together with an explicit `interpretation` ("Understood as: ...") so the
reviewer can see how the note was read. If the note is still unclear, the item
is escalated with an explicit question.

Note understanding (deterministic first, LLM only as fallback, always validated):
  mapping notes  : "it's Contents not Building Value", "should be BI", "ignore this column"
                   -> positive / negated field mentions (aliases + typos), exclusion
  data notes     : "partial means N, 100% is Y"        -> per-value mapping
                   "set to Y" / "should be 1991"        -> one value for all affected rows
                   "values are in thousands" / "x1000"  -> scaling
                   "keep" / "blank" / "remove rows" / "make positive"
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .. import schema
from ..state import Change, Recommendation, SOVState, stable_id
from ..text import best_match, parse_number, to_display
from ..values import coerce_value, norm_value
from .data_quality import DROP, NAME

# Intent vocabularies (matched on whole words, tolerant to small typos).
INTENT_WORDS = {
    "keep": ["keep", "as is", "as-is", "original", "leave", "unchanged", "correct as", "ok", "okay", "genuine",
             "dont change", "don't change", "do not change", "is correct", "are correct"],
    "blank": ["blank", "empty", "clear", "null", "remove value", "remove values", "remove the value", "erase", "wipe",
              "unknown", "leave empty", "leave blank"],
    "drop": ["remove row", "remove rows", "delete row", "delete rows", "drop row", "drop rows", "exclude row",
             "exclude rows", "remove the row", "remove these rows", "remove duplicate", "remove duplicates",
             "delete duplicates", "drop duplicates", "dedupe", "de-duplicate"],
    "abs": ["positive", "abs", "absolute", "sign", "flip", "minus", "negative is wrong", "remove minus"],
    "exclude_col": ["exclude", "ignore", "not needed", "skip", "unmapped", "not required", "irrelevant",
                    "drop column", "drop this column", "remove column", "not part of", "no field", "none of"],
}
NEGATORS = {"not", "isnt", "isn't", "no", "never", "wrong", "instead", "rather", "nor", "neither", "except", "aint"}
SCALE_WORDS = {"thousand": 1e3, "thousands": 1e3, "k": 1e3, "000s": 1e3, "'000": 1e3, "million": 1e6,
               "millions": 1e6, "mn": 1e6, "mm": 1e6}
_SET_VALUE = re.compile(
    r"(?:(?<![a-z])(?:set(?: it| them| all)?(?: to)?|change(?: it| them| all)?(?: to)?|make(?: it| them| all)?|"
    r"should be|must be|replace(?: it| them)? (?:with|by)|use|correct(?:ed)? (?:to|value is)|update(?: it)? to|"
    r"actual(?:ly)? (?:is|value is)|it'?s|it is|they are|value is)(?![a-z])|=|->|=>|→)"
    r"\s*['\"]?(.+?)['\"]?\s*[.!]?$",
    re.I)
_REPLACE_PATTERN = re.compile(
    r'^(?:(?:please\s+)?(?:change|replace|set|update)(?:\s+(?:it|this|them|the\s+values?))?\s+(?:to|with|by|into|as)\s+|'
    r'(?:make(?:\s+(?:it|this|them))?\s+(?:into\s+|to\s+)?)|'
    r'(?:(?:it\s+)?should\s+be|ought\s+to\s+be)\s+|'
    r'(?:correct\s+(?:value\s+is|to))\s+|'
    r'(?:use|put)\s+|'
    r'(?:->|=>|→)\s*)["\']?([^"\'\n]+?)["\']?\s*$',
    re.I
)
# Words that follow "it is / should be" but are judgements, not values.
NOT_VALUES = {"wrong", "incorrect", "right", "correct", "fine", "ok", "okay", "bad", "invalid", "valid", "wrong value",
              "a mistake", "an error", "error", "not right", "not correct", "unclear", "fake", "a typo", "typo",
              "positive", "negative", "removed", "deleted", "kept", "blank", "empty"}
_PAIR_SEP = re.compile(r"\s*(?:->|=>|→|=|:|\bmeans\b|\bis\b|\bare\b|\bto\b|\bas\b|\bshould be\b|\bbecomes\b)\s*", re.I)
_MULT = re.compile(r"(?:x|times|multiply(?: it| them)? by|\*)\s*([\d,]+(?:\.\d+)?)", re.I)


def _extract_replacement(note: str, field: Optional[str]) -> Optional[Any]:
    n = note.strip()
    m = _REPLACE_PATTERN.match(n)
    if m:
        raw = m.group(1).strip()
        val, err = coerce_value(field, raw)
        if not err and val is not None:
            return val
    # Direct candidate value without instructions or keywords (e.g. 'VI', 'CA', '1995', 'Y')
    if (len(n) <= 30 and not any(_has_intent(n, k) for k in ("keep", "blank", "drop", "abs", "exclude_col"))
            and not re.search(r"\b(because|why|maybe|think|guess|please|check|error)\b", n, re.I)):
        val, err = coerce_value(field, n)
        if not err and val is not None:
            return val
    return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%()'\- ]", " ", text.lower())).strip()


def _has_intent(note: str, intent: str) -> bool:
    low = " " + _norm(note) + " "
    for w in INTENT_WORDS[intent]:
        if f" {w} " in low:
            return True
    # typo tolerance on single words ("blnak", "postive")
    words = low.split()
    singles = [w for w in INTENT_WORDS[intent] if " " not in w and len(w) >= 5]
    return any(best_match(t, singles)[1] >= 0.84 for t in words if len(t) >= 5)


def field_mentions(note: str) -> tuple[list[str], list[str]]:
    """Return (positive fields in order of appearance, negated fields)."""
    low = re.sub(r"\s+", " ", note.lower())          # punctuation kept: it bounds negation scope
    norm_note = schema.normalize_header(note)
    found: list[tuple[int, str, str]] = []           # (position, field, matched text)
    phrases: list[tuple[str, str]] = [(f.lower(), f) for f in schema.TARGET_FIELDS]
    phrases += [(a, f) for a, f in schema.alias_index().items() if len(a) >= 4 or a in ("bi", "bpp", "rcv")]
    for text_, hay in ((low, "raw"), (norm_note, "norm")):
        for phrase, field in sorted(phrases, key=lambda x: -len(x[0])):
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text_):
                if not any(abs(m.start() - p) < 3 and f == field for p, f, _ in found):
                    found.append((m.start() if hay == "raw" else m.start() + 10_000, field, phrase))
    if not found:  # typo tolerance on the whole note against field names
        cand, score = best_match(low, [f.lower() for f in schema.TARGET_FIELDS])
        if cand and score >= 0.8:
            found.append((0, next(f for f in schema.TARGET_FIELDS if f.lower() == cand), cand))
    found.sort()
    pos, neg = [], []
    for p, field, phrase in found:
        source = low if p < 10_000 else norm_note
        start = p if p < 10_000 else p - 10_000
        seg = source[max(0, start - 40):start]
        seg = re.split(r"[,.;]|\bbut\b|\bit'?s\b|\bit is\b|\bactually\b", seg)[-1]
        before = seg.split()[-3:]
        bucket = neg if any(w.strip("'") in NEGATORS for w in before) else pos
        if field not in bucket:
            bucket.append(field)
    pos = [f for f in pos if f not in neg]
    return pos, neg


def value_mapping(note: str, field: Optional[str], current_values: dict[str, str]) -> tuple[dict, list[str]]:
    """Parse 'partial means N, 100% is Y' -> {'partial': 'N', '100%': 'Y'} (keys = norm_value of cells).
    `current_values` maps norm key -> display value of the affected cells."""
    mapping, errors = {}, []
    clauses = re.split(r"[;,\n]|\band\b|\bwhile\b|\bbut\b", note, flags=re.I)
    keys = sorted(current_values, key=len, reverse=True)
    for clause in clauses:
        c = clause.strip().strip(".")
        if not c:
            continue
        src = next((k for k in keys if k and re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", c.lower())), None)
        if src is None:
            continue
        idx = c.lower().find(src)
        rest = c[idx + len(src):]                       # original case preserved
        parts = _PAIR_SEP.split(rest, maxsplit=1)
        target = (parts[1] if len(parts) > 1 else parts[0]).strip().strip("'\"").strip()
        target = re.split(r"\s+(?:for|in|since|because)\s+", target)[0].strip()
        if not target:                                  # reverse order: "N for partial"
            head = re.sub(r"\b(?:use|set|put|for|with|as)\b", " ", c[:idx], flags=re.I).strip().strip("'\"")
            target = head.split()[-1] if head.split() else ""
        if not target:
            continue
        val, err = coerce_value(field, target)
        if err:
            errors.append(f"'{current_values[src]}' → '{target}': {err}")
        else:
            mapping[src] = val
    return mapping, errors


class ReasoningAgent:
    def __init__(self, mapper, llm=None):
        self.mapper, self.llm = mapper, llm

    def rereason(self, state: SOVState, rec: Recommendation, note: str) -> Recommendation:
        state.log(NAME, "re-reasoning", f"{rec.title} — note: {note}")
        self._in_llm = False
        if rec.action_type == "column_mapping":
            new = self._remap(state, rec, note)
        else:
            new = self._revise_data(state, rec, note)
        new.revision = rec.revision + 1
        new.history = list(rec.history)        # includes the rejection recorded by the orchestrator
        new.id = stable_id(rec.id, "rev", new.revision)
        new.issue_id = rec.issue_id
        new.reviewer_note = note
        rec.status = "superseded"
        state.recommendations.append(new)
        state.log(NAME, "escalated" if new.status == "escalated" else "revised",
                  f"{new.title} · {new.interpretation or ''}")
        return new

    # ------------------------------------------------------------------ mappings
    def _remap(self, state: SOVState, rec: Recommendation, note: str) -> Recommendation:
        src = rec.source_column
        rejected = {h.get("target") for h in rec.history if h.get("decision") == "rejected" and h.get("target")}
        rejected |= {rec.proposed_target}
        rejected.discard(None)
        taken = {t for s, t in state.effective_mapping(include_pending=False).items() if s != src}
        base = rec.model_copy(deep=True)
        base.status, base.decided_by, base.decided_at, base.question = "pending", None, None, None

        pos, neg = field_mentions(note)
        rejected |= set(neg)
        hinted = next((f for f in pos if f not in rejected), None)
        neg_txt = f" (you said it is not {', '.join(neg)})" if neg else ""
        if hinted:
            from .schema_mapping import compatibility, profile
            fit = compatibility(hinted, profile(state.source_df[src]))
            conf = 0.88 if fit is None or fit >= 0.5 else 0.7
            base.interpretation = f"Understood: '{src}' should map to {hinted}{neg_txt}."
            return self._map_rec(base, hinted, conf, "reviewer_hint",
                                 f"Revised after your note: you identified the column as '{hinted}'."
                                 + (f" {fit:.0%} of the sample values fit that field." if fit is not None else ""),
                                 taken)
        if _has_intent(note, "exclude_col") and not pos:
            base.interpretation = f"Understood: exclude '{src}' from the output{neg_txt}."
            return self._map_rec(base, None, 0.9, "reviewer_hint",
                                 f"Revised after your note: '{src}' will be left out of Cleaned_SOV.xlsx.", taken)

        m = self.mapper.suggest(src, state.source_df[src], exclude=rejected, note=note)
        target, conf, why, method = m.target, m.confidence, m.rationale, m.method
        llm_pick = self._llm_remap(src, m.sample_values, note, rejected, taken)
        if llm_pick and llm_pick[1] > conf:
            target, conf, why, method = llm_pick
        if target is None or conf < 0.5:
            base.proposed_target, base.field, base.status = None, None, "escalated"
            base.confidence = round(conf or 0.0, 2)
            base.title = f"Needs your decision: '{src}'"
            base.interpretation = (f"Understood: not {', '.join(sorted(rejected))}. No field was named in the note, "
                                   f"and no other field fits confidently.")
            base.rationale = f"After your note ('{note}') the agent found no confident alternative. {why}"
            base.uncertainty = m.uncertainty or "No candidate above 0.50 confidence."
            base.question = (f"Which target field should '{src}' map to (samples: {', '.join(m.sample_values[:3])})? "
                             f"Use ✏️ Edit to pick it, or approve to exclude the column.")
            return base
        base.interpretation = f"Understood: not {', '.join(sorted(rejected))}; next best match is {target}."
        return self._map_rec(base, target, conf, method, f"Revised after feedback: {why}", taken)

    @staticmethod
    def _map_rec(base: Recommendation, target, conf, method, why, taken) -> Recommendation:
        base.proposed_target, base.field, base.method = target, target, method
        base.confidence = round(min(conf, 0.89), 2)    # revised items always get a human look
        base.title = f"Map '{base.source_column}' → {target}" if target else f"Exclude '{base.source_column}'"
        base.rationale = why
        base.examples = [{"before": base.source_column, "after": target or "(excluded)"}]
        base.uncertainty = "Revised proposal; please confirm."
        if target and target in taken:
            base.uncertainty = f"'{target}' is already mapped from another column; fix that one first."
            base.confidence = min(base.confidence, 0.5)
        return base

    def _llm_remap(self, src, samples, note, rejected, taken):
        if self.llm is None or not self.llm.available():
            return None
        options = [f for f in schema.TARGET_FIELDS if f not in rejected and f not in taken]
        resp = self.llm.chat_json(
            "You fix a rejected column mapping for an insurance Statement of Values. Read the reviewer's note "
            "carefully (it may contain typos or say what the column is NOT). Pick one option or null.",
            json.dumps({"source_column": src, "samples": samples, "reviewer_note": note,
                        "rejected_targets": sorted(rejected), "options": options,
                        "output_format": {"target": "option or null", "confidence": "0..1", "reason": "str"}}),
            max_tokens=200)
        if not resp:
            return None
        t = resp.get("target")
        if t is not None and t not in options:
            return None
        try:
            conf = max(0.0, min(0.85, float(resp.get("confidence", 0))))
        except (TypeError, ValueError):
            return None
        return (t, conf, f"LLM reasoning: {str(resp.get('reason', ''))[:300]}", "llm") if t else None

    # ------------------------------------------------------------------ data recs
    def _revise_data(self, state: SOVState, rec: Recommendation, note: str) -> Recommendation:
        base = rec.model_copy(deep=True)
        base.status, base.decided_by, base.decided_at, base.question = "pending", None, None, None
        rows = [c.row for c in rec.changes] or self._issue_rows(state, rec)
        col, field = rec.source_column, rec.field
        cur = {r: state.source_df.at[r, col] for r in rows} if col and col in state.source_df.columns else {}
        distinct = {}
        for v in cur.values():
            distinct.setdefault(norm_value(v), str(to_display(v)).strip())

        # 1) explicit per-value mapping: "partial means N, 100% is Y"
        vmap, verrors = value_mapping(note, field, distinct) if cur else ({}, [])
        # 2) direct replacement or single value: "set to Y", "change it to VI", "VI", "should be 1991"
        single, serr = None, None
        if not vmap:
            rep = _extract_replacement(note, field)
            if rep is not None:
                single, serr = rep, None
            else:
                m = _SET_VALUE.search(note.strip())
                cand = m.group(1).strip() if m else ""
                if cand and cur and len(cand.split()) <= 4 and cand.lower() not in NOT_VALUES:
                    single, serr = coerce_value(field, cand)
        scale = self._scale(note)

        if vmap:
            changes = [Change(row=r, before=v, after=vmap[norm_value(v)]) for r, v in cur.items()
                       if norm_value(v) in vmap]
            uncovered = sorted({distinct[k] for k in distinct if k not in vmap})
            self._as_correction(base, changes, f"Apply your values in {field}")
            pairs = ", ".join(f"'{distinct[k]}' → {vmap[k] if vmap[k] is not None else '(blank)'}" for k in vmap)
            base.interpretation = f"Understood: replace {pairs}."
            if uncovered:
                base.interpretation += f" Not covered by the note: {', '.join(map(repr, uncovered))} (unchanged)."
            if verrors:
                base.interpretation += " Ignored: " + "; ".join(verrors)
            base.rationale = (f"Revised after your note: {len(changes)} cell(s) get the values you specified. "
                              f"Values were checked against the {field} data dictionary.")
        elif m_ok(single, serr) and cur:
            changes = [Change(row=r, before=v, after=single) for r, v in cur.items()]
            self._as_correction(base, changes, f"Set {len(changes)} value(s) in {field} to {single}")
            base.interpretation = f"Understood: set every affected {field} value to {single!r}."
            base.rationale = f"Revised after your note: all {len(changes)} affected cell(s) become {single!r}."
        elif scale and cur and field in schema.MONEY_FIELDS:
            changes = []
            for r, v in cur.items():
                x, _ = parse_number(v)
                if x is not None:
                    changes.append(Change(row=r, before=v, after=x * scale))
            self._as_correction(base, changes, f"Multiply {len(changes)} value(s) in {field} by {scale:,.0f}")
            base.interpretation = f"Understood: the values are stated in units of {scale:,.0f}; multiply them."
            base.rationale = f"Revised after your note: {len(changes)} value(s) are scaled by {scale:,.0f}."
        elif _has_intent(note, "drop") and rows:
            self._as_correction(base, [Change(row=r, before="(row)", after=DROP) for r in rows],
                                f"Exclude {len(rows)} row(s)")
            base.interpretation = f"Understood: remove the affected rows ({', '.join(map(str, rows[:8]))}…) from the output."
            base.rationale = "Revised after your note: these rows will be excluded from Cleaned_SOV.xlsx."
        elif _has_intent(note, "abs") and cur and field in schema.MONEY_FIELDS:
            changes = [Change(row=r, before=v, after=abs(parse_number(v)[0])) for r, v in cur.items()
                       if parse_number(v)[0] is not None]
            self._as_correction(base, changes, f"Make {len(changes)} value(s) positive in {field}")
            base.interpretation = "Understood: the minus sign is wrong; keep the amount, drop the sign."
            base.rationale = "Revised after your note: the sign will be removed (magnitude kept)."
        elif _has_intent(note, "blank") and cur:
            self._as_correction(base, [Change(row=r, before=v, after=None) for r, v in cur.items()],
                                f"Blank {len(cur)} value(s) in {field}")
            base.interpretation = "Understood: leave these cells empty in the output (nothing invented)."
            base.rationale = "Revised after your note: the flagged values will be left blank."
        elif _has_intent(note, "keep"):
            base.action_type, base.changes = "flag_for_review", []
            base.title = f"Keep original values ({field})"
            base.interpretation = "Understood: the current values are correct; keep them unchanged."
            base.rationale = f"Revised after your note: the {len(rows)} value(s) will be kept as they are."
        else:
            llm = self._llm_data(rec, note, distinct)
            if llm:                                          # re-run with the LLM's normalised instruction
                out = self._revise_data(state, rec, llm)
                out.interpretation = f"LLM read your note as “{llm}”. " + (out.interpretation or "")
                return out
            base.status = "escalated"
            base.title = f"Needs your decision: {rec.title}"
            base.interpretation = "Could not interpret the note." + (f" {serr}" if serr else "")
            base.rationale = (f"The agent could not turn '{note}' into an action. Try e.g. 'keep', 'blank', "
                              f"'remove rows', 'set to Y', or 'Partial means N, 100% is Y' — or use ✏️ Edit.")
            base.question = "What should happen to these values? Use ✏️ Edit to enter them directly."
        base.confidence = 0.88 if base.status == "pending" else 0.3
        base.uncertainty = ("Revised from your note; please confirm." if base.status == "pending"
                            else "Instruction unclear.")
        if base.changes:
            base.examples = [{"row": c.row, "before": to_display(c.before),
                              "after": "(row excluded)" if c.after == DROP else to_display(c.after)}
                             for c in base.changes[:3]]
        return base

    @staticmethod
    def _as_correction(base: Recommendation, changes: list[Change], title: str) -> None:
        base.action_type, base.changes, base.title = "data_correction", changes, title

    @staticmethod
    def _scale(note: str) -> Optional[float]:
        low = _norm(note)
        m = _MULT.search(note.lower())
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
        if re.search(r"\bin\b", low):
            for w, f in SCALE_WORDS.items():
                if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", low):
                    return f
        return None

    def _llm_data(self, rec: Recommendation, note: str, distinct: dict) -> Optional[str]:
        """Ask the LLM to restate an unclear note as one of the supported instructions."""
        if self.llm is None or not self.llm.available() or getattr(self, "_in_llm", False):
            return None
        resp = self.llm.chat_json(
            "Restate a reviewer's note about a data-cleansing recommendation as ONE short instruction in this "
            "grammar: 'keep' | 'blank' | 'remove rows' | 'make positive' | 'set to <value>' | "
            "'<old value> means <new value>, <old value> means <new value>' | 'multiply by <n>'. "
            "Use only values from the allowed list when one is given. Return null if the note is not an instruction.",
            json.dumps({"recommendation": rec.title, "field": rec.field,
                        "current_values": list(distinct.values())[:20],
                        "allowed_values": (sorted(schema.SPRINKLER_CODES) if rec.field == "Fire Sprinklers (Y/N)"
                                           else (sorted(schema.STATE_ABBRS) if rec.field == "State" else None)),
                        "reviewer_note": note,
                        "output_format": {"instruction": "str or null"}}, default=str),
            max_tokens=120)
        ins = resp.get("instruction") if resp else None
        if isinstance(ins, str) and 2 <= len(ins) <= 300 and ins.strip().lower() != note.strip().lower():
            self._in_llm = True
            return ins.strip()
        return None

    @staticmethod
    def _issue_rows(state: SOVState, rec: Recommendation) -> list[int]:
        for i in state.issues:
            if i.id == rec.issue_id:
                return list(i.affected_rows)
        return []


def m_ok(value, err) -> bool:
    return err is None and value is not None
