"""String / value helpers shared by agents. RapidFuzz with a stdlib fallback."""
from __future__ import annotations

import difflib
import math
import re
from typing import Any, Iterable, Optional

try:
    from rapidfuzz import fuzz, process  # type: ignore

    def fuzzy_ratio(a: str, b: str) -> float:
        return fuzz.token_sort_ratio(a, b) / 100.0

    def best_match(query: str, choices: Iterable[str]) -> tuple[Optional[str], float]:
        choices = list(choices)
        if not choices:
            return None, 0.0
        m = process.extractOne(query, choices, scorer=fuzz.token_sort_ratio)
        return (m[0], m[1] / 100.0) if m else (None, 0.0)

except ImportError:  # pragma: no cover - fallback
    def fuzzy_ratio(a: str, b: str) -> float:
        a2, b2 = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
        return difflib.SequenceMatcher(None, a2, b2).ratio()

    def best_match(query: str, choices: Iterable[str]) -> tuple[Optional[str], float]:
        best, score = None, 0.0
        for c in choices:
            s = fuzzy_ratio(query, c)
            if s > score:
                best, score = c, s
        return best, score


_NUM = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")
CURRENCY = re.compile(r"[$€£₹¥]|\b(usd|inr|eur|gbp)\b", re.I)
MAGNITUDE = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*([kmb])$", re.I)


def is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def is_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return not (isinstance(v, float) and math.isnan(v))
    return isinstance(v, str) and bool(_NUM.match(v.strip().replace(",", "")))


def parse_number(v: Any) -> tuple[Optional[float], str]:
    """Return (value, kind). kind: ok | blank | placeholder | currency | magnitude | invalid.
    'currency' covers symbols, thousands separators and accounting negatives."""
    from .schema import PLACEHOLDERS

    if is_blank(v):
        return None, "blank"
    if isinstance(v, bool):
        return None, "invalid"
    if isinstance(v, (int, float)):
        return float(v), "ok"
    s = str(v).strip()
    if s.lower() in PLACEHOLDERS:
        return None, "placeholder"
    if _NUM.match(s):
        return float(s), "ok"
    t = CURRENCY.sub("", s).replace(",", "").replace(" ", "")
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    if _NUM.match(t):
        x = float(t)
        return (-x if neg else x), "currency"
    m = MAGNITUDE.match(t)
    if m:
        mult = {"k": 1e3, "m": 1e6, "b": 1e9}[m.group(2).lower()]
        return float(m.group(1)) * mult, "magnitude"
    return None, "invalid"


def to_display(v: Any) -> Any:
    if is_blank(v):
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def sample_values(series, n: int = 5) -> list[str]:
    vals = [str(to_display(v)) for v in series.tolist() if not is_blank(v)]
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v[:40])
        if len(out) >= n:
            break
    return out
