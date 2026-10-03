"""Validation of human-entered values (Edit box, rejection notes).

A reviewer-entered value is accepted only if it fits the target field's data
dictionary type / allowed value set, so edits can never break the output schema.
"""
from __future__ import annotations

from typing import Any, Optional

from . import schema
from .text import is_blank, parse_number, to_display

SPRINKLER = "Fire Sprinklers (Y/N)"
BLANK_WORDS = {"", "(blank)", "blank", "empty", "none", "null", "clear"}


def allowed_choices(field: Optional[str]) -> Optional[list[str]]:
    if field == SPRINKLER:
        return ["Y", "N", "Y13", "Y(13R)"]
    if field == "State":
        return sorted(schema.STATE_ABBRS)
    return None


def coerce_value(field: Optional[str], raw: Any) -> tuple[Any, Optional[str]]:
    """Return (typed value, error). None value with no error means 'blank cell'."""
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in BLANK_WORDS) or is_blank(raw):
        return None, None
    s = str(to_display(raw)).strip().strip("'\"")
    if field is None or field not in schema.FIELD_TYPES:
        return s, None
    if field == SPRINKLER:
        if s in schema.SPRINKLER_CODES:
            return s, None
        code = schema.SPRINKLER_MAP.get(s.upper().replace(" ", ""))
        return (code, None) if code else (None, f"'{s}' is not a sprinkler code — use Y, N, Y13 or Y(13R)")
    if field == "State":
        if s.upper() in schema.STATE_ABBRS:
            return s.upper(), None
        if s.lower() in schema.US_STATES:
            return schema.US_STATES[s.lower()], None
        return s, None
    dtype = schema.FIELD_TYPES[field]
    if dtype == "string":
        return s, None
    x, kind = parse_number(s)
    if x is None:
        return None, f"'{s}' is not a valid number for {field}"
    if dtype == "int":
        if not float(x).is_integer():
            return None, f"'{s}' must be a whole number for {field}"
        x = int(x)
        if field == "Year Built" and not (1700 <= x <= schema.CURRENT_YEAR):
            return None, f"Year Built must be between 1700 and {schema.CURRENT_YEAR}"
        if field in ("Storeys", "Number of Buildings") and x < 1:
            return None, f"{field} must be at least 1"
        return x, None
    if x < 0:
        return None, f"{field} cannot be negative"
    return float(x), None


def norm_value(v: Any) -> str:
    """Key used to group identical raw cell values ('Partial', ' partial ' -> 'partial')."""
    return "" if is_blank(v) else str(to_display(v)).strip().lower()
