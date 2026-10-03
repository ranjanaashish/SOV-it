"""Target schema, data dictionary and seeded domain knowledge.

Everything here is deterministic reference data. It seeds the knowledge graph
at start-up so the system is useful from the very first SOV.
"""
from __future__ import annotations

import datetime as _dt
import re

CURRENT_YEAR = _dt.date.today().year

# Exact, case-sensitive, ordered output schema (C-03).
TARGET_FIELDS: list[str] = [
    "Reference",
    "Address",
    "City",
    "State",
    "Zip",
    "County",
    "Country",
    "Building Value",
    "Contents",
    "BI",
    "Occupancy",
    "Construction",
    "Storeys",
    "Number of Buildings",
    "Year Built",
    "Fire Sprinklers (Y/N)",
    "Other",
]

# dtype: "string" | "int" | "float"
FIELD_TYPES: dict[str, str] = {
    "Reference": "string",
    "Address": "string",
    "City": "string",
    "State": "string",
    "Zip": "int",
    "County": "string",
    "Country": "string",
    "Building Value": "float",
    "Contents": "float",
    "BI": "float",
    "Occupancy": "string",
    "Construction": "string",
    "Storeys": "int",
    "Number of Buildings": "int",
    "Year Built": "int",
    "Fire Sprinklers (Y/N)": "string",
    "Other": "float",
}

FIELD_DESCRIPTIONS: dict[str, str] = {
    "Reference": "Unique identifier for the insured property or location number",
    "Address": "Full street address of the insured property",
    "City": "City or town where the insured property is located",
    "State": "State or region, 2-letter abbreviation",
    "Zip": "Postal or ZIP code of the property",
    "County": "County or parish for regional risk analysis",
    "Country": "Country of the property, jurisdiction of coverage",
    "Building Value": "Insured value of the building structure, replacement cost",
    "Contents": "Value of contents or business personal property in the building",
    "BI": "Business income or business interruption value, loss of rents",
    "Occupancy": "Use or purpose of the building, residential commercial industrial",
    "Construction": "Construction type or material, frame masonry steel concrete",
    "Storeys": "Number of floors or stories of the building",
    "Number of Buildings": "Total count of buildings at the location",
    "Year Built": "Year the building was constructed",
    "Fire Sprinklers (Y/N)": "Whether fire sprinklers are installed, Y N Y13 Y(13R)",
    "Other": "Other insured values not building or contents, outdoor property",
}

MONEY_FIELDS = ["Building Value", "Contents", "BI", "Other"]
INT_FIELDS = ["Zip", "Storeys", "Number of Buildings", "Year Built"]
STRING_FIELDS = [f for f, t in FIELD_TYPES.items() if t == "string"]

# Curated insurance synonym dictionary (raw spellings; normalised at load time).
SYNONYMS: dict[str, list[str]] = {
    "Reference": [
        "ref", "reference", "reference number", "ref no", "loc", "loc #", "loc no", "loc num",
        "location", "location number", "location no", "location id", "loc id", "property id",
        "property number", "site id", "site number", "risk id", "bldg #", "building number",
        "building id", "unit no", "premises number", "prem #",
    ],
    "Address": [
        "address", "street address", "street", "address 1", "address line 1", "addr", "addr 1",
        "location address", "property address", "physical address", "site address",
    ],
    "City": ["city", "town", "city name", "municipality", "city/town"],
    "State": ["state", "st", "state code", "province", "state/province", "region", "state abbr"],
    "Zip": ["zip", "zip code", "zipcode", "postal", "postal code", "postcode", "zip/postal", "zip5"],
    "County": ["county", "county name", "parish", "parish/county", "county/parish", "borough"],
    "Country": ["country", "cntry", "country code", "nation", "ctry"],
    "Building Value": [
        "building value", "building", "bldg", "bldg value", "building limit", "bldg limit",
        "bldg repl cost", "bldg repl cost new", "building replacement cost", "replacement cost",
        "rcv", "repl cost", "building rcv", "bldg tiv", "structure value", "real property",
        "building replacement value", "bldg val", "building amount",
    ],
    "Contents": [
        "contents", "contents value", "bpp", "business personal property", "personal property",
        "contents limit", "contents rcv", "stock", "equipment", "contents & equipment", "cont",
    ],
    "BI": [
        "bi", "business income", "business interruption", "bi/ee", "bi ee", "bi value",
        "time element", "loss of rents", "rents", "rental income", "bi limit", "extra expense",
        "business income/extra expense",
    ],
    "Occupancy": [
        "occupancy", "occ", "occupancy type", "occupancy description", "occ desc", "use",
        "building use", "property use", "occupancy class", "occ class", "property type",
    ],
    "Construction": [
        "construction", "const", "construction type", "construction class", "const class",
        "const type", "iso class", "iso construction", "building construction", "frame type",
        "construction description",
    ],
    "Storeys": [
        "storeys", "stories", "story", "no of stories", "number of stories", "# stories",
        "floors", "no of floors", "number of floors", "# floors", "# flrs", "num floors",
        "flrs", "stys", "height in stories",
    ],
    "Number of Buildings": [
        "number of buildings", "no of buildings", "no. bldgs", "# bldgs", "# of bldgs",
        "num bldgs", "building count", "bldg count", "no bldgs", "count of buildings",
        "total buildings",
    ],
    "Year Built": [
        "year built", "yr built", "yr blt", "year of construction", "construction year",
        "yob", "built", "year constructed", "original year built", "orig yr built",
    ],
    "Fire Sprinklers (Y/N)": [
        "fire sprinklers", "fire sprinklers y/n", "sprinklers", "sprinklered", "sprinkler",
        "sprinklered y/n", "sprinkler y/n", "fire prot.", "fire protection", "spr", "sprk",
        "sprinkler system", "auto sprinkler", "sprinklered (y/n)",
    ],
    "Other": [
        "other", "other value", "other values", "other tiv", "misc", "misc value", "misc tiv",
        "outdoor property", "other property", "other limit", "property in the open",
    ],
}

# Headers that look insurance-like but must NOT map to any target. Mapping TIV
# into Building Value would double count, so these are hard negatives.
NON_TARGET_HEADERS: dict[str, str] = {
    "tiv": "TIV is a derived total of building + contents + BI + other; mapping it would double count.",
    "total insured value": "TIV is a derived total; mapping it would double count.",
    "total tiv": "TIV is a derived total; mapping it would double count.",
    "total value": "A total column is derived from the component values.",
    "total": "A total column is derived from the component values.",
    "flood zone": "Flood zone is a hazard attribute, not part of the 17-field schema.",
    "eq deductible": "Deductibles are policy terms, not part of the 17-field schema.",
    "deductible": "Deductibles are policy terms, not part of the 17-field schema.",
    "named insured": "Named insured is an entity name, not a property attribute in the schema.",
    "latitude": "Geocodes are not part of the 17-field schema.",
    "longitude": "Geocodes are not part of the 17-field schema.",
}

# Abbreviation expansion used by header normalisation.
ABBREVIATIONS: dict[str, str] = {
    "bldg": "building", "bld": "building", "bldgs": "buildings", "blt": "built",
    "yr": "year", "yrs": "years", "no": "number", "num": "number", "nbr": "number",
    "const": "construction", "constr": "construction", "occ": "occupancy",
    "repl": "replacement", "val": "value", "vals": "values", "prot": "protection",
    "addr": "address", "cntry": "country", "ctry": "country", "flrs": "floors", "fl": "floors",
    "stys": "stories", "loc": "location", "ref": "reference", "desc": "description",
    "spr": "sprinkler", "sprk": "sprinkler", "amt": "amount", "misc": "miscellaneous",
}

SPRINKLER_CODES = {"Y", "N", "Y13", "Y(13R)"}
# Variant -> canonical code. Keys are compared after upper() and whitespace removal.
SPRINKLER_MAP: dict[str, str] = {
    "Y": "Y", "YES": "Y", "TRUE": "Y", "1": "Y", "SPRINKLERED": "Y", "FULL": "Y",
    "FULLYSPRINKLERED": "Y", "X": "Y",
    "N": "N", "NO": "N", "FALSE": "N", "0": "N", "NONE": "N", "UNSPRINKLERED": "N",
    "Y13": "Y13", "13": "Y13", "NFPA13": "Y13", "Y-13": "Y13",
    "Y(13R)": "Y(13R)", "Y13R": "Y(13R)", "13R": "Y(13R)", "NFPA13R": "Y(13R)", "Y-13R": "Y(13R)",
}

US_STATES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
    "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "puerto rico": "PR",
    "virgin islands": "VI",
    "u.s. virgin islands": "VI",
    "us virgin islands": "VI",
    "guam": "GU",
    "american samoa": "AS",
    "northern mariana islands": "MP",
}
STATE_ABBRS = set(US_STATES.values())

# Country spellings grouped by canonical country (used only to detect inconsistency).
COUNTRY_GROUPS: dict[str, set[str]] = {
    "united states": {"us", "usa", "u.s.", "u.s.a.", "united states", "united states of america", "america"},
    "canada": {"ca", "can", "canada"},
    "united kingdom": {"uk", "gb", "gbr", "united kingdom", "great britain", "england"},
}

PLACEHOLDERS = {"n/a", "na", "n.a.", "tbd", "tba", "unknown", "unk", "-", "--", "?", "none", "null", "nil", "#n/a", "not available"}

# Business rules (seeded into the KG as CONSTRAINED_BY edges).
RULES: dict[str, dict] = {
    "non_negative_money": {"fields": MONEY_FIELDS, "text": "Monetary values cannot be negative."},
    "year_not_future": {"fields": ["Year Built"], "text": f"Year Built cannot be after {CURRENT_YEAR}."},
    "year_plausible": {"fields": ["Year Built"], "text": "Year Built before 1700 is implausible."},
    "storeys_min_1": {"fields": ["Storeys"], "text": "Storeys must be at least 1."},
    "buildings_min_1": {"fields": ["Number of Buildings"], "text": "Number of Buildings must be at least 1."},
    "sprinkler_codes": {"fields": ["Fire Sprinklers (Y/N)"], "text": "Sprinkler code must be Y, N, Y13 or Y(13R)."},
    "state_2_letter": {"fields": ["State"], "text": "State should be a 2-letter abbreviation."},
    "reference_unique": {"fields": ["Reference"], "text": "Reference must be unique per row."},
}


def normalize_header(h) -> str:
    """Lower-case, strip punctuation, expand insurance abbreviations."""
    s = str(h if h is not None else "").lower().strip()
    s = s.replace("&", " and ").replace("#", " number ").replace("/", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [ABBREVIATIONS.get(t, t) for t in s.split()]
    tokens = [t for t in tokens if t not in {"of", "the"}]
    return " ".join(tokens)


def alias_index() -> dict[str, str]:
    """normalised alias -> target field (target names included)."""
    idx: dict[str, str] = {}
    for field in TARGET_FIELDS:
        idx[normalize_header(field)] = field
    idx[normalize_header("Fire Sprinklers")] = "Fire Sprinklers (Y/N)"
    for field, aliases in SYNONYMS.items():
        for a in aliases:
            idx.setdefault(normalize_header(a), field)
    return idx
