"""Generate 3 synthetic SOV test files with planted anomalies + ground truth.

    python samples/make_samples.py            -> samples/*.xlsx + samples/ground_truth.json

Mirrors the hackathon sample design:
  1. sample1_clean.xlsx      clean, well-structured headers (baseline)
  2. sample2_abbrev.xlsx     ambiguous / abbreviated headers, title rows, header on row 4
  3. sample3_multisheet.xlsx multi-sheet, merged group header, missing columns, totals row
Plus robustness files: empty sheet, summary-only workbook, wrong extension, 5,000 rows.
All names/addresses are synthetic.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from openpyxl import Workbook

OUT = Path(__file__).parent
random.seed(42)

CITIES = [("Austin", "TX", "Travis", 78701), ("Miami", "FL", "Miami-Dade", 33101), ("Denver", "CO", "Denver", 80202),
          ("Boston", "MA", "Suffolk", 2108), ("Phoenix", "AZ", "Maricopa", 85004), ("Atlanta", "GA", "Fulton", 30303),
          ("Columbus", "OH", "Franklin", 43215), ("Newark", "NJ", "Essex", 7102), ("Tampa", "FL", "Hillsborough", 33602),
          ("Dallas", "TX", "Dallas", 75201)]
STATE_NAMES = {"TX": "Texas", "FL": "Florida", "CO": "Colorado", "MA": "Massachusetts", "AZ": "Arizona",
               "GA": "Georgia", "OH": "Ohio", "NJ": "New Jersey"}
STREETS = ["Oak St", "Maple Ave", "Cedar Rd", "Pine Blvd", "Elm Dr", "Birch Ln", "Harbor Way", "Ridge Ct"]
OCC = ["Apartments", "Office", "Retail", "Warehouse", "Light Manufacturing", "Hotel"]
CONST = ["Frame", "Joisted Masonry", "Non-Combustible", "Masonry Non-Combustible", "Fire Resistive"]


def base_rows(n: int, prefix: str):
    rows = []
    for i in range(n):
        city, st, county, z = random.choice(CITIES)
        rows.append({
            "Reference": f"{prefix}-{i + 1:03d}",
            "Address": f"{random.randint(10, 9999)} {random.choice(STREETS)}",
            "City": city, "State": st, "Zip": z, "County": county, "Country": "USA",
            "Building Value": float(random.randint(5, 400) * 25_000),
            "Contents": float(random.randint(1, 80) * 10_000),
            "BI": float(random.randint(0, 50) * 10_000),
            "Occupancy": random.choice(OCC), "Construction": random.choice(CONST),
            "Storeys": random.randint(1, 12), "Number of Buildings": random.randint(1, 4),
            "Year Built": random.randint(1950, 2022),
            "Fire Sprinklers (Y/N)": random.choice(["Y", "N", "Y13", "Y(13R)"]),
            "Other": float(random.choice([0, 0, 15_000, 50_000])),
        })
    return rows


def plant(rows, start_row, plan, anomalies):
    """plan: list of (row_index, field, value, anomaly_type). Excel row = start_row + row_index."""
    for idx, field, value, kind in plan:
        rows[idx][field] = value
        anomalies.append({"row": start_row + idx, "field": field, "type": kind, "value": value})


def write(ws, start_row, headers, rows, order):
    for j, h in enumerate(headers, start=1):
        ws.cell(row=start_row, column=j, value=h)
    for i, r in enumerate(rows, start=1):
        for j, f in enumerate(order, start=1):
            if f is None:
                v = None
            elif f.startswith("="):
                v = r.get(f[1:])
            else:
                v = r.get(f)
            ws.cell(row=start_row + i, column=j, value=v)


def sample1(truth):
    rows = base_rows(40, "LOC")
    anomalies = []
    plant(rows, 2, [
        (3, "Building Value", -250000.0, "negative_value"),
        (7, "Year Built", 2091, "future_year"),
        (11, "Contents", "$125,000", "currency_format"),
        (15, "Storeys", 0, "below_min"),
        (18, "Fire Sprinklers (Y/N)", "Yes", "sprinkler_variant"),
        (22, "City", None, "missing"),
        (25, "Zip", "N/A", "placeholder"),
        (30, "State", "Texas", "state_full_name"),
    ], anomalies)
    headers = ["Location Number", "Street Address", "City", "State", "Zip Code", "County", "Country",
               "Building Value", "Contents Value", "Business Income", "Occupancy", "Construction Type",
               "Number of Stories", "Number of Buildings", "Year Built", "Sprinklered", "Other Value", "TIV"]
    for r in rows:
        r["TIV"] = None
    order = ["Reference", "Address", "City", "State", "Zip", "County", "Country", "Building Value", "Contents", "BI",
             "Occupancy", "Construction", "Storeys", "Number of Buildings", "Year Built", "Fire Sprinklers (Y/N)",
             "Other", "TIV"]
    wb = Workbook()
    ws = wb.active
    ws.title = "SOV"
    write(ws, 1, headers, rows, order)
    for i in range(len(rows)):  # TIV formula-free total so data_only reads a value
        r = rows[i]
        tiv = sum(x for x in (r["Building Value"], r["Contents"], r["BI"], r["Other"]) if isinstance(x, float))
        ws.cell(row=2 + i, column=18, value=tiv)
    wb.save(OUT / "sample1_clean.xlsx")
    truth["sample1_clean.xlsx"] = {
        "sheet": "SOV", "header_row": 1,
        "mapping": dict(zip(headers, order[:-1] + [None])), "anomalies": anomalies}


def sample2(truth):
    rows = base_rows(60, "P")
    anomalies = []
    for r in rows:  # make it messy: full state names, $ strings, Yes/No sprinklers, zip+4
        r["State"] = STATE_NAMES.get(r["State"], r["State"]) if random.random() < 0.3 else r["State"]
    plant(rows, 5, [
        (2, "Building Value", "$1,250,000", "currency_format"),
        (5, "Building Value", -75000.0, "negative_value"),
        (9, "Year Built", 2035, "future_year"),
        (12, "Fire Sprinklers (Y/N)", "13R", "sprinkler_variant"),
        (14, "Fire Sprinklers (Y/N)", "Partial", "sprinkler_invalid"),
        (17, "Zip", "78701-1234", "zip_plus4"),
        (20, "Storeys", "3 floors", "text_number"),
        (24, "Contents", "TBD", "placeholder"),
        (28, "Number of Buildings", 0, "below_min"),
        (33, "Year Built", "unknown", "placeholder"),
        (37, "BI", "abc", "type_error"),
        (41, "Address", None, "missing"),
        (44, "Reference", "P-001", "duplicate_reference"),
    ], anomalies)
    for a in anomalies:
        if a["type"] == "duplicate_reference":
            anomalies.append({"row": 5, "field": "Reference", "type": "duplicate_reference", "value": "P-001"})
            break
    headers = ["Loc #", "Addr 1", "Town", "St", "Postal", "Parish/County", "Cntry", "Insured Building Amount",
               "BPP", "Income Loss Exposure", "Occ Desc", "Const Class", "# Flrs", "No. Bldgs", "Year of Erection",
               "Fire Prot.", "Misc TIV", "Flood Zone", "EQ Deductible"]
    for r in rows:
        r["Flood"] = random.choice(["X", "AE", "A", "X500"])
        r["EQD"] = random.choice([0.05, 0.1])
    order = ["Reference", "Address", "City", "State", "Zip", "County", "Country", "Building Value", "Contents", "BI",
             "Occupancy", "Construction", "Storeys", "Number of Buildings", "Year Built", "Fire Sprinklers (Y/N)",
             "Other", "Flood", "EQD"]
    wb = Workbook()
    ws = wb.active
    ws.title = "Location Schedule"
    ws["A1"] = "Statement of Values — Redwood Housing LLC (synthetic)"
    ws["A2"] = "Policy: SAMPLE-UMR-0002   Valuation date: 2026-01-01"
    write(ws, 4, headers, rows, order)
    wb.save(OUT / "sample2_abbrev.xlsx")
    truth["sample2_abbrev.xlsx"] = {
        "sheet": "Location Schedule", "header_row": 4,
        "mapping": dict(zip(headers, order[:17] + [None, None])), "anomalies": anomalies}


def sample3(truth):
    rows = base_rows(35, "BLD")
    anomalies = []
    plant(rows, 7, [
        (1, "Building Value", None, "missing"),
        (4, "Year Built", 1650, "implausible_year"),
        (8, "Storeys", 2.5, "non_integer"),
        (10, "Building Value", "1.2M", "magnitude_suffix"),
        (13, "Fire Sprinklers (Y/N)", "no", "sprinkler_variant"),
        (19, "Contents", "-", "placeholder"),
        (23, "City", "  Miami ", "whitespace"),
        (27, "Year Built", None, "missing"),
    ], anomalies)
    wb = Workbook()
    cover = wb.active
    cover.title = "Cover"
    cover["A1"] = "Property Submission"
    cover["A3"] = "Broker"; cover["B3"] = "[REDACTED]"
    cover["A4"] = "Contact"; cover["B4"] = "sample@example.com"
    cover["A6"] = "Notes: values in USD; see Property Schedule tab."
    summ = wb.create_sheet("Summary")
    summ.append(["State", "Locations", "TIV"])
    for s in ("TX", "FL", "CO"):
        summ.append([s, random.randint(3, 12), random.randint(1, 9) * 1_000_000])
    ws = wb.create_sheet("Property Schedule")
    ws["A1"] = "Schedule of Locations"
    ws["A3"] = "As of 2026-01-01"
    # group header row 5 with merged cells, real header row 6 (County, Country, BI, Other are absent)
    ws["A5"] = "Location"; ws.merge_cells("A5:E5")
    ws["F5"] = "Values (USD)"; ws.merge_cells("F5:G5")
    ws["H5"] = "COPE"; ws.merge_cells("H5:L5")
    headers = ["Bldg ID", "Property Address", "City", "State", "ZIP", "Bldg Value", "Contents",
               "Occupancy Type", "Construction", "Stories", "Yr Built", "Sprinklers"]
    order = ["Reference", "Address", "City", "State", "Zip", "Building Value", "Contents", "Occupancy",
             "Construction", "Storeys", "Year Built", "Fire Sprinklers (Y/N)"]
    write(ws, 6, headers, rows, order)
    total_row = 6 + len(rows) + 1
    ws.cell(row=total_row, column=1, value="Total")
    ws.cell(row=total_row, column=6, value=sum(r["Building Value"] for r in rows if isinstance(r["Building Value"], float)))
    anomalies.append({"row": total_row, "field": "(row)", "type": "total_row", "value": "Total"})
    lists = wb.create_sheet("Lists")
    lists.append(["Construction codes"])
    for c in CONST:
        lists.append([c])
    wb.save(OUT / "sample3_multisheet.xlsx")
    truth["sample3_multisheet.xlsx"] = {
        "sheet": "Property Schedule", "header_row": 6,
        "mapping": dict(zip(headers, order)), "anomalies": anomalies}


def robustness():
    wb = Workbook(); wb.active.title = "Empty"; wb.save(OUT / "robust_empty.xlsx")
    wb = Workbook(); ws = wb.active; ws.title = "Summary"; ws.append(["Total TIV", 123456789]); wb.save(OUT / "robust_summary_only.xlsx")
    (OUT / "robust_wrong_ext.xlsx").write_text("this is not a spreadsheet", encoding="utf-8")
    rows = base_rows(5000, "BIG")
    wb = Workbook(); ws = wb.active; ws.title = "SOV"
    order = list(rows[0].keys())
    write(ws, 1, order, rows, order)
    wb.save(OUT / "robust_5000_rows.xlsx")


if __name__ == "__main__":
    truth: dict = {}
    sample1(truth); sample2(truth); sample3(truth); robustness()
    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=2, default=str), encoding="utf-8")
    print("Wrote samples to", OUT)
