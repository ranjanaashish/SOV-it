"""Headless runner + evaluation harness.

    python evaluate.py                       # score the 3 samples + robustness files
    python evaluate.py path/to/file.xlsx     # run one file end-to-end (auto-approve), write outputs/

Auto-approval here simulates a reviewer accepting every proposal so the
pipeline can be scored; the real product never auto-approves (C-01).
Metrics: mapping accuracy (NFR-1), anomaly recall (NFR-2), audit completeness
(NFR-3), crash-free robustness (NFR-4), output schema conformance (NFR-5).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from sov_agent.agents.transformation import validate_xlsx
from sov_agent.orchestrator import Orchestrator
from sov_agent.state import SOVState

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "outputs"


def auto_review(orc: Orchestrator, state: SOVState, reviewer: str = "eval-bot") -> None:
    for _ in range(5):                      # repeat: re-reasoning/refresh can add items
        open_items = [r for r in state.active_recs() if r.is_open]
        if not open_items:
            return
        for r in open_items:
            try:
                orc.decide(state, r.id, "approve", reviewer)
            except ValueError:                # e.g. duplicate target -> exclude instead
                orc.decide(state, r.id, "edit", reviewer, edit={"target": None})


def run_file(orc: Orchestrator, path: Path, write: bool = True) -> tuple[SOVState, dict | None]:
    state = orc.intake(path.read_bytes(), path.name)
    if not state.stages["discovery"].validated:
        return state, None
    auto_review(orc, state)
    files = orc.transform(state)
    if write:
        d = OUT / path.stem
        d.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (d / name).write_bytes(data)
    return state, files


def score(state: SOVState, truth: dict) -> dict:
    got = {m.source_column: m.target for m in state.mappings}
    keys = list(truth["mapping"].keys())
    correct = sum(1 for h in keys if got.get(h) == truth["mapping"][h])
    detected = {(r, i.field, i.issue_type) for i in state.issues for r in i.affected_rows}
    detected_rows = {(r, i.field) for i in state.issues for r in i.affected_rows}
    hits = [a for a in truth["anomalies"]
            if (a["row"], a["field"], a["type"]) in detected or (a["row"], a["field"]) in detected_rows]
    misses = [a for a in truth["anomalies"] if a not in hits]
    wrong = [{"header": h, "expected": truth["mapping"][h], "got": got.get(h)} for h in keys
             if got.get(h) != truth["mapping"][h]]
    return {
        "sheet_ok": state.selected_sheet == truth["sheet"],
        "header_row_ok": state.header_row == truth["header_row"],
        "mapping_accuracy": round(correct / len(keys), 3),
        "mapping_errors": wrong,
        "anomaly_recall": round(len(hits) / max(1, len(truth["anomalies"])), 3),
        "anomaly_misses": misses,
        "rationale_coverage": round(sum(1 for r in state.active_recs() if r.rationale)
                                    / max(1, len(state.active_recs())), 3),
        "audit_entries": len(state.audit_log),
        "audit_complete": all(a.timestamp and a.rationale for a in state.audit_log),
    }


def main() -> int:
    orc = Orchestrator()
    print(f"LLM: {orc.llm.label} available={orc.llm.available()}  embeddings={orc.embedder.backend}")
    if len(sys.argv) > 1:
        st, files = run_file(orc, Path(sys.argv[1]))
        print(json.dumps(st.mapping_json, indent=2))
        print("outputs written" if files else f"failed: {[s.message for s in st.stages.values() if s.message]}")
        return 0
    truth_path = SAMPLES / "ground_truth.json"
    if not truth_path.exists():
        print("Run `python samples/make_samples.py` first.")
        return 1
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    report = {}
    for name, t in truth.items():
        t0 = time.time()
        state, files = run_file(orc, SAMPLES / name)
        r = score(state, t)
        r["output_validation"] = validate_xlsx(files["Cleaned_SOV.xlsx"]) or "passed" if files else "no output"
        r["seconds"] = round(time.time() - t0, 2)
        report[name] = r
        print(f"\n== {name}: sheet_ok={r['sheet_ok']} header_ok={r['header_row_ok']} "
              f"mapping={r['mapping_accuracy']:.0%} recall={r['anomaly_recall']:.0%} "
              f"output={r['output_validation']} ({r['seconds']}s)")
        for e in r["mapping_errors"]:
            print("   mapping miss:", e)
        for m in r["anomaly_misses"]:
            print("   anomaly miss:", m)
    for name in ("robust_empty.xlsx", "robust_summary_only.xlsx", "robust_wrong_ext.xlsx", "robust_5000_rows.xlsx"):
        p = SAMPLES / name
        if not p.exists():
            continue
        t0 = time.time()
        try:
            state, files = run_file(orc, p, write=False)
            msg = "ok" if files else next((s.message for s in state.stages.values() if s.status in ("error", "blocked")), "no output")
            report[name] = {"crashed": False, "message": msg, "seconds": round(time.time() - t0, 2)}
        except Exception as exc:
            report[name] = {"crashed": True, "message": repr(exc)}
        print(f"== {name}: {report[name]}")
    OUT.mkdir(exist_ok=True)
    (OUT / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {OUT / 'evaluation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
