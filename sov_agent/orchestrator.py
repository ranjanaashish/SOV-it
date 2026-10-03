"""Orchestrator: stage-gated hand-offs between the four agents + human gate.

    ingest -> Agent 1 -> Agent 2 -> Agent 3 -> [human review gate] -> Agent 4 -> export

Each agent validates its own output and sets a flag on the shared state; the
next agent refuses to start without it. The event log drives the workflow
diagram in the UI. The class is stateless apart from shared services (KG,
LLM, embedder), so many runs can be served by one process, or the methods can
be wrapped as LangGraph nodes / Celery tasks / FastAPI endpoints unchanged.
"""
from __future__ import annotations

import logging
from typing import Optional

from . import schema
from .agents.data_quality import DataQualityAgent
from .agents.reasoning import ReasoningAgent
from .agents.schema_mapping import SchemaMappingAgent
from .agents.sheet_discovery import SheetDiscoveryAgent
from .agents.transformation import TransformationAgent, export_files, validate_xlsx
from .embeddings import Embedder
from .io_utils import IngestError, read_any
from .knowledge_graph import KnowledgeGraph
from .llm import LLMClient
from .state import Change, SOVState, StageGateError, now_iso
from .text import to_display
from .values import coerce_value, norm_value

log = logging.getLogger(__name__)
APPROVE_ALL_THRESHOLD = 0.90


class Orchestrator:
    def __init__(self, llm: Optional[LLMClient] = None, embedder: Optional[Embedder] = None,
                 kg: Optional[KnowledgeGraph] = None):
        self.llm = llm or LLMClient()
        self.embedder = embedder or Embedder()
        self.kg = kg or KnowledgeGraph(embedder=self.embedder)
        self.agent1 = SheetDiscoveryAgent(kg=self.kg)
        self.agent2 = SchemaMappingAgent(kg=self.kg, embedder=self.embedder, llm=self.llm)
        self.agent3 = DataQualityAgent(kg=self.kg, llm=self.llm)
        self.reasoner = ReasoningAgent(self.agent2, llm=self.llm)
        self.agent4 = TransformationAgent()

    # ------------------------------------------------------------------ intake
    def intake(self, data: bytes, file_name: str, listener=None) -> SOVState:
        state = SOVState(file_name=file_name)
        if listener is not None:
            state.set_listener(listener)
        state.start("ingest", "Ingestion")
        try:
            state.raw_sheets, state.merged_ranges = read_any(data, file_name)
        except IngestError as exc:
            state.fail("ingest", "Ingestion", str(exc))
            return state
        except Exception as exc:  # never crash on malformed input (NFR-4)
            log.exception("ingest failed")
            state.fail("ingest", "Ingestion", f"Could not read '{file_name}': {exc}")
            return state
        state.finish("ingest", "Ingestion", True, f"{len(state.raw_sheets)} sheet(s) read")
        return self.analyse(state)

    def analyse(self, state: SOVState, sheet: Optional[str] = None, header_row: Optional[int] = None) -> SOVState:
        """(Re)run Agents 1-3. Called on intake and when the reviewer overrides sheet/header."""
        state.invalidate_from("discovery")
        state.recommendations, state.mappings, state.issues = [], [], []
        state.summary.pop("explained_ids", None)
        try:
            self.agent1.run(state, sheet_override=sheet, header_override=header_row)
            if not state.stages["discovery"].validated:
                return state
            self.agent2.run(state)
            self.agent3.run(state)
            self._update_review_gate(state)
        except StageGateError as exc:
            state.log("Orchestrator", "gate_blocked", str(exc))
        except Exception as exc:
            log.exception("analysis failed")
            stage = next((s for s, st in state.stages.items() if st.status == "running"), "quality")
            state.fail(stage, "Orchestrator", f"Unexpected error: {exc}")
        return state

    # ------------------------------------------------------------------ human gate
    def decide(self, state: SOVState, rec_id: str, decision: str, reviewer: str,
               note: Optional[str] = None, edit: Optional[dict] = None) -> SOVState:
        """decision: approve | reject | edit. Rejection requires a note and triggers re-reasoning."""
        rec = state.rec(rec_id)
        if rec.status == "superseded":
            return state
        reviewer = (reviewer or "").strip() or "reviewer"
        mapping_before = state.effective_mapping(include_pending=True)
        if decision == "approve":
            if rec.action_type == "column_mapping" and rec.proposed_target:
                clash = [s for s, t in state.effective_mapping(include_pending=False).items()
                         if t == rec.proposed_target and s != rec.source_column]
                if clash:
                    raise ValueError(f"'{rec.proposed_target}' is already approved for column '{clash[0]}'.")
            rec.status = "approved"
        elif decision == "reject":
            if not note or not note.strip():
                raise ValueError("A rejection needs a short note so the agent can re-reason.")
            rec.status, rec.reviewer_note = "rejected", note.strip()
        elif decision == "edit":
            self._apply_edit(state, rec, edit or {})
            rec.status = "edited"
        else:
            raise ValueError(f"Unknown decision '{decision}'")
        rec.decided_by, rec.decided_at = reviewer, now_iso()
        rec.history.append({"decision": {"approve": "approved", "reject": "rejected", "edit": "edited"}[decision], "by": reviewer, "at": rec.decided_at, "note": note,
                            "target": rec.proposed_target})
        state.log("Human reviewer", decision, f"{rec.title}" + (f" — {note}" if note else ""))

        if decision == "reject":
            self.reasoner.rereason(state, rec, note.strip())
        if state.effective_mapping(include_pending=True) != mapping_before:
            self._refresh_quality(state)
        self._update_review_gate(state)
        return state

    @staticmethod
    def _apply_edit(state: SOVState, rec, edit: dict) -> None:
        if rec.action_type == "column_mapping":
            target = edit.get("target")
            if target is not None and target not in schema.TARGET_FIELDS:
                raise ValueError(f"'{target}' is not a target field")
            clash = [s for s, t in state.effective_mapping(include_pending=False).items()
                     if t == target and s != rec.source_column and target]
            if clash:
                raise ValueError(f"'{target}' is already approved for column '{clash[0]}'.")
            rec.proposed_target, rec.field = target, target
            rec.title = f"Map '{rec.source_column}' → {target}" if target else f"Exclude '{rec.source_column}'"
            rec.examples = [{"before": rec.source_column, "after": target or "(excluded)"}]
            rec.rationale += f" Reviewer edited the target to {target or '(excluded)'}."
            rec.interpretation = f"Edited by reviewer: {target or 'excluded'}."
            return

        col = rec.source_column
        has_col = col in state.source_df.columns if col else False
        rows = [c.row for c in rec.changes] or next(
            (list(i.affected_rows) for i in state.issues if i.id == rec.issue_id), [])
        new: list = []
        errors: list[str] = []
        if isinstance(edit.get("value_map"), dict):
            # {original value (any case/spacing) -> new value}; applied to EVERY affected row holding it
            vmap = {norm_value(k): v for k, v in edit["value_map"].items()}
            for row in rows:
                before = state.source_df.at[row, col] if has_col else None
                key = norm_value(before)
                if key not in vmap:
                    continue
                val, err = coerce_value(rec.field, vmap[key])
                if err:
                    errors.append(err)
                    continue
                if val != before:
                    new.append(Change(row=row, before=before, after=val))
        elif isinstance(edit.get("changes"), list):
            for c in edit["changes"]:
                row = int(c["row"])
                before = state.source_df.at[row, col] if has_col else None
                val, err = coerce_value(rec.field, c.get("after"))
                if err:
                    errors.append(f"row {row}: {err}")
                    continue
                new.append(Change(row=row, before=before, after=val))
        else:
            raise ValueError("Edit needs a value_map or a list of {row, after} changes")
        if errors:
            raise ValueError("; ".join(sorted(set(errors))[:3]))
        rec.changes = new
        rec.action_type = "data_correction" if rec.action_type == "flag_for_review" else rec.action_type
        rec.title = f"Reviewer values for {rec.field} ({len(new)} cell(s))"
        rec.examples = [{"row": c.row, "before": to_display(c.before),
                         "after": "(blank)" if c.after is None else to_display(c.after)} for c in new[:5]]
        rec.interpretation = "Edited by reviewer: " + ", ".join(sorted({
            f"'{to_display(c.before)}' → {'(blank)' if c.after is None else to_display(c.after)}" for c in new})[:6])
        if "Reviewer supplied" not in rec.rationale:
            rec.rationale += " Reviewer supplied the corrected values (human-entered, validated against the data dictionary)."

    def approve_all_high(self, state: SOVState, reviewer: str, threshold: float = APPROVE_ALL_THRESHOLD) -> int:
        """Bulk-approve pending items with confidence >= threshold. High-severity flags are excluded on purpose."""
        n = 0
        for r in list(state.active_recs()):
            if r.status == "pending" and r.confidence >= threshold and not (
                    r.action_type == "flag_for_review" and r.severity == "High"):
                try:
                    self.decide(state, r.id, "approve", reviewer)
                    n += 1
                except ValueError:
                    continue
        state.log("Human reviewer", "approve_all", f"{n} item(s) >= {threshold:.2f}")
        return n

    def _refresh_quality(self, state: SOVState) -> None:
        self.agent3.run(state)   # Agent 3 re-checks against the new mapping; decisions on unchanged issues are kept

    @staticmethod
    def _update_review_gate(state: SOVState) -> None:
        open_items = [r for r in state.active_recs() if r.is_open]
        st = state.stages["review"]
        if not state.stages["quality"].validated:
            st.status, st.validated = "idle", False
            return
        if open_items:
            st.status, st.validated = "running", False
            st.message = f"{len(open_items)} item(s) awaiting review"
        else:
            st.status, st.validated = "done", True
            st.message = "All recommendations reviewed"

    def can_export(self, state: SOVState) -> tuple[bool, list[str]]:
        reasons = []
        for s in ("discovery", "mapping", "quality"):
            if not state.stages[s].validated:
                reasons.append(f"Stage '{s}' is not validated")
        pending = [r for r in state.active_recs() if r.is_open]
        if pending:
            reasons.append(f"{len(pending)} recommendation(s) still pending or escalated")
        return (not reasons), reasons

    # ------------------------------------------------------------------ transform & export
    def transform(self, state: SOVState) -> dict[str, bytes]:
        ok, reasons = self.can_export(state)
        if not ok:
            raise StageGateError("Export blocked: " + "; ".join(reasons))
        self.agent4.run(state)
        if not state.stages["transform"].validated:
            raise RuntimeError(state.stages["transform"].message)
        files = export_files(state)
        problems = validate_xlsx(files["Cleaned_SOV.xlsx"])
        state.summary["output_validation"] = problems or "passed"
        if problems:
            raise RuntimeError("Output validation failed: " + "; ".join(problems))
        learned = self.kg.learn_from_run(state)
        state.log("Knowledge graph", "learned", f"{learned} edge update(s) from reviewer sign-off")
        files = export_files(state)  # include final events in the summary
        return files
