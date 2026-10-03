"""Shared, typed state object passed between all agents (NFR-7).

Each stage sets `validated=True` only after its own output checks pass; the
next agent calls `state.require(<stage>)` and refuses to run otherwise.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

STAGES = ["ingest", "discovery", "mapping", "quality", "review", "transform"]

ActionType = Literal["column_mapping", "data_correction", "standardisation", "flag_for_review"]
Status = Literal["pending", "approved", "rejected", "edited", "escalated", "superseded"]


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def stable_id(*parts: Any) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


class StageGateError(RuntimeError):
    """Raised when an agent is asked to run on unvalidated upstream output."""


class Event(BaseModel):
    ts: str = Field(default_factory=now_iso)
    agent: str
    event: str
    detail: str = ""


class StageStatus(BaseModel):
    name: str
    status: Literal["idle", "running", "done", "blocked", "error"] = "idle"
    validated: bool = False
    message: str = ""


class SheetInfo(BaseModel):
    sheet_name: str
    classification: Literal["Primary", "Secondary", "Reject"]
    confidence: float
    header_row: Optional[int] = None          # 1-based Excel row number
    n_rows: int = 0
    n_cols: int = 0
    null_ratio: float = 1.0
    reasoning: list[str] = []
    header_candidates: list[dict] = []


class Mapping(BaseModel):
    source_column: str
    target: Optional[str] = None
    confidence: float = 0.0
    method: str = "none"                      # memory|exact|synonym|fuzzy|semantic|llm|value_profile|none
    rationale: str = ""
    uncertainty: str = ""
    alternatives: list[dict] = []
    flag: Optional[str] = None
    kg_evidence: list[str] = []
    sample_values: list[str] = []


class Change(BaseModel):
    row: int                                  # Excel row number in the source sheet
    before: Any = None
    after: Any = None


class Issue(BaseModel):
    id: str
    issue_type: str
    field: str
    source_column: Optional[str] = None
    severity: Literal["High", "Medium", "Low"] = "Medium"
    affected_rows: list[int] = []
    count: int = 0
    description: str = ""
    examples: list[dict] = []


class Recommendation(BaseModel):
    id: str
    action_type: ActionType
    title: str
    field: Optional[str] = None               # target field
    source_column: Optional[str] = None
    proposed_target: Optional[str] = None     # column_mapping only
    issue_id: Optional[str] = None
    issue_type: Optional[str] = None
    changes: list[Change] = []                # row-level proposal (corrections/standardisation)
    examples: list[dict] = []                 # [{"before":..,"after":..}]
    rationale: str = ""
    uncertainty: str = ""
    confidence: float = 0.0
    severity: Literal["High", "Medium", "Low"] = "Medium"
    method: str = ""
    kg_evidence: list[str] = []
    status: Status = "pending"
    reviewer_note: Optional[str] = None
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    revision: int = 0
    question: Optional[str] = None            # explicit question when escalated
    interpretation: Optional[str] = None      # how the agent understood the reviewer's note
    history: list[dict] = []

    @property
    def is_open(self) -> bool:
        return self.status in ("pending", "escalated")


class AuditEntry(BaseModel):
    source_column: Optional[str]
    target_column: Optional[str]
    transformation_applied: str
    row: Optional[int] = None
    before_value: Any = None
    after_value: Any = None
    confidence: Optional[float] = None
    approved_by: Optional[str] = None
    timestamp: str = Field(default_factory=now_iso)
    rationale: str = ""
    recommendation_id: Optional[str] = None


class SOVState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    file_name: str = ""
    created_at: str = Field(default_factory=now_iso)

    # DataFrames are carried in-memory only and never serialised.
    raw_sheets: dict[str, Any] = Field(default_factory=dict, exclude=True)
    merged_ranges: dict[str, list[str]] = {}
    source_df: Any = Field(default=None, exclude=True)     # Primary sheet, headers applied, index = Excel row
    output_df: Any = Field(default=None, exclude=True)

    sheet_manifest: list[SheetInfo] = []
    selected_sheet: Optional[str] = None
    header_row: Optional[int] = None

    mappings: list[Mapping] = []
    mapping_json: dict = {}
    quality: dict = {}                         # completeness, issues, dq_score
    issues: list[Issue] = []
    recommendations: list[Recommendation] = []
    audit_log: list[AuditEntry] = []
    summary: dict = {}
    events: list[Event] = []
    stages: dict[str, StageStatus] = Field(
        default_factory=lambda: {s: StageStatus(name=s) for s in STAGES}
    )

    # UI hook: called after every event so a live view can re-render (never serialised).
    _listener: Any = PrivateAttr(default=None)

    def set_listener(self, fn) -> None:
        self._listener = fn

    # ---- helpers -----------------------------------------------------------
    def log(self, agent: str, event: str, detail: str = "") -> None:
        self.events.append(Event(agent=agent, event=event, detail=detail))
        if self._listener is not None:
            try:
                self._listener(self)
            except Exception:  # a broken UI hook must never break the pipeline
                pass

    def start(self, stage: str, agent: str) -> None:
        self.stages[stage].status = "running"
        self.stages[stage].validated = False
        self.log(agent, "started", stage)

    def finish(self, stage: str, agent: str, validated: bool, message: str = "") -> None:
        st = self.stages[stage]
        st.status = "done" if validated else "blocked"
        st.validated = validated
        st.message = message
        self.log(agent, "validated" if validated else "blocked", message)

    def fail(self, stage: str, agent: str, message: str) -> None:
        st = self.stages[stage]
        st.status, st.validated, st.message = "error", False, message
        self.log(agent, "error", message)

    def require(self, *stages: str) -> None:
        for s in stages:
            if not self.stages[s].validated:
                raise StageGateError(f"Stage '{s}' has not been validated; refusing to run.")

    def invalidate_from(self, stage: str) -> None:
        idx = STAGES.index(stage)
        for s in STAGES[idx:]:
            self.stages[s] = StageStatus(name=s)

    def rec(self, rec_id: str) -> Recommendation:
        for r in self.recommendations:
            if r.id == rec_id:
                return r
        raise KeyError(rec_id)

    def active_recs(self) -> list[Recommendation]:
        return [r for r in self.recommendations if r.status != "superseded"]

    def effective_mapping(self, include_pending: bool = True) -> dict[str, str]:
        """source_column -> target from column_mapping recommendations."""
        out: dict[str, str] = {}
        for r in self.active_recs():
            if r.action_type != "column_mapping" or not r.source_column or not r.proposed_target:
                continue
            ok = r.status in ("approved", "edited") or (include_pending and r.status in ("pending", "escalated"))
            if ok:
                out[r.source_column] = r.proposed_target
        return out
