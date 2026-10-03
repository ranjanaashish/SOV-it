"""Knowledge graph + vector memory shared by Agents 2 and 3.

The graph supplies evidence only; it never applies a change. Edge weights are
updated only after a reviewer signs off (export). It stores headers and
mapping decisions, never row values.

Node types : target_field, source_header, alias, allowed_value, rule
Edge types : ALIAS_OF, MAPS_TO, REJECTED_AS, ALLOWED_VALUE, CONSTRAINED_BY, CO_OCCURS_WITH

Backend: NetworkX in memory, persisted as node-link JSON. For multi-user
scale the same interface can be backed by Neo4j / Memgraph, and the vector
memory by Qdrant / Chroma / pgvector (see README).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from . import schema
from .state import SOVState, now_iso


def _f(name: str) -> str:
    return f"field:{name}"


def _h(norm: str) -> str:
    return f"header:{norm}"


class KnowledgeGraph:
    def __init__(self, path: str | os.PathLike | None = None, embedder=None):
        self.path = Path(path or os.getenv("KG_PATH", "data/knowledge_graph.json"))
        self.embedder = embedder
        self._lock = threading.Lock()
        self.g = nx.MultiDiGraph()
        self.vectors: list[dict] = []      # [{"text","target","vec"}] approved header memory
        if self.path.exists():
            self._load()
        self._seed()

    # ---------------------------------------------------------------- seeding
    def _seed(self) -> None:
        g = self.g
        for f in schema.TARGET_FIELDS:
            g.add_node(_f(f), kind="target_field", dtype=schema.FIELD_TYPES[f],
                       description=schema.FIELD_DESCRIPTIONS[f])
        for f, aliases in schema.SYNONYMS.items():
            for a in aliases:
                n = f"alias:{schema.normalize_header(a)}"
                g.add_node(n, kind="alias", raw=a)
                if not self._has_edge(n, _f(f), "ALIAS_OF"):
                    g.add_edge(n, _f(f), key="ALIAS_OF", type="ALIAS_OF")
        for code in schema.SPRINKLER_CODES:
            self._ensure_value("Fire Sprinklers (Y/N)", code)
        for abbr in schema.STATE_ABBRS:
            self._ensure_value("State", abbr)
        for rid, rule in schema.RULES.items():
            g.add_node(f"rule:{rid}", kind="rule", text=rule["text"])
            for f in rule["fields"]:
                if not self._has_edge(_f(f), f"rule:{rid}", "CONSTRAINED_BY"):
                    g.add_edge(_f(f), f"rule:{rid}", key="CONSTRAINED_BY", type="CONSTRAINED_BY")

    def _ensure_value(self, field: str, value: str) -> None:
        n = f"value:{field}:{value}"
        self.g.add_node(n, kind="allowed_value", value=value)
        if not self._has_edge(n, _f(field), "ALLOWED_VALUE"):
            self.g.add_edge(n, _f(field), key="ALLOWED_VALUE", type="ALLOWED_VALUE")

    def _has_edge(self, u: str, v: str, key: str) -> bool:
        return self.g.has_edge(u, v, key=key)

    # ---------------------------------------------------------------- lookups
    def lookup(self, raw_header: str) -> dict:
        """Evidence for a header: approved targets, rejected targets, explanation strings."""
        norm = schema.normalize_header(raw_header)
        node = _h(norm)
        approved: list[dict] = []
        rejected: set[str] = set()
        evidence: list[str] = []
        if self.g.has_node(node):
            for _, v, k, d in self.g.out_edges(node, keys=True, data=True):
                target = v.split(":", 1)[1]
                if k == "MAPS_TO" and d.get("approvals", 0) > 0:
                    approved.append({"target": target, "approvals": d["approvals"],
                                     "last_confidence": d.get("last_confidence")})
                    evidence.append(f"'{raw_header}' -> {target} approved in {d['approvals']} earlier submission(s)")
                elif k == "REJECTED_AS":
                    rejected.add(target)
                    evidence.append(f"Reviewer previously rejected '{raw_header}' -> {target}"
                                    + (f" (note: {d.get('last_note')})" if d.get("last_note") else ""))
        approved.sort(key=lambda x: -x["approvals"])
        return {"approved": approved, "rejected": rejected, "evidence": evidence}

    def vector_lookup(self, text: str, min_sim: float = 0.88) -> Optional[dict]:
        """Nearest approved header from earlier submissions (cross-submission memory)."""
        if not self.vectors or self.embedder is None:
            return None
        q = self.embedder.encode([text])
        mat = np.vstack([np.asarray(v["vec"], dtype=np.float32) for v in self.vectors])
        if mat.shape[1] != q.shape[1]:          # embedder backend changed
            return None
        sims = (mat @ q.T).ravel()
        i = int(np.argmax(sims))
        if sims[i] >= min_sim:
            return {"target": self.vectors[i]["target"], "similar_to": self.vectors[i]["text"],
                    "similarity": float(sims[i])}
        return None

    def neighbours(self, raw_header: str, limit: int = 5) -> list[str]:
        node = _h(schema.normalize_header(raw_header))
        if not self.g.has_node(node):
            return []
        out = []
        for _, v, k, d in self.g.out_edges(node, keys=True, data=True):
            if k == "CO_OCCURS_WITH":
                out.append((d.get("count", 1), self.g.nodes[v].get("raw", v)))
        return [n for _, n in sorted(out, reverse=True)[:limit]]

    def allowed_values(self, field: str) -> set[str]:
        return {self.g.nodes[u]["value"] for u, _, k in self.g.in_edges(_f(field), keys=True)
                if k == "ALLOWED_VALUE"}

    def rules_for(self, field: str) -> list[str]:
        return [self.g.nodes[v]["text"] for _, v, k in self.g.out_edges(_f(field), keys=True)
                if k == "CONSTRAINED_BY"]

    # ---------------------------------------------------------------- learning
    def learn_from_run(self, state: SOVState) -> int:
        """Update edges from reviewer decisions. Call only after sign-off."""
        n = 0
        with self._lock:
            headers = [m.source_column for m in state.mappings]
            for r in state.active_recs():
                if r.action_type != "column_mapping" or not r.source_column:
                    continue
                norm = schema.normalize_header(r.source_column)
                hn = _h(norm)
                self.g.add_node(hn, kind="source_header", raw=r.source_column)
                if r.status in ("approved", "edited") and r.proposed_target:
                    d = self.g.get_edge_data(hn, _f(r.proposed_target), key="MAPS_TO") or {}
                    self.g.add_edge(hn, _f(r.proposed_target), key="MAPS_TO", type="MAPS_TO",
                                    approvals=d.get("approvals", 0) + 1,
                                    last_confidence=r.confidence, last_seen=now_iso())
                    if self.embedder is not None and not any(
                            v["text"] == r.source_column and v["target"] == r.proposed_target for v in self.vectors):
                        self.vectors.append({"text": r.source_column, "target": r.proposed_target,
                                             "vec": self.embedder.encode([r.source_column])[0].tolist()})
                    n += 1
                for h in r.history:
                    if h.get("decision") == "rejected" and h.get("target"):
                        d = self.g.get_edge_data(hn, _f(h["target"]), key="REJECTED_AS") or {}
                        self.g.add_edge(hn, _f(h["target"]), key="REJECTED_AS", type="REJECTED_AS",
                                        count=d.get("count", 0) + 1, last_note=h.get("note"))
                        n += 1
            for i, a in enumerate(headers):          # adjacency in the sheet
                for b in headers[i + 1:i + 2]:
                    for x, y in ((a, b), (b, a)):
                        u, v = _h(schema.normalize_header(x)), _h(schema.normalize_header(y))
                        self.g.add_node(u, kind="source_header", raw=x)
                        self.g.add_node(v, kind="source_header", raw=y)
                        d = self.g.get_edge_data(u, v, key="CO_OCCURS_WITH") or {}
                        self.g.add_edge(u, v, key="CO_OCCURS_WITH", type="CO_OCCURS_WITH",
                                        count=d.get("count", 0) + 1)
            self.save()
        return n

    def reset(self, learned_only: bool = True) -> dict:
        """Forget learned memory. learned_only=True keeps the seeded schema, synonyms, value
        sets and rules and drops MAPS_TO / REJECTED_AS / CO_OCCURS_WITH edges, source-header
        nodes and the vector memory. learned_only=False rebuilds the graph from scratch."""
        with self._lock:
            before = self.stats()
            if learned_only:
                drop = [(u, v, k) for u, v, k in self.g.edges(keys=True)
                        if k in ("MAPS_TO", "REJECTED_AS", "CO_OCCURS_WITH")]
                self.g.remove_edges_from(drop)
                self.g.remove_nodes_from([n for n, d in list(self.g.nodes(data=True))
                                          if d.get("kind") == "source_header"])
            else:
                self.g = nx.MultiDiGraph()
            self.vectors = []
            self._seed()
            self.save()
            return {"before": before, "after": self.stats()}

    # ---------------------------------------------------------------- persistence
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"graph": nx.node_link_data(self.g, edges="edges"), "vectors": self.vectors}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.g = nx.node_link_graph(data["graph"], directed=True, multigraph=True, edges="edges")
        self.vectors = data.get("vectors", [])

    # ---------------------------------------------------------------- UI helpers
    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
        edges: dict[str, int] = {}
        for *_, k in self.g.edges(keys=True):
            edges[k] = edges.get(k, 0) + 1
        return {"nodes": kinds, "edges": edges, "vector_memory": len(self.vectors)}

    def learned_edges(self) -> list[dict]:
        rows = []
        for u, v, k, d in self.g.edges(keys=True, data=True):
            if k in ("MAPS_TO", "REJECTED_AS"):
                rows.append({"source_header": self.g.nodes[u].get("raw", u), "edge": k,
                             "target_field": v.split(":", 1)[1],
                             "approvals": d.get("approvals"), "rejections": d.get("count"),
                             "last_confidence": d.get("last_confidence"), "note": d.get("last_note")})
        return rows

    def to_dot(self, max_headers: int = 40) -> str:
        lines = [
            "digraph KG {",
            "rankdir=LR;",
            'bgcolor="transparent";',
            'node [shape=box, style="filled,rounded", fontname="Montserrat", fontsize=10, fillcolor="#223859", fontcolor="#FFFFFF", color="#7bdcb5"];',
            'edge [fontname="Montserrat", fontsize=9, fontcolor="#B8C8DA", color="#7bdcb5"];',
        ]
        for f in schema.TARGET_FIELDS:
            lines.append(f'"{f}" [shape=ellipse, fillcolor="#1a2c47", fontcolor="#7bdcb5", color="#7bdcb5", penwidth=2];')
        shown = 0
        for u, v, k, d in self.g.edges(keys=True, data=True):
            if k not in ("MAPS_TO", "REJECTED_AS") or shown >= max_headers:
                continue
            raw = str(self.g.nodes[u].get("raw", u)).replace('"', "'")
            tgt = v.split(":", 1)[1]
            color = '#10B981' if k == "MAPS_TO" else '#EF4444'
            style = 'solid' if k == "MAPS_TO" else 'dashed'
            label = f"x{d.get('approvals', d.get('count', 1))}"
            lines.append(f'"{raw}" -> "{tgt}" [color="{color}", style={style}, label="{label}"];')
            shown += 1
        lines.append("}")
        return "\n".join(lines)
