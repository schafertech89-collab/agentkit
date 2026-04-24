"""Kuzu graph store.

Embedded Kuzu database. The schema maps trading-domain entities:

    (:Asset)-[:LISTED_ON]->(:Venue)
    (:Asset)-[:CORRELATED_WITH {rho}]->(:Asset)
    (:Decision)-[:ABOUT]->(:Asset)
    (:Decision)-[:PLACED_ON]->(:Venue)
    (:Decision)-[:CAUSED]->(:Outcome)
    (:Prediction)-[:SUBMITTED_TO]->(:AlloraTopic)
    (:MemCube)-[:DERIVED_FROM]->(:MemCube)

Kuzu is optional — when the package is missing the store falls back to an
in-memory adjacency list so the rest of MATS keeps running.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_STATEMENTS = [
    "CREATE NODE TABLE IF NOT EXISTS Asset(symbol STRING PRIMARY KEY, name STRING, tags STRING)",
    "CREATE NODE TABLE IF NOT EXISTS Venue(name STRING PRIMARY KEY)",
    "CREATE NODE TABLE IF NOT EXISTS Decision(id STRING PRIMARY KEY, stance DOUBLE, ts DOUBLE, payload STRING)",
    "CREATE NODE TABLE IF NOT EXISTS Outcome(id STRING PRIMARY KEY, pnl DOUBLE, ts DOUBLE)",
    "CREATE NODE TABLE IF NOT EXISTS Prediction(id STRING PRIMARY KEY, topic_id INT64, value DOUBLE, ts DOUBLE)",
    "CREATE NODE TABLE IF NOT EXISTS AlloraTopic(topic_id INT64 PRIMARY KEY, name STRING)",
    "CREATE NODE TABLE IF NOT EXISTS MemCube(id STRING PRIMARY KEY, title STRING, tier STRING)",
    "CREATE REL TABLE IF NOT EXISTS LISTED_ON(FROM Asset TO Venue)",
    "CREATE REL TABLE IF NOT EXISTS CORRELATED_WITH(FROM Asset TO Asset, rho DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS ABOUT(FROM Decision TO Asset)",
    "CREATE REL TABLE IF NOT EXISTS PLACED_ON(FROM Decision TO Venue)",
    "CREATE REL TABLE IF NOT EXISTS CAUSED(FROM Decision TO Outcome)",
    "CREATE REL TABLE IF NOT EXISTS SUBMITTED_TO(FROM Prediction TO AlloraTopic)",
    "CREATE REL TABLE IF NOT EXISTS DERIVED_FROM(FROM MemCube TO MemCube, confidence DOUBLE)",
]


class KuzuStore:
    def __init__(self, db_path: str | os.PathLike = "./vault/.graph/kuzu.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._inmem: dict[str, list[tuple]] = {}
        self._try_connect()

    def _try_connect(self) -> None:
        try:
            import kuzu
            self._db = kuzu.Database(str(self.db_path))
            self._conn = kuzu.Connection(self._db)
            for stmt in SCHEMA_STATEMENTS:
                try:
                    self._conn.execute(stmt)
                except Exception:
                    pass
        except Exception:
            self._conn = None

    # ------------------------------------------------------------ CRUD

    def ensure_asset(self, symbol: str, name: str = "", tags: list[str] | None = None) -> None:
        tags_s = ",".join(tags or [])
        if self._conn is not None:
            try:
                self._conn.execute(
                    "MERGE (a:Asset {symbol: $s}) ON CREATE SET a.name = $n, a.tags = $t",
                    {"s": symbol, "n": name, "t": tags_s},
                )
                return
            except Exception:
                pass
        self._inmem.setdefault("Asset", []).append((symbol, name, tags_s))

    def ensure_venue(self, name: str) -> None:
        if self._conn is not None:
            try:
                self._conn.execute("MERGE (v:Venue {name: $n})", {"n": name})
                return
            except Exception:
                pass
        self._inmem.setdefault("Venue", []).append((name,))

    def add_decision(self, *, decision_id: str, stance: float, ts: float,
                     symbol: str, venue: str, payload: str = "") -> None:
        self.ensure_asset(symbol)
        self.ensure_venue(venue)
        if self._conn is not None:
            try:
                self._conn.execute(
                    "MERGE (d:Decision {id: $id}) ON CREATE SET d.stance = $s, "
                    "d.ts = $t, d.payload = $p",
                    {"id": decision_id, "s": stance, "t": ts, "p": payload},
                )
                self._conn.execute(
                    "MATCH (d:Decision {id: $id}), (a:Asset {symbol: $s}) "
                    "MERGE (d)-[:ABOUT]->(a)",
                    {"id": decision_id, "s": symbol},
                )
                self._conn.execute(
                    "MATCH (d:Decision {id: $id}), (v:Venue {name: $n}) "
                    "MERGE (d)-[:PLACED_ON]->(v)",
                    {"id": decision_id, "n": venue},
                )
                return
            except Exception:
                pass
        self._inmem.setdefault("Decision", []).append(
            (decision_id, stance, ts, symbol, venue, payload)
        )

    def add_outcome(self, *, decision_id: str, pnl: float, ts: float) -> None:
        outcome_id = f"outcome:{decision_id}"
        if self._conn is not None:
            try:
                self._conn.execute(
                    "MERGE (o:Outcome {id: $id}) ON CREATE SET o.pnl = $p, o.ts = $t",
                    {"id": outcome_id, "p": pnl, "t": ts},
                )
                self._conn.execute(
                    "MATCH (d:Decision {id: $d}), (o:Outcome {id: $o}) "
                    "MERGE (d)-[:CAUSED]->(o)",
                    {"d": decision_id, "o": outcome_id},
                )
                return
            except Exception:
                pass
        self._inmem.setdefault("Outcome", []).append((outcome_id, pnl, ts, decision_id))

    def add_prediction(self, *, prediction_id: str, topic_id: int, value: float,
                        ts: float, topic_name: str = "") -> None:
        if self._conn is not None:
            try:
                self._conn.execute(
                    "MERGE (t:AlloraTopic {topic_id: $t}) ON CREATE SET t.name = $n",
                    {"t": topic_id, "n": topic_name},
                )
                self._conn.execute(
                    "MERGE (p:Prediction {id: $id}) ON CREATE SET p.topic_id = $t, "
                    "p.value = $v, p.ts = $ts",
                    {"id": prediction_id, "t": topic_id, "v": value, "ts": ts},
                )
                self._conn.execute(
                    "MATCH (p:Prediction {id: $p}), (t:AlloraTopic {topic_id: $t}) "
                    "MERGE (p)-[:SUBMITTED_TO]->(t)",
                    {"p": prediction_id, "t": topic_id},
                )
                return
            except Exception:
                pass
        self._inmem.setdefault("Prediction", []).append(
            (prediction_id, topic_id, value, ts, topic_name)
        )

    def correlate(self, a: str, b: str, rho: float) -> None:
        self.ensure_asset(a); self.ensure_asset(b)
        if self._conn is not None:
            try:
                self._conn.execute(
                    "MATCH (x:Asset {symbol: $a}), (y:Asset {symbol: $b}) "
                    "MERGE (x)-[r:CORRELATED_WITH]->(y) SET r.rho = $r",
                    {"a": a, "b": b, "r": rho},
                )
                return
            except Exception:
                pass
        self._inmem.setdefault("CORRELATED_WITH", []).append((a, b, rho))

    # ------------------------------------------------------------ queries

    def decisions_for(self, symbol: str, limit: int = 32) -> list[dict]:
        if self._conn is not None:
            try:
                res = self._conn.execute(
                    "MATCH (d:Decision)-[:ABOUT]->(a:Asset {symbol: $s}) "
                    "RETURN d.id, d.stance, d.ts ORDER BY d.ts DESC LIMIT $l",
                    {"s": symbol, "l": limit},
                )
                return [{"id": r[0], "stance": r[1], "ts": r[2]} for r in res]
            except Exception:
                pass
        decisions = self._inmem.get("Decision", [])
        filtered = [d for d in decisions if d[3] == symbol]
        filtered.sort(key=lambda d: d[2], reverse=True)
        return [{"id": d[0], "stance": d[1], "ts": d[2]} for d in filtered[:limit]]

    def multihop(self, symbol: str, hops: int = 2) -> list[dict]:
        if self._conn is not None:
            try:
                res = self._conn.execute(
                    f"MATCH p = (a:Asset {{symbol: $s}})-[*1..{hops}]-(n) RETURN p LIMIT 32",
                    {"s": symbol},
                )
                return [{"path": str(r)} for r in res]
            except Exception:
                pass
        return []
