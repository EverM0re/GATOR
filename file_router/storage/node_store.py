"""Persistent key-value store for EvidenceNodes (SQLite + JSON blob)."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, Iterable, List, Optional

from ..schemas import EvidenceNode


class NodeStore:
    def __init__(self, store_dir: str):
        os.makedirs(store_dir, exist_ok=True)
        self.path = os.path.join(store_dir, "nodes.sqlite")
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS nodes (
                   node_id   TEXT PRIMARY KEY,
                   level     INTEGER,
                   node_class TEXT,
                   doc_id    TEXT,
                   blob      TEXT
               )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_doc ON nodes(doc_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON nodes(level)")
        self._conn.commit()

    def put(self, node: EvidenceNode) -> None:
        blob = json.dumps(node.to_dict())
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes(node_id, level, node_class, doc_id, blob) VALUES (?,?,?,?,?)",
            (node.node_id, node.level, node.node_class, node.source_ref.doc_id, blob),
        )
        self._conn.commit()

    def put_many(self, nodes: Iterable[EvidenceNode]) -> None:
        rows = [
            (n.node_id, n.level, n.node_class, n.source_ref.doc_id, json.dumps(n.to_dict()))
            for n in nodes
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO nodes(node_id, level, node_class, doc_id, blob) VALUES (?,?,?,?,?)",
            rows,
        )
        self._conn.commit()

    def get(self, node_id: str) -> Optional[EvidenceNode]:
        row = self._conn.execute(
            "SELECT blob FROM nodes WHERE node_id=?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return EvidenceNode.from_dict(json.loads(row[0]))

    def get_many(self, node_ids: List[str]) -> Dict[str, EvidenceNode]:
        if not node_ids:
            return {}
        placeholders = ",".join("?" * len(node_ids))
        rows = self._conn.execute(
            f"SELECT node_id, blob FROM nodes WHERE node_id IN ({placeholders})",
            node_ids,
        ).fetchall()
        return {nid: EvidenceNode.from_dict(json.loads(b)) for nid, b in rows}

    def iter_all(self) -> Iterable[EvidenceNode]:
        for row in self._conn.execute("SELECT blob FROM nodes"):
            yield EvidenceNode.from_dict(json.loads(row[0]))

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def close(self):
        self._conn.close()
