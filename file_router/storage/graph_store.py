"""Multimodal knowledge graph — networkx DiGraph with typed edges.

Edges carry `edge_type` and `traversal_cost` so the router can reason about
graph-hop cost if we ever extend selection to multi-hop evidence gathering.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import networkx as nx


EDGE_TYPES = (
    "part_of",          # region → page, page → section, section → doc
    "summary_of",       # summary → original
    "linked_image",     # image → parent text chunk
    "mentions",         # text → entity
    "depicts",          # image → entity
    "co_occurs",        # entity ↔ entity in same page/chunk
    "expansion_of",     # finer-grained node → coarser sibling
)


class GraphStore:
    def __init__(self, store_dir: str):
        self.path = os.path.join(store_dir, "mmkg.pkl")
        self._g: nx.DiGraph = nx.DiGraph()
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self._g = pickle.load(f)

    @property
    def graph(self) -> nx.DiGraph:
        return self._g

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self._g, f)

    def add_node(self, node_id: str, **attrs):
        self._g.add_node(node_id, **attrs)

    def add_edge(self, src: str, dst: str, edge_type: str, traversal_cost: float = 1.0,
                 mandatory_for_group: bool = False, **attrs):
        self._g.add_edge(src, dst, edge_type=edge_type,
                         traversal_cost=traversal_cost,
                         mandatory_for_group=mandatory_for_group, **attrs)

    def neighbors(self, node_id: str, edge_types: Optional[List[str]] = None) -> List[str]:
        if node_id not in self._g:
            return []
        out = []
        for _, nbr, data in self._g.out_edges(node_id, data=True):
            if edge_types is None or data.get("edge_type") in edge_types:
                out.append(nbr)
        for nbr, _, data in self._g.in_edges(node_id, data=True):
            if edge_types is None or data.get("edge_type") in edge_types:
                out.append(nbr)
        return list(dict.fromkeys(out))

    def expand_one_hop(self, node_ids: List[str],
                       edge_types: Optional[List[str]] = None) -> List[str]:
        out = set()
        for nid in node_ids:
            out.update(self.neighbors(nid, edge_types))
        return list(out - set(node_ids))
