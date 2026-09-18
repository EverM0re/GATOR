"""Core data structures for CNEM-V.

The central atom is `EvidenceNode`: a single memory entry at a single
granularity. Nodes cover all modalities (text, image, pdf page, pdf region,
audio, video) and multiple levels (0=raw ... 3=doc/theme). Cost is baked in
at ingestion, not recomputed at query time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


NODE_CLASSES = (
    "text_span",
    "text_cluster",
    "image",
    "pdf_page",
    "pdf_region",
    "pdf_section",
    "doc_summary",
)


@dataclass
class SourceRef:
    doc_id: str
    doc_type: str                         # "text" | "image" | "pdf"
    page_num: Optional[int] = None
    bbox: Optional[List[float]] = None    # [x0,y0,x1,y1] for pdf_region
    char_range: Optional[List[int]] = None


@dataclass
class EvidenceNode:
    node_id: str
    level: int                            # 0=raw, 1=local cluster/caption, 2=section, 3=doc
    node_class: str                       # see NODE_CLASSES
    source_ref: SourceRef

    # Content (only one or two of these are populated depending on node_class)
    text: str = ""                        # raw text, caption, or summary
    image_path: Optional[str] = None      # absolute path on disk
    page_image_path: Optional[str] = None

    # Embeddings (stored as list[float]; None if not applicable)
    text_embedding: Optional[List[float]] = None
    visual_embedding: Optional[List[float]] = None           # single-vector (CLIP) pooled form
    visual_multi_vector: Optional[List[List[float]]] = None  # multi-vector (ColPali)

    # Hierarchy links
    summary_of: List[str] = field(default_factory=list)      # node this summarizes
    expansions: List[str] = field(default_factory=list)      # finer-grained siblings
    parent_summary_id: Optional[str] = None

    # Cost (pre-computed at ingestion)
    text_token_cost: float = 0.0
    visual_token_cost: float = 0.0
    token_equivalent_cost: float = 0.0

    # Redundancy / coverage signals (pre-computed)
    redundancy_cluster_id: str = ""
    coverage_signature: Optional[List[float]] = None         # compact vector for submod

    # Hard dependencies
    mandatory_companions: List[str] = field(default_factory=list)

    # Bookkeeping
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvidenceNode":
        src = d.pop("source_ref")
        if isinstance(src, dict):
            src = SourceRef(**src)
        return EvidenceNode(source_ref=src, **d)


@dataclass
class ModalityBias:
    """Query-time signal: which modality does the question want?"""
    active_modality: str = "none"         # text | image | pdf | none
    confidence: float = 0.0
    apply_bias: bool = False


@dataclass
class EvidenceGroup:
    """What the router actually ranks: a node + its mandatory companions."""
    group_id: str
    root_node_id: str
    node_ids: List[str]                   # root + mandatory companions
    node_class: str
    level: int
    redundancy_cluster_id: str

    base_score: float = 0.0
    modality_bias_score: float = 0.0
    router_score: float = 0.0
    router_probability: float = 0.0

    token_equivalent_cost: float = 0.0
    coverage_signature: Optional[List[float]] = None
    source_doc_id: str = ""


@dataclass
class RouterResult:
    question: str
    answer: str
    selected_groups: List[str]
    candidate_count: int
    strategy: str
    total_cost: float
    modality_bias: Dict[str, Any]
    fell_back: bool = False
