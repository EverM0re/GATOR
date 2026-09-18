from types import SimpleNamespace

from file_router.schemas import EvidenceNode, SourceRef
from scripts.evaluate_validation import _oracle_groups


class NodeStore:
    def __init__(self, nodes):
        self._nodes = {node.node_id: node for node in nodes}

    def get(self, node_id):
        return self._nodes.get(node_id)

    def iter_all(self):
        return iter(self._nodes.values())


def _node(page, suffix, node_class, cost):
    return EvidenceNode(
        node_id=f"doc_p{page}_{suffix}", level=0, node_class=node_class,
        source_ref=SourceRef(doc_id="doc", doc_type="pdf", page_num=page),
        text="description", page_image_path=("page.png" if suffix == "img" else None),
        redundancy_cluster_id=f"doc:p{page}", token_equivalent_cost=cost,
    )


def test_oracle_uses_text_for_text_quote_and_image_for_visual_quote():
    nodes = NodeStore([
        _node(0, "txt", "text_span", 100),
        _node(0, "img", "pdf_page", 1000),
        _node(1, "txt", "text_span", 100),
        _node(1, "img", "pdf_page", 1000),
    ])
    qa = {
        "doc_id": "doc",
        "evidence": {
            "pages": [0, 1],
            "modalities": ["text", "table"],
            "page_modalities": {"0": ["text"], "1": ["table"]},
        },
    }
    cfg = SimpleNamespace(router=SimpleNamespace(
        max_total_cost=6000.0, max_groups=5))
    selected = _oracle_groups(qa, nodes, cfg)
    assert [(group.node_class, nodes.get(group.root_node_id).source_ref.page_num)
            for group in selected] == [("text_span", 0), ("pdf_page", 1)]
