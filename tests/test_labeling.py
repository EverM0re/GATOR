from types import SimpleNamespace

from file_router.schemas import EvidenceGroup, EvidenceNode, SourceRef
from file_router.training.labeling import AutoLabeler


class NodeStore:
    def __init__(self, nodes):
        self.nodes = {node.node_id: node for node in nodes}

    def get(self, node_id):
        return self.nodes.get(node_id)


def _cfg():
    return SimpleNamespace(
        labeling=SimpleNamespace(
            mode="evidence", answer_match="substring", token_f1_threshold=0.5),
        cost=SimpleNamespace(max_cost_normalizer=6000.0),
    )


def _node(node_id, doc, page, node_class, text=""):
    return EvidenceNode(
        node_id=node_id, level=0, node_class=node_class,
        source_ref=SourceRef(doc_id=doc, doc_type="pdf", page_num=page),
        text=text, redundancy_cluster_id=f"{doc}:p{page}",
        token_equivalent_cost=10 if node_class == "doc_summary" else 100,
    )


def _group(node):
    return EvidenceGroup(
        group_id=f"grp_{node.node_id}", root_node_id=node.node_id,
        node_ids=[node.node_id], node_class=node.node_class, level=node.level,
        redundancy_cluster_id=node.redundancy_cluster_id,
        token_equivalent_cost=node.token_equivalent_cost,
        source_doc_id=node.source_ref.doc_id,
    )


def test_gold_page_requires_the_gold_document_and_minimal_is_sufficient():
    caption = _node("cap", "gold", 0, "doc_summary", "unrelated summary")
    fulltext = _node("txt", "gold", 0, "text_span", "the answer is 17")
    wrong_doc = _node("wrong", "other", 0, "text_span", "the answer is 17")
    groups = [_group(caption), _group(fulltext), _group(wrong_doc)]
    result = AutoLabeler(_cfg()).label(
        {"qa_id": "q", "doc_id": "gold", "answer": "17",
         "evidence": {"pages": [0], "modalities": ["text"]}},
        groups, NodeStore([caption, fulltext, wrong_doc]),
    )
    assert result["gold"] == ["grp_cap", "grp_txt"]
    assert result["minimal"] == ["grp_txt"]


def test_visual_gold_uses_screenshot_but_never_unverified_caption():
    caption = _node("cap", "gold", 1, "doc_summary", "generic")
    image = _node("img", "gold", 1, "pdf_page", "")
    result = AutoLabeler(_cfg()).label(
        {"qa_id": "q", "doc_id": "gold", "answer": "yes",
         "evidence": {"pages": [1], "modalities": ["image"]}},
        [_group(caption), _group(image)], NodeStore([caption, image]),
    )
    assert result["minimal"] == ["grp_img"]


def test_mixed_question_uses_each_evidence_pages_own_modality():
    text_caption = _node("text_cap", "gold", 0, "doc_summary", "generic")
    text_full = _node("text_txt", "gold", 0, "text_span", "supporting prose")
    image_caption = _node("image_cap", "gold", 1, "doc_summary", "generic")
    image = _node("image_img", "gold", 1, "pdf_page", "")
    nodes = [text_caption, text_full, image_caption, image]
    result = AutoLabeler(_cfg()).label(
        {"qa_id": "q", "doc_id": "gold", "answer": "not verbatim",
         "evidence": {"pages": [0, 1], "modalities": ["text", "table"],
                      "page_modalities": {"0": ["text"], "1": ["table"]}}},
        [_group(node) for node in nodes], NodeStore(nodes),
    )
    assert result["minimal"] == ["grp_text_txt", "grp_image_img"]
