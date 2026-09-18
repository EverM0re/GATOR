import json
import os
import tempfile

from file_router.schemas import EvidenceGroup, ModalityBias
from file_router.training.data_exporter import DataExporter


def test_compact_export_writes_labels_once_without_coverage_vector():
    group = EvidenceGroup(
        group_id="g", root_node_id="n", node_ids=["n"],
        node_class="doc_summary", level=1,
        redundancy_cluster_id="d:p0", base_score=0.8,
        token_equivalent_cost=10.0, coverage_signature=[0.1] * 64,
        source_doc_id="d",
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "records.jsonl")
        exporter = DataExporter(path, compact=True)
        exporter.export(
            "question", "answer", [group], [], ModalityBias(), query_id="q",
            gold_groups=["g"], minimal_groups=["g"], correct=True,
        )
        with open(path, encoding="utf-8") as handle:
            row = json.loads(handle.read())
    assert row["gold_supporting_groups"] == ["g"]
    assert row["minimal_sufficient_groups"] == ["g"]
    assert "coverage_signature" not in row["candidate_groups"][0]
