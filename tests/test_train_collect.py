import json
from types import SimpleNamespace

from file_router.schemas import EvidenceGroup, ModalityBias
from scripts import train


def test_collect_exports_one_labeled_record_without_name_error(tmp_path, monkeypatch):
    splits = tmp_path / "splits"
    unified = tmp_path / "unified"
    dataset_dir = unified / "tiny"
    splits.mkdir()
    dataset_dir.mkdir(parents=True)
    (splits / "tiny.train.txt").write_text("q1\n", encoding="utf-8")
    (dataset_dir / "qa.jsonl").write_text(
        json.dumps({
            "qa_id": "q1", "question": "question", "answer": "answer",
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "router_training.jsonl"
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            splits_dir=str(splits), unified_dir=str(unified),
            training_data_path=str(output),
        ),
        datasets=SimpleNamespace(enabled=["tiny"]),
    )
    group = EvidenceGroup(
        group_id="g1", root_node_id="n1", node_ids=["n1"],
        node_class="doc_summary", level=1,
        redundancy_cluster_id="d:p0", base_score=1.0,
        token_equivalent_cost=10.0, source_doc_id="d",
    )

    class FakeRetriever:
        def recall(self, _question):
            return [group], ModalityBias()

    class FakeMemory:
        def __init__(self, _cfg, _dataset, _enc):
            self.nodes = {}

        def retriever(self):
            return FakeRetriever()

    class FakeLabeler:
        def __init__(self, _cfg):
            pass

        def label(self, *_args, **_kwargs):
            return {"gold": ["g1"], "minimal": ["g1"]}

    monkeypatch.setattr(train, "DatasetMemory", FakeMemory)
    monkeypatch.setattr(train, "AutoLabeler", FakeLabeler)
    monkeypatch.setattr(train, "ZeroShotRouter", lambda _cfg: object())

    assert train.collect(cfg, enc=object()) == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["query_id"] == "q1"
    assert row["gold_supporting_groups"] == ["g1"]
