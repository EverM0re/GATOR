import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from file_router.data.loaders import load_mmdocrag


def test_duplicate_document_quotes_are_merged():
    rows = [
        {"q_id": 1, "doc_name": "same", "question": "q1", "answer_short": "a1",
         "text_quotes": [{"quote_id": "t1", "text": "one"}],
         "img_quotes": [], "gold_quotes": ["t1"]},
        {"q_id": 2, "doc_name": "same", "question": "q2", "answer_short": "a2",
         "text_quotes": [{"quote_id": "t2", "text": "two"}],
         "img_quotes": [], "gold_quotes": ["t2"]},
    ]
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        raw_dir = tmp_path / "raw" / "mmdocrag"
        resource_dir = tmp_path / "resources" / "mmdocrag"
        unified_dir = tmp_path / "unified"
        raw_dir.mkdir(parents=True)
        resource_dir.mkdir(parents=True)
        (raw_dir / "qa.json").write_text(json.dumps(rows), encoding="utf-8")
        cfg = SimpleNamespace(
            paths=SimpleNamespace(raw_dir=str(tmp_path / "raw"),
                                  resources_dir=str(tmp_path / "resources"),
                                  unified_dir=str(unified_dir)),
            datasets=SimpleNamespace(subset_docs=10),
        )
        load_mmdocrag(cfg)
        corpus = [json.loads(line) for line in
                  (unified_dir / "mmdocrag" / "corpus.jsonl").read_text().splitlines()]
        qa = [json.loads(line) for line in
              (unified_dir / "mmdocrag" / "qa.jsonl").read_text().splitlines()]
    assert len(corpus) == 1
    assert [page["quote_id"] for page in corpus[0]["pages"]] == ["t1", "t2"]
    assert qa[0]["evidence"]["pages"] == [0]
    assert qa[1]["evidence"]["pages"] == [1]


def test_qa_local_quote_ids_keep_their_own_content_and_modality():
    rows = [
        {"q_id": 1, "doc_name": "same", "question": "q1", "answer_short": "a1",
         "text_quotes": [{"quote_id": "text1", "text": "first"}],
         "img_quotes": [], "gold_quotes": ["text1"],
         "evidence_modality_type": ["text"]},
        {"q_id": 2, "doc_name": "same", "question": "q2", "answer_short": "a2",
         "text_quotes": [],
         "img_quotes": [{"quote_id": "text1", "type": "table",
                         "img_description": "second", "img_path": "second.png"}],
         "gold_quotes": ["text1"], "evidence_modality_type": ["table"]},
    ]
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        raw_dir = tmp_path / "raw" / "mmdocrag"
        resource_dir = tmp_path / "resources" / "mmdocrag"
        unified_dir = tmp_path / "unified"
        raw_dir.mkdir(parents=True)
        resource_dir.mkdir(parents=True)
        (raw_dir / "qa.json").write_text(json.dumps(rows), encoding="utf-8")
        cfg = SimpleNamespace(
            paths=SimpleNamespace(raw_dir=str(tmp_path / "raw"),
                                  resources_dir=str(tmp_path / "resources"),
                                  unified_dir=str(unified_dir)),
            datasets=SimpleNamespace(subset_docs=10),
        )
        load_mmdocrag(cfg)
        corpus = [json.loads(line) for line in
                  (unified_dir / "mmdocrag" / "corpus.jsonl").read_text().splitlines()]
        qa = [json.loads(line) for line in
              (unified_dir / "mmdocrag" / "qa.jsonl").read_text().splitlines()]
    assert [page["text"] for page in corpus[0]["pages"]] == ["first", "second"]
    assert qa[0]["evidence"]["pages"] == [0]
    assert qa[0]["evidence"]["page_modalities"] == {"0": ["text"]}
    assert qa[1]["evidence"]["pages"] == [1]
    assert qa[1]["evidence"]["page_modalities"] == {"1": ["table"]}
