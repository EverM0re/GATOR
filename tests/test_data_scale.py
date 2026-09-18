import json
from types import SimpleNamespace

from scripts.check_data_scale import minimum_qa, validate_scale


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_medium_scale_rejects_smoke_cache(tmp_path):
    for dataset in ("unidoc", "mmdocrag", "vidore", "real_mm_rag"):
        root = tmp_path / dataset
        count = 8 if dataset in {"unidoc", "mmdocrag"} else 30
        _write_jsonl(root / "qa.jsonl", [
            {"qa_id": f"q{i}", "doc_id": f"d{i}"} for i in range(count)
        ])
        _write_jsonl(root / "corpus.jsonl", [
            {"doc_id": f"d{i}", "pages": [{"image_path": "image.png"}]}
            for i in range(count)
        ])
    cfg = SimpleNamespace(
        datasets=SimpleNamespace(
            enabled=["unidoc", "mmdocrag"],
            retrieval_only=["vidore", "real_mm_rag"],
        ),
        paths=SimpleNamespace(unified_dir=str(tmp_path)),
    )
    _rows, problems = validate_scale(cfg, 300, 80, 50)
    assert len(problems) == 4
    assert any("unidoc: only 8 QA" in problem for problem in problems)


def test_known_public_caps_allow_valid_large_counts():
    # UniDoc's finance split yields 157 QA, so a large run must not be asked for
    # more than that -- overnight_20260901_113518 failed its large phase on a
    # requirement the source could never satisfy.
    assert minimum_qa("unidoc", 1000, 200, 100) == 157
    assert minimum_qa("vidore", 1000, 200, 100) == 200
    assert minimum_qa("mmdocrag", 1000, 200, 100) == 200
    assert minimum_qa("unidoc", 30, 8, 8) == 8


def test_requirement_never_exceeds_the_source_cap():
    """The invariant behind the fix, checked directly rather than by example."""
    from scripts.check_data_scale import SOURCE_QA_CAPS

    for dataset, cap in SOURCE_QA_CAPS.items():
        for subset in (300, 1000, 5000):
            assert minimum_qa(dataset, subset, 200, 100) <= cap, (
                f"{dataset} would require more QA than the source provides")


def test_build_unified_accepts_the_same_caps_as_the_download_stage():
    """`--docs 80` must mean 80 documents in both stages.

    The download stage takes --docs, but the unified build originally read the
    cap only from the config, so fetching 80 documents and then building left
    only `datasets.subset_docs` (12) of them -- the corpus silently shrank
    between two consecutive commands.
    """
    import argparse
    import inspect
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import build_unified, download_data

    def flags(module):
        parser_args = set()
        source = inspect.getsource(module.main)
        for name in ("--docs", "--subset"):
            if f'"{name}"' in source:
                parser_args.add(name)
        return parser_args

    assert flags(build_unified) == flags(download_data) == {"--docs", "--subset"}


def test_validation_pipeline_forwards_the_caps_to_the_unified_build():
    """A cap honoured by one stage and ignored by the next is a silent shrink."""
    import inspect
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import run_validation

    source = inspect.getsource(run_validation)
    build_call = source[source.index("scripts.build_unified"):]
    build_call = build_call[:build_call.index("])")]
    assert "--docs" in build_call and "--subset" in build_call
