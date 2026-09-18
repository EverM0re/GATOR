import json
import tempfile
from pathlib import Path

from scripts.summarize_campaign import summarize


def test_campaign_summary_collects_run_metadata():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "runs" / "validation_demo"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({
            "arguments": {"subset": 30},
            "encoders": {"visual": "clip"},
            "evaluation": {"split_seed": 42},
        }), encoding="utf-8")
        (run / "summary.json").write_text(json.dumps({
            "verdict": "promising",
            "routers": {"overall": {"learned": {
                "f1_valid": 0.7, "judge_accuracy": 0.8, "avg_cost": 123.0,
            }}},
            "pairwise": {"learned_vs_zeroshot": {"cost_saving_rate": 0.25}},
        }), encoding="utf-8")

        rows = summarize(root)
        report = (root / "CAMPAIGN_REPORT.md").read_text(encoding="utf-8")

    assert len(rows) == 1
    assert rows[0]["split_seed"] == 42
    assert rows[0]["visual_backend"] == "clip"
    assert "validation_demo" in report
    assert "0.25" in report


def _write_run(root: Path, name: str, seed: int, subset: int, f1: float,
               saving: float) -> None:
    run = root / "runs" / name
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "arguments": {"subset": subset},
        "encoders": {"visual": "clip"},
        "evaluation": {"split_seed": seed},
    }), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({
        "verdict": "promising",
        "routers": {"overall": {"learned": {
            "f1_valid": f1, "judge_accuracy": 0.25, "avg_cost": 1400.0,
            "gold_page_any_selection_rate": 0.5,
        }}},
        "pairwise": {"learned_vs_zeroshot": {
            "cost_saving_rate": saving, "f1_delta_valid": 0.1,
        }},
    }), encoding="utf-8")


def test_campaign_summary_aggregates_across_seeds(tmp_path):
    """A multi-seed large run is only interpretable as mean +/- std."""
    _write_run(tmp_path, "validation_a", 42, 1000, 0.10, 0.10)
    _write_run(tmp_path, "validation_b", 43, 1000, 0.20, 0.20)
    _write_run(tmp_path, "validation_c", 44, 1000, 0.30, 0.30)

    summarize(tmp_path)
    payload = json.loads(
        (tmp_path / "campaign_summary.json").read_text(encoding="utf-8"))

    assert len(payload["runs"]) == 3
    aggregates = payload["aggregates"]
    assert len(aggregates) == 1
    metrics = aggregates[0]["metrics"]
    assert aggregates[0]["runs"] == 3
    assert metrics["learned_f1"]["mean"] == 0.2
    assert metrics["learned_f1"]["min"] == 0.1
    assert metrics["learned_f1"]["max"] == 0.3
    assert metrics["learned_f1"]["std"] > 0
    report = (tmp_path / "CAMPAIGN_REPORT.md").read_text(encoding="utf-8")
    assert "Across-seed summary" in report


def test_campaign_summary_keeps_different_scales_apart(tmp_path):
    """A medium and a large run in one campaign must not be averaged together."""
    _write_run(tmp_path, "validation_medium", 42, 300, 0.10, 0.10)
    _write_run(tmp_path, "validation_large", 42, 1000, 0.30, 0.30)

    summarize(tmp_path)
    payload = json.loads(
        (tmp_path / "campaign_summary.json").read_text(encoding="utf-8"))

    subsets = sorted(entry["subset"] for entry in payload["aggregates"])
    assert subsets == [300, 1000]


def test_single_run_aggregate_reports_zero_spread(tmp_path):
    _write_run(tmp_path, "validation_only", 42, 300, 0.15, 0.13)
    summarize(tmp_path)
    payload = json.loads(
        (tmp_path / "campaign_summary.json").read_text(encoding="utf-8"))
    assert payload["aggregates"][0]["metrics"]["learned_f1"]["std"] == 0.0
