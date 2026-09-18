import json
import tempfile
from pathlib import Path

from scripts.check_experiment_gate import check_gate


def test_quality_gate_blocks_expensive_run_when_cost_regresses():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "runs" / "validation_demo"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({
            "status": "complete", "arguments": {"subset": 300},
        }), encoding="utf-8")
        (run / "summary.json").write_text(json.dumps({
            "routers": {"overall": {"oracle": {
                "judge_accuracy": 0.8, "llm_success_rate": 1.0,
                "candidate_gold_doc_recall_at_5": 0.9,
            }}},
            "pairwise": {"learned_vs_full": {"cost_saving_rate": -0.1}},
        }), encoding="utf-8")
        result = check_gate(root, subset=300)
    # Costing more than reading everything defeats the purpose of routing.
    assert result["passed"] is False
    assert result["checks"]["learned_cost_saving_vs_full"] is False


def test_quality_gate_requires_usable_learned_quality_and_selection(tmp_path):
    run = tmp_path / "runs" / "validation_demo"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "status": "complete", "arguments": {"subset": 300},
    }), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({
        "routers": {"overall": {
            "oracle": {
                "judge_accuracy": 0.6, "llm_success_rate": 1.0,
                "candidate_gold_doc_recall_at_5": 0.9,
                "candidate_gold_page_recall": 0.75,
                "visual_gold_n": 10,
                "visual_gold_image_send_rate": 1.0,
            },
            "learned": {
                "judge_accuracy": 0.2,
                "gold_page_any_selection_rate": 0.4,
                "f1_per_1k_tokens": 0.11,
            },
            "zeroshot": {"f1_per_1k_tokens": 0.05},
        }},
        "pairwise": {
            "learned_vs_zeroshot": {
                "cost_saving_rate": 0.02, "f1_delta_valid": 0.03,
            },
            "learned_vs_full": {
                "cost_saving_rate": 0.35, "f1_delta_valid": 0.01,
            },
        },
    }), encoding="utf-8")
    result = check_gate(tmp_path, subset=300)
    assert result["passed"] is True


def test_quality_gate_blames_retrieval_when_page_recall_is_the_ceiling(tmp_path):
    """Doc-level recall can look healthy while page-level recall is binding.

    campaign_20260823_010042 passed candidate_gold_doc_recall_at_5 at 0.90 and
    still could not select Gold pages, because page-level recall was 0.72 and
    page_recall@5 only 0.40.  The gate must name retrieval in that case instead
    of reporting the router as the sole failure.
    """
    run = tmp_path / "runs" / "validation_demo"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "status": "complete", "arguments": {"subset": 300},
    }), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({
        "routers": {"overall": {
            "oracle": {
                "judge_accuracy": 0.6, "llm_success_rate": 1.0,
                "candidate_gold_doc_recall_at_5": 0.9,
                "candidate_gold_page_recall": 0.42,
                "visual_gold_n": 10,
                "visual_gold_image_send_rate": 1.0,
            },
            "learned": {
                "judge_accuracy": 0.2,
                "gold_page_any_selection_rate": 0.4,
            },
        }},
        "pairwise": {"learned_vs_zeroshot": {
            "cost_saving_rate": 0.02, "f1_delta_valid": 0.03,
        }},
    }), encoding="utf-8")
    result = check_gate(tmp_path, subset=300)
    assert result["passed"] is False
    assert result["checks"]["candidate_gold_page_recall"] is False
    assert result["checks"]["candidate_gold_doc_recall_at_5"] is True


def _full_summary(cost_saving_vs_zeroshot: float, learned_eff: float,
                  zeroshot_eff: float, saving_vs_full: float,
                  f1_delta_vs_full: float) -> dict:
    return {
        "routers": {"overall": {
            "oracle": {
                "judge_accuracy": 0.6, "llm_success_rate": 1.0,
                "candidate_gold_doc_recall_at_5": 0.9,
                "candidate_gold_page_recall": 0.84,
                "visual_gold_n": 10, "visual_gold_image_send_rate": 0.95,
            },
            "learned": {
                "judge_accuracy": 0.27,
                "gold_page_any_selection_rate": 0.47,
                "f1_per_1k_tokens": learned_eff,
            },
            "zeroshot": {"f1_per_1k_tokens": zeroshot_eff},
        }},
        "pairwise": {
            "learned_vs_zeroshot": {
                "cost_saving_rate": cost_saving_vs_zeroshot,
                "f1_delta_valid": 0.14,
            },
            "learned_vs_full": {
                "cost_saving_rate": saving_vs_full,
                "f1_delta_valid": f1_delta_vs_full,
            },
        },
    }


def _write_summary(tmp_path, summary: dict) -> None:
    run = tmp_path / "runs" / "validation_demo"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "status": "complete", "arguments": {"subset": 300},
    }), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_gate_passes_a_router_that_answers_despite_a_cheap_unknown_baseline(tmp_path):
    """campaign_20260831_052759: zeroshot answered UNKNOWN 85% of the time.

    Being cheaper than a baseline that gives up is not evidence of anything, so
    cost_saving_vs_zeroshot must not block a router that is 4.5x more efficient
    and strictly better than reading everything.
    """
    _write_summary(tmp_path, _full_summary(
        cost_saving_vs_zeroshot=-0.4978, learned_eff=0.1094,
        zeroshot_eff=0.0244, saving_vs_full=0.3765, f1_delta_vs_full=0.0185))
    result = check_gate(tmp_path, subset=300)
    assert result["passed"] is True
    assert "learned_cost_saving_vs_zeroshot" not in result["checks"]
    assert result["values"]["learned_cost_saving_vs_zeroshot"] == -0.4978


def test_gate_blocks_a_router_less_efficient_than_zeroshot(tmp_path):
    _write_summary(tmp_path, _full_summary(
        cost_saving_vs_zeroshot=0.5, learned_eff=0.02,
        zeroshot_eff=0.0244, saving_vs_full=0.3, f1_delta_vs_full=0.01))
    result = check_gate(tmp_path, subset=300)
    assert result["passed"] is False
    assert result["checks"]["learned_efficiency_ratio_vs_zeroshot"] is False


def test_gate_blocks_a_router_that_loses_quality_against_full(tmp_path):
    """Cheap-but-worse than reading everything defeats the point of routing."""
    _write_summary(tmp_path, _full_summary(
        cost_saving_vs_zeroshot=0.5, learned_eff=0.11,
        zeroshot_eff=0.0244, saving_vs_full=0.6, f1_delta_vs_full=-0.20))
    result = check_gate(tmp_path, subset=300)
    assert result["passed"] is False
    assert result["checks"]["learned_f1_delta_vs_full"] is False
