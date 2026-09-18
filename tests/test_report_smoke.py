import tempfile

from scripts.evaluate_validation import (
    ROUTERS,
    pairwise_summary,
    summarize_routers,
    write_report,
)


def _record(router):
    return {
        "dataset": "demo", "qa_id": "q1", "router": router,
        "question": "How many?", "gold_answer": "17", "predicted_answer": "17",
        "llm_ok": True, "llm_stub": False, "em": 1.0, "exact_match": 1.0,
        "token_precision": 1.0, "token_recall": 1.0, "token_f1": 1.0,
        "bleu_1": 1.0, "bleu_2": 1.0, "rouge_l": 1.0,
        "char_f1": 1.0, "edit_similarity": 1.0, "numeric_exact": 1.0,
        "contains_answer": 1.0, "judge_valid": True, "judge_correct": 1.0,
        "judge_score": 1.0, "cost": 10.0, "cost_over_oracle": 1.0,
        "selected_count": 1, "selected": [{"group_id": f"{router}-g"}],
        "tiers": ["caption"], "selected_gold_doc_hit": True,
        "selected_gold_page_any_hit": True, "selected_gold_page_recall": 1.0,
        "doc_hit_at_5": True, "page_recall_all_candidates": 1.0,
        "page_recall_at_5": 1.0, "page_recall_at_10": 1.0,
        "images_selected": 0, "images_sent": 0,
        "answer_prompt_tokens": 10, "answer_completion_tokens": 1,
        "answer_total_tokens": 11, "answer_tokens_estimated": False,
        "judge_prompt_tokens": 10, "judge_completion_tokens": 5,
        "judge_total_tokens": 15, "judge_tokens_estimated": False,
        "retrieval_ms": 1.0, "routing_ms": 1.0, "materialization_ms": 1.0,
        "generation_ms": 1.0, "judge_ms": 1.0, "total_ms": 5.0,
        "gold_doc_rank": 1,
    }


def test_comprehensive_markdown_report_renders():
    records = [_record(router) for router in ROUTERS]
    overall, per_dataset = summarize_routers(records)
    summary = {
        "environment": {
            "generated_at": "2026-01-01T00:00:00", "platform": "test",
            "python": "3", "torch": {"version": "x", "cuda_available": False,
                                      "cuda_device": None},
            "text_encoder_requested": "stub", "text_encoder_actual": "stub",
            "visual_encoder_requested": "stub", "visual_encoder_actual": "stub",
            "llm_backend": "stub", "llm_model": "stub", "llm_vision": False,
            "learned_model_ready": True,
        },
        "data": {"demo": {"documents": 1, "train_qa": 1, "test_qa": 1}},
        "stores": {"demo": {"nodes": 1, "tiers": {"caption": 1}}},
        "training": {"records": 1, "avg_candidates": 1,
                     "minimal_tier_distribution": {"caption": 1}},
        "routers": {"overall": overall, "per_dataset": per_dataset},
        "pairwise": {
            "learned_vs_rule": pairwise_summary(records, "learned", "rule"),
            "learned_vs_zeroshot": pairwise_summary(records, "learned", "zeroshot"),
        },
        "retrieval_only": {},
        "token_and_time": {"ingestion_vlm": {}, "evaluation_vlm": {}},
    }
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".md") as handle:
        write_report(summary, records, handle.name)
        handle.seek(0)
        report = handle.read()
    assert "BLEU-1" in report
    assert "LM Judge" in report
    assert "Token usage and per-stage timing" in report
