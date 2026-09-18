from file_router.evaluation.judge import run_lm_judge


class FakeVLM:
    last_call_ok = True
    last_call_was_stub = False
    last_error = ""
    last_latency_ms = 12.5
    last_usage = {
        "prompt_tokens": 31,
        "completion_tokens": 12,
        "total_tokens": 43,
        "estimated": False,
    }

    def judge_answer(self, **kwargs):
        return '{"correct": true, "score": 0.95, "reason": "equivalent"}'


def test_lm_judge_parses_json_and_usage():
    result = run_lm_judge(FakeVLM(), "q", "gold", "candidate")
    assert result["judge_valid"] is True
    assert result["judge_correct"] == 1.0
    assert result["judge_score"] == 0.95
    assert result["judge_total_tokens"] == 43
    assert result["judge_ms"] == 12.5
