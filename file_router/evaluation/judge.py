"""Local LM-as-a-Judge prompt execution and robust JSON parsing."""

from __future__ import annotations

import json
import re


def _parse_judgment(raw: str) -> tuple[bool, bool, float, str]:
    text = (raw or "").strip()
    candidate = text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text,
                       flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    else:
        obj = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if obj:
            candidate = obj.group(0)
    try:
        parsed = json.loads(candidate)
        correct_value = parsed.get("correct")
        if isinstance(correct_value, str):
            correct_value = correct_value.strip().lower() in {"true", "yes", "1"}
        correct = bool(correct_value)
        score = max(0.0, min(1.0, float(parsed.get("score", float(correct)))))
        return True, correct, score, str(parsed.get("reason", ""))[:300]
    except Exception:  # noqa: BLE001
        match = re.search(r"(?:correct|judgment)\s*[:=]\s*(true|false|yes|no)",
                          text, flags=re.IGNORECASE)
        if match:
            correct = match.group(1).lower() in {"true", "yes"}
            return True, correct, float(correct), text[:300]
        return False, False, 0.0, "unparseable judge response"


def run_lm_judge(vlm, question: str, gold_answer: str,
                 predicted_answer: str, max_tokens: int = 128) -> dict:
    raw = vlm.judge_answer(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        max_tokens=max_tokens,
    )
    call_ok = bool(vlm.last_call_ok and not vlm.last_call_was_stub)
    parsed, correct, score, reason = _parse_judgment(raw) if call_ok else (
        False, False, 0.0, vlm.last_error or "judge call failed")
    usage = dict(vlm.last_usage)
    return {
        "judge_valid": bool(call_ok and parsed),
        "judge_correct": float(correct) if call_ok and parsed else None,
        "judge_score": score if call_ok and parsed else None,
        "judge_reason": reason,
        "judge_raw": raw[:500],
        "judge_prompt_tokens": usage.get("prompt_tokens", 0),
        "judge_completion_tokens": usage.get("completion_tokens", 0),
        "judge_total_tokens": usage.get("total_tokens", 0),
        "judge_tokens_estimated": usage.get("estimated", False),
        "judge_ms": vlm.last_latency_ms,
    }
