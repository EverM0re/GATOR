"""A 429 must be waited out, not turned into a stub.

backbone_strong and backbone_mid both ran to completion against valid
credentials and produced F1 0.0000 across every arm, because the first rate
limit stubbed and each subsequent call did the same: 720 consecutive failures.
The retry list varies the image payload, which cannot help a 429.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from file_router.encoders import vlm as vlm_mod  # noqa: E402


class _RateLimitError(Exception):
    """Shaped like the SDK's error, which is matched by class name."""


_RateLimitError.__name__ = "RateLimitError"


def test_rate_limit_is_detected_by_class_name_and_by_code():
    assert vlm_mod._is_rate_limit(_RateLimitError("slow down"))
    assert vlm_mod._is_rate_limit(Exception("Error code: 429 - too many"))
    assert not vlm_mod._is_rate_limit(Exception("Error code: 401 - bad key"))


def test_server_stated_wait_is_honoured():
    """The endpoint asked for 120s; retrying sooner extends the lockout."""
    exc = Exception("Error code: 429 - {'message': "
                    "'您多次使用无效令牌"
                    "，请等待 120 秒后再试'}")
    assert vlm_mod._retry_after(exc, 1) == 125.0

    english = Exception("429 rate limit, retry after 30 seconds")
    assert vlm_mod._retry_after(english, 1) == 35.0


def test_backoff_grows_and_is_capped_without_a_stated_wait():
    exc = Exception("Error code: 429")
    first = vlm_mod._retry_after(exc, 1)
    second = vlm_mod._retry_after(exc, 2)
    assert second > first
    assert vlm_mod._retry_after(exc, 99) <= 120.0


def test_a_recovered_rate_limit_returns_the_answer_not_a_stub(monkeypatch):
    """The whole point: one 429 must not void the rest of the run."""
    calls = {"n": 0}

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        raise _RateLimitError("Error code: 429 - wait 1 seconds")
                    return SimpleNamespace(
                        usage=SimpleNamespace(prompt_tokens=5,
                                              completion_tokens=2),
                        choices=[SimpleNamespace(
                            message=SimpleNamespace(content="real answer"))])

    client = object.__new__(vlm_mod.VLM)
    client.cfg = SimpleNamespace(max_tokens=32, model="m")
    client.backend = "openai"
    client._client = _Client()
    client._rate_limit_retries = 3
    client.last_call_ok = False
    client.last_call_was_stub = False
    client.last_error = ""
    client.failed_call_count = 0
    client.call_count = 0
    client._usage_by_purpose = {}
    client._raw_usage = {}
    client._last_sent_image_count = 0
    monkeypatch.setattr(vlm_mod.time, "sleep", lambda s: None)

    answer = client._call("q", image_paths=None, max_tokens=32, purpose="answer")

    assert answer == "real answer"
    assert client.last_call_ok is True
    assert client.last_call_was_stub is False
    assert calls["n"] == 2, "should retry once after waiting"
