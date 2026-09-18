"""An answer truncated before it started must be retried, not discarded.

gpt-5.6-luna returns finish_reason="length" with empty content when the token
budget is spent on a preamble before the answer begins.  Returning that empty
string dropped 186 of 804 questions in the backbone ablation -- and not at
random: the loss concentrated on unidoc (48% empty) versus locomo (0.7%), i.e.
on the longest, most image-heavy prompts, which biases F1 upward by silently
excluding the hardest questions.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from file_router.encoders import vlm as vlm_mod  # noqa: E402


def _client(responses):
    """A VLM whose endpoint returns `responses` in order."""
    calls = []

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    content, finish = responses[len(calls) - 1]
                    return SimpleNamespace(
                        usage=SimpleNamespace(prompt_tokens=100,
                                              completion_tokens=10),
                        choices=[SimpleNamespace(
                            finish_reason=finish,
                            message=SimpleNamespace(content=content))])

    client = object.__new__(vlm_mod.VLM)
    client.cfg = SimpleNamespace(max_tokens=256, model="m",
                                 max_images_per_call=4, vision=True,
                                 image_max_side=768)
    client.backend = "openai"
    client._client = _Client()
    client._rate_limit_retries = 3
    client._max_answer_tokens = 2048
    client.last_call_ok = False
    client.last_call_was_stub = False
    client.last_error = ""
    client.failed_call_count = 0
    client.call_count = 0
    client._raw_usage = {}
    client._usage_by_purpose = {}
    client._last_sent_image_count = 0
    return client, calls


def test_length_truncated_empty_answer_is_retried_with_more_tokens():
    client, calls = _client([
        ("", "length"),                 # budget spent on a preamble
        ("$4.2 billion", "stop"),       # succeeds with more room
    ])
    answer = client._call("q", image_paths=None, max_tokens=256,
                          purpose="answer")

    assert answer == "$4.2 billion"
    assert client.last_call_ok is True
    assert client.last_call_was_stub is False
    assert len(calls) == 2, "must retry rather than return the empty string"
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"], (
        "the retry must raise the ceiling that caused the truncation")


def test_the_token_retry_does_not_consume_a_payload_downgrade(tmp_path):
    """Retrying for room must re-send the same payload, not a degraded one."""
    import base64
    pngs = []
    for name in ("a.png", "b.png"):
        f = tmp_path / name
        # 1x1 PNG: real bytes, so the encoder path behaves normally.
        f.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42m"
            "NkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="))
        pngs.append(str(f))
    client, calls = _client([
        ("", "length"),
        ("answer", "stop"),
    ])
    client._call("q", image_paths=pngs, max_tokens=256, purpose="answer")

    def n_images(call):
        content = call["messages"][0]["content"]
        if isinstance(content, str):
            return 0
        return len([c for c in content if c.get("type") == "image_url"])
    sent = [n_images(c) for c in calls]
    assert sent[0] == sent[1], (
        f"retry changed the payload ({sent}); it should only add token room")


def test_an_empty_answer_that_is_not_truncation_falls_through_to_downgrades(tmp_path):
    """finish=stop with no content is a different failure; try fewer images."""
    import base64
    pngs = []
    for name in ("a.png", "b.png"):
        f = tmp_path / name
        f.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42m"
            "NkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="))
        pngs.append(str(f))
    client, calls = _client([("", "stop")] * 6)
    client._call("q", image_paths=pngs, max_tokens=256, purpose="answer")

    assert len(calls) > 1, "must attempt a payload downgrade"
    assert client.last_call_was_stub is True
    assert all(c["max_tokens"] == calls[0]["max_tokens"] for c in calls), (
        "finish=stop is not a budget problem; the ceiling must not grow")
