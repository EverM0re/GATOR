from types import SimpleNamespace
from unittest.mock import patch
import tempfile

from PIL import Image

from file_router.encoders.vlm import VLM


def test_stub_usage_is_marked_estimated_and_grouped_by_purpose():
    cfg = SimpleNamespace(backend="stub", max_tokens=96, vision=False)
    # Production deliberately lets LLM_BACKEND override YAML. Isolate this unit
    # test so an exported real backend cannot turn a stub test into an API call.
    with patch.dict("os.environ", {"LLM_BACKEND": "stub"}):
        vlm = VLM(cfg)
        vlm.generate_answer("What?", "Evidence")
    assert vlm.last_usage["purpose"] == "answer"
    assert vlm.last_usage["estimated"] is True
    assert vlm.last_usage["total_tokens"] > 0
    summary = vlm.usage_summary()
    assert summary["answer"]["calls"] == 1
    assert summary["answer"]["estimated_calls"] == 1


def test_image_call_retries_at_lower_resolution():
    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient image failure")
            usage = SimpleNamespace(prompt_tokens=20, completion_tokens=3)
            message = SimpleNamespace(content="complete answer")
            return SimpleNamespace(usage=usage,
                                   choices=[SimpleNamespace(message=message)])

    cfg = SimpleNamespace(
        backend="stub", max_tokens=256, vision=True,
        max_images_per_call=4, max_image_side=1024, image_jpeg_quality=88,
        text_fallback_on_image_error=True,
    )
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        Image.new("RGB", (1800, 2400), "white").save(handle.name)
        with patch.dict("os.environ", {"LLM_BACKEND": "stub",
                                        "LLM_VISION": "true"}):
            vlm = VLM(cfg)
            fake = FakeCompletions()
            vlm.backend = "openai_compatible"
            vlm._client = SimpleNamespace(
                chat=SimpleNamespace(completions=fake))
            answer = vlm.generate_answer("What?", "caption", [handle.name])
    assert answer == "complete answer"
    assert fake.calls == 2
    assert vlm.last_call_ok is True
    assert vlm.last_usage["image_count"] == 1
