"""Unified LLM/VLM call interface.

Supports Anthropic, OpenAI, and an offline `stub` backend that produces
deterministic placeholder text (so the full pipeline runs with no API key).
"""

from __future__ import annotations

import base64
import copy
import io
import math
import os
import re
import time
from typing import List, Optional


def _is_rate_limit(exc) -> bool:
    """429s arrive as RateLimitError, or as a generic error carrying the code."""
    if type(exc).__name__ == "RateLimitError":
        return True
    return "429" in repr(exc) or "rate limit" in repr(exc).lower()


def _retry_after(exc, attempt: int) -> float:
    """Honour a server-stated wait, else back off exponentially.

    Endpoints that say "wait 120 seconds" mean it; retrying sooner extends the
    lockout instead of clearing it.
    """
    text = repr(exc)
    match = re.search(r"(\d+)\s*(?:seconds|秒)", text)
    if match:
        return min(float(match.group(1)) + 5.0, 300.0)
    return min(15.0 * (2 ** (attempt - 1)), 120.0)


class VLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = os.environ.get("LLM_BACKEND", cfg.backend)
        self._client = None
        self.call_count = 0
        self.failed_call_count = 0
        self.last_call_ok = False
        self.last_call_was_stub = self.backend == "stub"
        self.last_error = ""
        # Distinct from max_retries, which varies the image payload: this is how
        # many times a 429 is waited out and retried unchanged.
        self._rate_limit_retries = int(
            getattr(cfg, "rate_limit_retries", 3))
        # Ceiling for the empty-answer retry.  Some models spend the whole
        # budget on a preamble and return finish_reason="length" with no
        # content; raising the ceiling for that one call recovers the answer
        # without inflating the cost model, which prices the evidence sent in
        # rather than the tokens generated.
        self._max_answer_tokens = int(
            getattr(cfg, "max_answer_tokens",
                    int(getattr(cfg, "max_tokens", 256)) * 8))
        self.last_latency_ms = 0.0
        self.last_usage = self._empty_usage()
        self._raw_usage = None
        self._usage_by_purpose = {}
        self._last_sent_image_count = 0
        if self.backend == "anthropic":
            self._init_anthropic()
        elif self.backend in ("openai", "openai_compatible"):
            self._init_openai()

    def _init_anthropic(self):
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except Exception as e:  # noqa: BLE001
            print(f"[VLM] anthropic init failed: {e}. Falling back to stub.")
            self.backend = "stub"

    def _init_openai(self):
        """OpenAI / OpenAI-compatible (vLLM, DeepSeek, OpenRouter, ...).

        Reads optional cfg.base_url and cfg.api_key. If unset, falls back to
        the standard OPENAI_API_KEY / default endpoint.
        """
        try:
            from openai import OpenAI
            config_url = getattr(self.cfg, "base_url", "") or ""
            env_url = os.environ.get("LLM_BASE_URL", "")
            base_url = env_url or config_url or None
            # Config normally wins: endpoints carry their own credentials, so a
            # stale exported key must not silently replace all of them.  But
            # base_url is taken from the environment first, so when the
            # environment redirects to a DIFFERENT host, its key must come along
            # -- otherwise the config's key is sent to a service it does not
            # belong to.  That mismatch is an AuthenticationError on every call,
            # which the stub fallback then hides for the length of the run.
            env_key = os.environ.get("LLM_API_KEY", "")
            if env_key and env_url and env_url != config_url:
                api_key = env_key
            else:
                api_key = (getattr(self.cfg, "api_key", "") or
                           env_key or
                           os.environ.get("OPENAI_API_KEY") or "EMPTY")
            # The SDK defaults to a 600s timeout and 2 retries, so one
            # unresponsive remote endpoint can stall a run for 30 minutes per
            # call.  A local vLLM answers in under a second; a hosted API that
            # takes more than a minute is not going to finish 1600 calls.
            timeout = float(getattr(self.cfg, "request_timeout", 90.0))
            retries = int(getattr(self.cfg, "max_retries", 1))
            kwargs = {"api_key": api_key, "timeout": timeout,
                      "max_retries": retries}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
            print(f"[VLM] timeout={timeout}s retries={retries}")
            print(f"[VLM] openai backend ready (base_url={base_url or 'default'}, "
                  f"model={self._model_name()})")
        except Exception as e:  # noqa: BLE001
            print(f"[VLM] openai init failed: {e}. Falling back to stub.")
            self.backend = "stub"

    def _model_name(self) -> str:
        """Resolve model name across config variants (model | openai_model)."""
        return (os.environ.get("LLM_MODEL") or
                getattr(self.cfg, "model", "") or
                getattr(self.cfg, "openai_model", "") or "gpt-4o-mini")

    def _vision_enabled(self) -> bool:
        value = os.environ.get("LLM_VISION")
        if value is None:
            value = getattr(self.cfg, "vision", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    # ------------------------------------------------------------ public api
    def caption_image(self, image_path: str) -> str:
        prompt = (
            "Describe this image in 2-4 sentences. Include any visible text, "
            "objects, and notable structure."
        )
        return self._call(prompt, image_path=image_path, purpose="caption")

    def summarize_page(self, image_path: str, ocr_text: str = "") -> str:
        prompt = (
            "Produce a structured 3-6 sentence summary of this document page. "
            "Mention headings, any tables/figures, and the key facts."
        )
        if ocr_text:
            prompt += f"\n\nExtracted text (may be noisy):\n{ocr_text[:1500]}"
        return self._call(prompt, image_path=image_path, purpose="ingestion_caption")

    def summarize_text(self, texts: List[str], scope: str = "local cluster") -> str:
        joined = "\n\n---\n\n".join(texts)[:8000]
        prompt = (
            f"Write a concise summary (3-5 sentences) covering the following "
            f"{scope}:\n\n{joined}"
        )
        return self._call(prompt, purpose="summary")

    def generate_answer(self, question: str, context: str,
                        image_paths: Optional[List[str]] = None,
                        style: str = "direct") -> str:
        if style == "selfask":
            # Self-Ask: decompose into follow-up questions before answering,
            # over the same evidence. Tests whether structured decomposition
            # rather than granularity choice explains the gain.
            prompt = (
                "Answer the question using ONLY the provided context and "
                "attached images. First decide whether follow-up questions are "
                "needed. If so, write each as 'Follow up: ...' with its "
                "'Intermediate answer: ...'. Then write the final answer on a "
                "new line beginning with 'So the final answer is:'. The final "
                "answer must be concise but COMPLETE: include every requested "
                "list item, comparison point, date, number and unit. If the "
                "evidence is genuinely insufficient, answer UNKNOWN.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n"
            )
            raw = self._call(prompt, image_paths=image_paths,
                             max_tokens=int(self.cfg.max_tokens) * 3,
                             purpose="answer")
            # Score only the final answer, so the arm pays for decomposition in
            # cost but is graded on the same target as every other arm.
            for marker in ("So the final answer is:", "final answer is:",
                           "Answer:"):
                if marker in raw:
                    raw = raw.rsplit(marker, 1)[1]
                    break
            return raw.strip()
        if style == "cot":
            # Chain-of-thought baseline: same evidence, more reasoning tokens.
            # The answer marker keeps parsing identical to the direct arm.
            prompt = (
                "Answer the question using ONLY the provided context and "
                "attached images. Think step by step, then give the final "
                "answer on a new line beginning with 'Answer:'. The final "
                "answer must be concise but COMPLETE: include every requested "
                "list item, comparison point, date, number and unit. If the "
                "evidence is genuinely insufficient, answer UNKNOWN.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n\n"
                "Reasoning:"
            )
            # Reasoning is emitted before the answer, so the direct arm's token
            # ceiling would truncate the answer away.
            raw = self._call(prompt, image_paths=image_paths,
                             max_tokens=int(self.cfg.max_tokens) * 3,
                             purpose="answer")
            # Keep only the final answer, so CoT pays for reasoning tokens in
            # cost but is scored on the same target as every other arm.
            if "Answer:" in raw:
                raw = raw.rsplit("Answer:", 1)[1]
            return raw.strip()
        prompt = (
            "Answer the question using ONLY the provided context and attached images. "
            "Return a concise but COMPLETE final answer without a preamble. Include every "
            "requested list item, comparison point, date, number and unit. Perform any "
            "arithmetic explicitly requested by the question before giving the result. "
            "Do not shorten a multi-part answer to only one item. For yes/no questions, "
            "answer yes or no and add only the fact needed to disambiguate. If the evidence "
            "is genuinely insufficient, answer UNKNOWN.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        return self._call(prompt, image_paths=image_paths,
                          max_tokens=int(self.cfg.max_tokens),
                          purpose="answer")

    def judge_answer(self, question: str, gold_answer: str,
                     predicted_answer: str, max_tokens: int = 128) -> str:
        """Ask the configured local model for a compact correctness judgment."""
        prompt = (
            "You are an answer evaluator. Determine whether the candidate answer is "
            "substantively correct for the question given the reference answer. Accept "
            "equivalent wording, reordered lists, harmless extra units, and normalized "
            "date formats. Reject contradictions, wrong numbers, missing required list "
            "items, and unsupported UNKNOWN answers. Output exactly one JSON object and "
            "nothing else using this schema: "
            '{"correct": true, "score": 0.0, "reason": "brief reason"}. '
            "The score must be between 0 and 1.\n\n"
            f"Question: {question}\n"
            f"Reference answer: {gold_answer}\n"
            f"Candidate answer: {predicted_answer}\n"
        )
        return self._call(prompt, max_tokens=max_tokens, purpose="judge")

    # ------------------------------------------------------------ dispatch
    def _call(self, prompt: str, image_path: Optional[str] = None,
              image_paths: Optional[List[str]] = None,
              max_tokens: Optional[int] = None,
              purpose: str = "other") -> str:
        started = time.perf_counter()
        max_tokens = max_tokens or self.cfg.max_tokens
        paths = list(image_paths or [])
        if image_path:
            paths.insert(0, image_path)
        paths = list(dict.fromkeys(p for p in paths if p and os.path.exists(p)))
        max_images = max(0, int(getattr(self.cfg, "max_images_per_call", 4)))
        if max_images:
            paths = paths[:max_images]
        else:
            paths = []
        self.call_count += 1
        self.last_call_ok = False
        self.last_call_was_stub = False
        self.last_error = ""
        self._raw_usage = None
        self._last_sent_image_count = 0
        if self.backend == "anthropic":
            answer = self._call_anthropic(prompt, paths, max_tokens)
        elif self.backend in ("openai", "openai_compatible"):
            answer = self._call_openai(prompt, paths, max_tokens)
        else:
            self.last_call_was_stub = True
            self.last_error = "stub backend"
            answer = self._stub(prompt, paths[0] if paths else None)
        self._finish_usage(
            purpose=purpose, prompt=prompt, answer=answer, image_paths=paths,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        return answer

    def _call_anthropic(self, prompt: str, image_paths: List[str], max_tokens: int) -> str:
        content = []
        for image_path in image_paths if self._vision_enabled() else []:
            img_b64, mime = _encode_image_for_request(
                image_path,
                max_side=int(getattr(self.cfg, "max_image_side", 1024)),
                jpeg_quality=int(getattr(self.cfg, "image_jpeg_quality", 88)),
            )
            content.append({"type": "image", "source": {"type": "base64",
                                                        "media_type": mime, "data": img_b64}})
        content.append({"type": "text", "text": prompt})
        try:
            resp = self._client.messages.create(
                model=(os.environ.get("LLM_MODEL") or
                       getattr(self.cfg, "anthropic_model", "claude-3-5-sonnet-latest")),
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            usage = getattr(resp, "usage", None)
            self._raw_usage = {
                "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            }
            self._last_sent_image_count = len(image_paths)
            self.last_call_ok = True
            return resp.content[0].text.strip()
        except Exception as e:  # noqa: BLE001
            self.failed_call_count += 1
            self.last_call_was_stub = True
            self.last_error = repr(e)[:240]
            print(f"[VLM] anthropic call failed: {repr(e)[:120]}; returning stub.")
            return self._stub(prompt, image_paths[0] if image_paths else None)

    def _call_openai(self, prompt: str, image_paths: List[str], max_tokens: int) -> str:
        # Text-only path (most LLM-only vLLM servers don't accept images): send a
        # plain string. Only attach image when the configured model is multimodal.
        send_image = bool(image_paths) and self._vision_enabled()
        max_side = int(getattr(self.cfg, "max_image_side", 1024))
        quality = int(getattr(self.cfg, "image_jpeg_quality", 88))
        attempts = [(list(image_paths) if send_image else [], max_side)]
        if send_image and len(image_paths) > 1:
            attempts.append((image_paths[:max(1, len(image_paths) // 2)],
                             min(max_side, 896)))
        if send_image:
            single = (image_paths[:1], min(max_side, 768))
            if single not in attempts:
                attempts.append(single)
            if bool(getattr(self.cfg, "text_fallback_on_image_error", True)):
                attempts.append(([], max_side))

        encoded_cache = {}

        def messages_for(paths, side):
            if not paths:
                return [{"role": "user", "content": prompt}]
            content = [{"type": "text", "text": prompt}]
            for image_path in paths:
                key = (image_path, side, quality)
                if key not in encoded_cache:
                    encoded_cache[key] = _encode_image_for_request(
                        image_path, max_side=side, jpeg_quality=quality)
                img_b64, mime = encoded_cache[key]
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                })
            return [{"role": "user", "content": content}]

        errors = []
        retry_tokens = max_tokens
        # A token-budget retry must re-send the SAME payload with more room, so
        # it must not consume one of the payload-downgrade attempts.
        queue = list(enumerate(attempts, 1))
        while queue:
            attempt_index, (active_paths, side) = queue.pop(0)
            try:
                resp = self._client.chat.completions.create(
                    model=self._model_name(),
                    max_tokens=retry_tokens,
                    messages=messages_for(active_paths, side),
                    temperature=0.0,
                )
                usage = getattr(resp, "usage", None)
                self._raw_usage = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                }
                self._last_sent_image_count = len(active_paths)
                answer = (resp.choices[0].message.content or "").strip()
                finish = getattr(resp.choices[0], "finish_reason", "")
                if not answer:
                    # A model that spends the whole budget on a preamble before
                    # reaching the answer returns finish_reason="length" with
                    # empty content.  Retrying with more room is what fixes it;
                    # returning here instead discarded 186 of 804 questions,
                    # and not at random -- the loss concentrated on the longest,
                    # most image-heavy prompts, which are the hardest ones.
                    if (finish == "length"
                            and retry_tokens < self._max_answer_tokens):
                        errors.append(f"empty response (finish=length, "
                                      f"max_tokens={retry_tokens})")
                        retry_tokens = min(retry_tokens * 4,
                                           self._max_answer_tokens)
                        print(f"[VLM] empty answer truncated by max_tokens; "
                              f"retrying with {retry_tokens}")
                        queue.insert(0, (attempt_index,
                                         (active_paths, side)))
                        continue
                    self.last_error = f"empty response (finish={finish})"
                    self.last_call_ok = False
                    # Fall through to the next payload downgrade rather than
                    # returning: fewer or smaller images can also unstick it.
                    errors.append(self.last_error)
                    continue
                self.last_call_ok = True
                if errors:
                    self.last_error = (
                        f"recovered on attempt {attempt_index}; "
                        f"previous={errors[-1][:160]}")
                return answer
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
                # A rate limit is the one failure that retrying *identically*
                # can fix, but only after waiting.  The attempt list varies the
                # image payload, which does nothing for a 429, so without this
                # every remaining call in the run stubs out in a cascade --
                # backbone_strong lost 720 consecutive calls that way while the
                # credentials were valid the whole time.
                if _is_rate_limit(exc) and attempt_index <= self._rate_limit_retries:
                    delay = _retry_after(exc, attempt_index)
                    print(f"[VLM] rate limited; waiting {delay:.0f}s "
                          f"(attempt {attempt_index}/{self._rate_limit_retries})")
                    time.sleep(delay)
                    try:
                        resp = self._client.chat.completions.create(
                            model=self._model_name(),
                            max_tokens=max_tokens,
                            messages=messages_for(active_paths, side),
                            temperature=0.0,
                        )
                        usage = getattr(resp, "usage", None)
                        self._raw_usage = {
                            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                            "completion_tokens": int(
                                getattr(usage, "completion_tokens", 0) or 0),
                        }
                        self._last_sent_image_count = len(active_paths)
                        answer = (resp.choices[0].message.content or "").strip()
                        self.last_call_ok = bool(answer)
                        if not answer:
                            self.last_error = "empty response"
                        return answer
                    except Exception as retry_exc:  # noqa: BLE001
                        errors.append(repr(retry_exc))

        self.failed_call_count += 1
        self.last_call_was_stub = True
        self.last_error = (errors[-1] if errors else "OpenAI call failed")[:240]
        print(f"[VLM] openai call failed after {len(attempts)} attempts: "
              f"{self.last_error[:120]}; returning stub.")
        return self._stub(prompt, image_paths[0] if image_paths else None)

    def _stub(self, prompt: str, image_path: Optional[str]) -> str:
        """Deterministic offline placeholder."""
        if image_path:
            name = os.path.basename(image_path)
            return f"[stub caption] Visual content from {name}. Prompt context: {prompt[:80]}..."
        return f"[stub summary] {prompt[:180]}..."

    @staticmethod
    def _empty_usage() -> dict:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "image_count": 0,
            "estimated": False,
            "purpose": "",
        }

    def _finish_usage(self, purpose: str, prompt: str, answer: str,
                      image_paths: List[str], elapsed_ms: float) -> None:
        raw = self._raw_usage or {}
        prompt_tokens = int(raw.get("prompt_tokens", 0) or 0)
        completion_tokens = int(raw.get("completion_tokens", 0) or 0)
        estimated = not (prompt_tokens or completion_tokens)
        if estimated:
            # Used only when a compatible server omits ``usage``. The fixed
            # image estimate is deliberately marked as estimated in every row.
            prompt_tokens = (max(1, math.ceil(len(prompt) / 4)) +
                             256 * self._last_sent_image_count)
            completion_tokens = max(1, math.ceil(len(answer or "") / 4))
        self.last_latency_ms = float(elapsed_ms)
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "image_count": (self._last_sent_image_count
                            if self._vision_enabled() else 0),
            "estimated": estimated,
            "purpose": purpose,
        }
        aggregate = self._usage_by_purpose.setdefault(purpose, {
            "calls": 0, "successful_calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0, "image_count": 0,
            "latency_ms": 0.0, "estimated_calls": 0,
        })
        aggregate["calls"] += 1
        aggregate["successful_calls"] += int(self.last_call_ok and
                                              not self.last_call_was_stub)
        aggregate["prompt_tokens"] += prompt_tokens
        aggregate["completion_tokens"] += completion_tokens
        aggregate["total_tokens"] += prompt_tokens + completion_tokens
        aggregate["image_count"] += self.last_usage["image_count"]
        aggregate["latency_ms"] += float(elapsed_ms)
        aggregate["estimated_calls"] += int(estimated)

    def usage_summary(self) -> dict:
        """Return a secret-free snapshot grouped by call purpose."""
        return copy.deepcopy(self._usage_by_purpose)


def _read_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _encode_image_for_request(path: str, max_side: int,
                              jpeg_quality: int) -> tuple[str, str]:
    """Resize document images in memory to control Qwen visual token usage."""
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if max_side > 0 and max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=max(40, min(95, jpeg_quality)),
                       optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii"), "image/jpeg"
    except Exception:  # noqa: BLE001
        return _read_image_base64(path), _mime_for_path(path)


def _mime_for_path(path: str) -> str:
    p = path.lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith(".gif"):
        return "image/gif"
    if p.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"
