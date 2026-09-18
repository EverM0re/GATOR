"""Fail-fast check for an OpenAI-compatible text + vision model endpoint.

Checks, in order:
  1. effective config (environment variables override YAML)
  2. GET /v1/models through the OpenAI client
  3. a tiny deterministic text completion
  4. a generated solid-red PNG sent as a real image_url payload

No API key is printed or written.  Exit code is zero only when both text and
vision calls succeed.

Usage:
    python -m scripts.check_vlm --config config/file_router.yaml
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _read_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _first(*values, default=""):
    for value in values:
        if value is not None and value != "":
            return value
    return default


def resolve_config(path: str) -> Dict[str, Any]:
    raw = _read_yaml(path)
    llm = raw.get("llm") or {}
    encoder_vlm = (raw.get("encoders") or {}).get("vlm") or {}

    backend = _first(
        os.environ.get("LLM_BACKEND"), llm.get("backend"),
        encoder_vlm.get("backend"), default="openai_compatible")
    base_url = _first(
        os.environ.get("LLM_BASE_URL"), os.environ.get("OPENAI_BASE_URL"),
        llm.get("base_url"), encoder_vlm.get("base_url"))
    # Config wins over the environment: each endpoint now carries its own
    # credential, and a stale exported LLM_API_KEY from an earlier shell would
    # otherwise override every one of them with a single wrong key.  The
    # environment remains a fallback for an endpoint that declares none.
    api_key = _first(
        llm.get("api_key"), encoder_vlm.get("api_key"),
        os.environ.get("LLM_API_KEY"), os.environ.get("OPENAI_API_KEY"),
        default="EMPTY")
    model = _first(
        os.environ.get("LLM_MODEL"), llm.get("model"), llm.get("openai_model"),
        encoder_vlm.get("model"), encoder_vlm.get("openai_model"))
    vision_raw = _first(
        os.environ.get("LLM_VISION"), llm.get("vision"),
        encoder_vlm.get("vision"), default=True)
    return {
        "backend": str(backend),
        "base_url": str(base_url).rstrip("/"),
        "api_key": str(api_key),
        "model": str(model),
        "vision": _as_bool(vision_raw),
        "config_path": str(Path(path).resolve()) if path else "",
    }


def _red_png_data_url() -> str:
    from PIL import Image

    image = Image.new("RGB", (128, 128), color=(235, 20, 20))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _response_text(response) -> str:
    return (response.choices[0].message.content or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/file_router.yaml")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", default="logs/vlm_preflight.json")
    args = parser.parse_args()

    cfg = resolve_config(args.config)
    safe_cfg = {key: value for key, value in cfg.items() if key != "api_key"}
    safe_cfg["api_key_present"] = bool(cfg["api_key"])
    result = {
        "ok": False,
        "config": safe_cfg,
        "checks": {},
    }

    print("[VLM CHECK] effective configuration")
    print(json.dumps(safe_cfg, ensure_ascii=False, indent=2))

    errors = []
    if cfg["backend"] not in {"openai", "openai_compatible"}:
        errors.append(f"backend must be openai/openai_compatible, got {cfg['backend']!r}")
    if not cfg["base_url"]:
        errors.append("LLM_BASE_URL/base_url is empty")
    if not cfg["model"]:
        errors.append("LLM_MODEL/model is empty")
    if not cfg["vision"]:
        errors.append("LLM_VISION/vision must be enabled for this multimodal experiment")
    if errors:
        result["error"] = "; ".join(errors)
        _write_result(args.output, result)
        print(f"[FAIL] {result['error']}")
        return 2

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=cfg["api_key"] or "EMPTY",
            base_url=cfg["base_url"],
            timeout=args.timeout,
            max_retries=0,
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"OpenAI client initialization failed: {exc!r}"
        _write_result(args.output, result)
        print(f"[FAIL] {result['error']}")
        return 3

    # 1) Models endpoint.
    started = time.perf_counter()
    try:
        models = client.models.list()
        model_ids = [item.id for item in models.data]
        result["checks"]["models"] = {
            "ok": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "available": model_ids[:20],
            "configured_model_listed": cfg["model"] in model_ids,
        }
        print(f"[PASS] /models reachable; models={model_ids[:5]}")
    except Exception as exc:  # noqa: BLE001
        result["checks"]["models"] = {
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": repr(exc)[:500],
        }
        result["error"] = "model endpoint is not reachable"
        _write_result(args.output, result)
        print(f"[FAIL] /models: {exc!r}")
        return 3

    # 2) Text completion.
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            temperature=0.0,
            max_tokens=32,
            messages=[{
                "role": "user",
                "content": "Reply with exactly API_OK and nothing else.",
            }],
        )
        text = _response_text(response)
        ok = "API_OK" in text.upper()
        result["checks"]["text"] = {
            "ok": ok,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "response": text[:300],
        }
        print(f"[{'PASS' if ok else 'FAIL'}] text response: {text!r}")
        if not ok:
            result["error"] = "text call returned an unexpected response"
            _write_result(args.output, result)
            return 4
    except Exception as exc:  # noqa: BLE001
        result["checks"]["text"] = {
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": repr(exc)[:500],
        }
        result["error"] = "text completion failed"
        _write_result(args.output, result)
        print(f"[FAIL] text completion: {exc!r}")
        return 4

    # 3) Real vision payload. The prompt does not reveal the expected color.
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            temperature=0.0,
            max_tokens=32,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "What is the dominant color of this image? "
                        "Reply with one lowercase English color word only.")},
                    {"type": "image_url", "image_url": {
                        "url": _red_png_data_url()}},
                ],
            }],
        )
        text = _response_text(response)
        normalized = text.lower().strip(" .!\n\t")
        ok = normalized == "red" or normalized.startswith("red ")
        result["checks"]["vision"] = {
            "ok": ok,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "response": text[:300],
            "expected": "red",
        }
        print(f"[{'PASS' if ok else 'FAIL'}] vision response: {text!r}")
        if not ok:
            result["error"] = "vision call did not identify the generated red image"
            _write_result(args.output, result)
            return 5
    except Exception as exc:  # noqa: BLE001
        result["checks"]["vision"] = {
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": repr(exc)[:500],
        }
        result["error"] = "vision completion failed"
        _write_result(args.output, result)
        print(f"[FAIL] vision completion: {exc!r}")
        return 5

    result["ok"] = True
    _write_result(args.output, result)
    print(f"[ALL PASS] text + vision endpoint is ready; report={args.output}")
    return 0


def _write_result(path: str, result: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
