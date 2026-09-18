"""Emit the configured answer-model backbones as pipe-separated rows.

Shell has no YAML parser, so the sweep scripts read their arms through this
instead of hardcoding endpoints. Emitting even unconfigured arms lets the
caller report "not configured" rather than silently running fewer arms than
the config appears to declare.

Format: name|model|base_url|api_key|vision|max_questions
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_router.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/file_router.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backbones = getattr(cfg, "backbones", None)
    if backbones is None:
        return 0

    llm = cfg.llm
    for name in vars(backbones):
        entry = getattr(backbones, name)
        model = str(getattr(entry, "model", "") or "")
        # Fall back to the main endpoint so an arm can name only a model when it
        # is served from the same place.
        base_url = str(getattr(entry, "base_url", "") or
                       getattr(llm, "base_url", "") or "")
        # Each backbone may live behind its own credential, so its own key wins.
        # The environment is only a fallback for an arm that declares none --
        # letting LLM_API_KEY take precedence would silently collapse several
        # distinct credentials into one.
        api_key = str(getattr(entry, "api_key", "")
                      or os.environ.get("LLM_API_KEY")
                      or os.environ.get("OPENAI_API_KEY")
                      or getattr(llm, "api_key", "")
                      or "")
        vision = "true" if bool(getattr(entry, "vision", True)) else "false"
        max_questions = getattr(entry, "max_questions", 0) or ""
        print(f"{name}|{model}|{base_url.rstrip('/')}|{api_key}|"
              f"{vision}|{max_questions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
