"""Verify a *stage subprocess* can authenticate, not just the parent process.

This is the failure that cost the overnight_v2 campaign five phases: the
preflight read the source config and passed, while the stages ran against a
copy with the key stripped and 401'd on every call. Checking the parent's
credentials therefore proves nothing about the run.

So this spawns a subprocess through the same environment the stage runner
builds, and has it make one real answered call.

    python3 -m scripts.check_stage_auth
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_validation import resolved_api_key  # noqa: E402

PROBE = """
import json, sys
sys.path.insert(0, %r)
from file_router.config import load_config
from file_router.encoders.vlm import VLM
cfg = load_config(%r)
client = VLM(cfg.llm)
answer = client.generate_answer("Reply with the word OK.", context="")
print(json.dumps({
    "ok": bool(client.last_call_ok),
    "stub": bool(client.last_call_was_stub),
    "error": str(client.last_error or ""),
    "answer": (answer or "")[:80],
}))
"""


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = os.environ.get("FR_CONFIG", "config/file_router.yaml")

    api_key = resolved_api_key(config)
    env = dict(os.environ)
    # Exactly what StageRunner injects; if this is empty the stages get nothing.
    if api_key:
        env["LLM_API_KEY"] = api_key
    else:
        print("[stage-auth] no API key resolved from config or environment.\n"
              "  Set llm.api_key in the config, or export LLM_API_KEY.",
              file=sys.stderr)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE % (str(root), config)],
            env=env, cwd=str(root), capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("[stage-auth] FAIL: the endpoint did not answer within 120s.",
              file=sys.stderr)
        return 2

    line = next((l for l in reversed(proc.stdout.splitlines())
                 if l.strip().startswith("{")), "")
    if not line:
        print("[stage-auth] FAIL: probe did not run. This is an environment "
              "problem, not an authentication one:", file=sys.stderr)
        for stream in (proc.stdout, proc.stderr):
            for text in stream.strip().splitlines()[-6:]:
                print("    " + text, file=sys.stderr)
        return 3

    result = json.loads(line)
    if result["stub"]:
        print("[stage-auth] FAIL: the client fell back to its stub, so answers "
              "would be fabricated rather than generated.", file=sys.stderr)
        print(f"    {result['error']}", file=sys.stderr)
        return 4
    if not result["ok"]:
        print("[stage-auth] FAIL: the call was rejected.", file=sys.stderr)
        print(f"    {result['error']}", file=sys.stderr)
        return 5

    print(f"[stage-auth] OK -- stage subprocess authenticated "
          f"(answer: {result['answer']!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
