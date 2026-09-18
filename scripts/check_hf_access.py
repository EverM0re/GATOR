"""Check that the dataset hosts are reachable before a repair tries to use them.

Every corpus here is fetched from HuggingFace, and `download_data` warns per
dataset but still exits 0, so an unreachable or gated hub looks exactly like a
successful download that produced nothing. Testing reachability first turns that
into a specific message.

    python3 -m scripts.check_hf_access
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The repos the download stage pulls from, as named in download_data.py.
REPOS = [
    ("Salesforce/UniDoc-Bench", "dataset"),
    ("MMDocIR/MMDocRAG", "dataset"),
    ("vidore/colpali_train_set", "dataset"),
]


def main() -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[hf] huggingface_hub is not installed: pip install huggingface_hub",
              file=sys.stderr)
        return 2

    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)
    api = HfApi()
    failures = []
    for repo, kind in REPOS:
        try:
            api.repo_info(repo, repo_type=kind, token=token, timeout=20)
            print(f"[hf] OK       {repo}")
        except Exception as exc:  # noqa: BLE001
            reason = repr(exc)
            failures.append((repo, reason))
            print(f"[hf] FAIL     {repo}: {reason[:160]}", file=sys.stderr)

    if not failures:
        print(f"[hf] hub reachable (token: {'yes' if token else 'no'})")
        return 0

    print("\n[hf] the dataset hub is not usable from this machine.",
          file=sys.stderr)
    joined = " ".join(r for r, _ in failures).lower()
    blob = " ".join(reason for _, reason in failures).lower()
    if "401" in blob or "403" in blob or "gated" in blob or "authenticate" in blob:
        print("     A repo is gated or the token is missing. Run "
              "'huggingface-cli login', or export HF_TOKEN.", file=sys.stderr)
    elif "proxy" in blob or "connection" in blob or "timed out" in blob \
            or "resolve" in blob or "network" in blob:
        print("     Looks like network egress is blocked. If this host uses a "
              "proxy, export HTTPS_PROXY before running.", file=sys.stderr)
        print("     An offline mirror also works: set HF_HOME / "
              "HF_HUB_OFFLINE=1 with a populated cache.", file=sys.stderr)
    else:
        print(f"     Unrecognised failure on: {joined}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
