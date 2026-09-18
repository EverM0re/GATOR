#!/usr/bin/env bash
# Install the optional mount hosts and datasets wave 4 can use.
#
# Everything here is optional: a phase whose dependency is missing skips rather
# than failing the run, so this only widens what wave 4 can cover.
#
#   bash scripts/setup_hosts.sh            # everything
#   bash scripts/setup_hosts.sh memos      # just one
#
# Safe to re-run; anything already present is left alone.
set -uo pipefail
cd "$(dirname "$0")/.."

WANT="${*:-memos memgallery}"
ok()   { echo "[setup] OK    $*"; }
skip() { echo "[setup] SKIP  $*"; }
fail() { echo "[setup] FAIL  $*" >&2; }

# ------------------------------------------------------------------- memos
# MemTensor/MemOS publishes to PyPI as `MemoryOS`, which is also the name of a
# different project (BAI-LAB/MemoryOS). Importing `memos` is what tells the two
# apart, so verify the import rather than trusting that pip succeeded.
if [[ "$WANT" == *memos* ]]; then
  if python3 -c "import memos, chonkie" 2>/dev/null; then
    ok "memos already importable (with chonkie)"
  else
    echo "[setup] installing MemoryOS (MemTensor/MemOS) ..."
    if pip install -q "MemoryOS>=2.0" 2>&1 | tail -3; then :; fi
    # MemOS declares its chunker dependency lazily: `import memos` succeeds
    # without chonkie, and the failure only surfaces when the reader builds a
    # chunker -- i.e. after each mount run has already started.
    pip install -q chonkie 2>&1 | tail -2
    if python3 -c "import memos, chonkie" 2>/dev/null; then
      ok "memos installed (with chonkie)"
    else
      fail "MemoryOS installed but 'import memos' still fails."
      echo "        The PyPI name is shared with a different project; if pip" >&2
      echo "        resolved to BAI-LAB/MemoryOS, install from source instead:" >&2
      echo "          pip install git+https://github.com/MemTensor/MemOS.git" >&2
    fi
  fi
fi

# -------------------------------------------------------------- memgallery
# Mem-Gallery ships local page images rather than hot-linked URLs, which is the
# reason to use it: it exercises the screenshot rung on data we did not build.
if [[ "$WANT" == *memgallery* ]]; then
  # Raw datasets live OUTSIDE File_Router: the project folder is re-uploaded
  # wholesale between runs, so anything inside it is destroyed on every update.
  # FR_DATA_ROOT overrides; the default is a sibling of File_Router, matching
  # where memory_hosts/ already lives.
  DATA_ROOT="${FR_DATA_ROOT:-$(cd .. && pwd)/data}"
  TARGET="$DATA_ROOT/raw/memgallery"
  if [[ -n "$(ls -A "$TARGET" 2>/dev/null)" ]]; then
    ok "memgallery already downloaded ($TARGET)"
  else
    if ! command -v huggingface-cli >/dev/null 2>&1; then
      echo "[setup] installing huggingface_hub ..."
      pip install -q "huggingface_hub[cli]" 2>&1 | tail -2
    fi
    if command -v huggingface-cli >/dev/null 2>&1; then
      mkdir -p "$TARGET"
      # `huggingface-cli download` prints the path and exits 0 even when it
      # fetched nothing, so check that files actually landed.
      huggingface-cli download YuanchenBei/Mem-Gallery \
        --repo-type dataset --local-dir "$TARGET" 2>&1 | tail -4
      if [[ -n "$(find "$TARGET" -name '*.json' -print -quit 2>/dev/null)" ]]; then
        ok "memgallery downloaded"
      else
        fail "no .json landed in $TARGET."
        echo "        If the repo is gated:  huggingface-cli login" >&2
        echo "        If the hub is blocked: export HF_ENDPOINT=https://hf-mirror.com" >&2
        echo "        Then re-run: bash scripts/setup_hosts.sh memgallery" >&2
      fi
    else
      fail "huggingface-cli unavailable; cannot fetch Mem-Gallery"
    fi
  fi

  # Converting here rather than at run time surfaces a schema mismatch now,
  # while there is someone watching, instead of mid-campaign.
  if [[ -n "$(ls -A "$TARGET" 2>/dev/null)" ]]; then
    if python3 -m scripts.download_memgallery --src "$TARGET" \
         --out "$DATA_ROOT/unified/memgallery" 2>&1 | tail -3; then
      ok "memgallery converted to unified format"
    else
      fail "conversion failed -- the released layout differs from what"
      echo "        scripts/download_memgallery.py expects; inspect the JSON" >&2
      echo "        under $TARGET and adjust convert()." >&2
    fi
  fi
fi

echo
echo "[setup] status:"
python3 -c "import memos, chonkie" 2>/dev/null \
  && echo "  memos       ready" || echo "  memos       missing (phase will skip)"
[[ -s "${FR_DATA_ROOT:-$(cd .. && pwd)/data}/unified/memgallery/qa.jsonl" ]] \
  && echo "  memgallery  ready" || echo "  memgallery  missing (phase will skip)"
