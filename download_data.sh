#!/usr/bin/env bash
# Fetch and prepare the two corpora, then build the three-tier cost ladder.
#
#   bash download_data.sh              # full corpora (slow: hours of ingestion)
#   SUBSET=300 DOCS=80 bash download_data.sh   # small subset, for a smoke run
#
# Both datasets are public and need no credentials. If your network blocks the
# HuggingFace hub, set HF_ENDPOINT to a mirror before running.
set -uo pipefail
cd "$(dirname "$0")"

DOCS="${DOCS:-1200}"
SUBSET="${SUBSET:-5000}"

echo "[data] checking hub reachability"
python3 -m scripts.check_hf_access || {
  echo "[data] hub unreachable. If it is blocked here, try:" >&2
  echo "       export HF_ENDPOINT=https://hf-mirror.com" >&2
  exit 1
}

echo "[data] downloading (docs<=$DOCS, qa<=$SUBSET)"
python3 -m scripts.download_data --subset "$SUBSET" --docs "$DOCS"

echo "[data] building unified format and document-grouped splits"
python3 -m scripts.build_unified --subset "$SUBSET" --docs "$DOCS"

echo "[data] verifying scale"
python3 -m scripts.check_data_scale --subset "$SUBSET" --docs "$DOCS" \
  --eval-per-dataset 50

echo "[data] ingesting: renders pages, runs OCR, builds the cost ladder"
echo "       (this is the slow step; no LLM calls are made here)"
python3 -m scripts.ingest --config config/file_router.yaml

echo "[data] done."
