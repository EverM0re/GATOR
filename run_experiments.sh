#!/usr/bin/env bash
# Train the router and evaluate every arm reported in the paper.
#
#   bash run_experiments.sh                  # one seed
#   SEEDS="42 43 44" bash run_experiments.sh # three seeds, as in the paper
#
# Requires an OpenAI-compatible endpoint for answer generation and judging.
# Set it first:
#   export LLM_BASE_URL=http://localhost:8000/v1
#   export LLM_MODEL=Qwen3-VL-8B-Instruct
#   export LLM_API_KEY=...        # omit if your server needs no key
set -uo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-42}"
OUT="${OUT:-runs}"
mkdir -p "$OUT"

if [[ -z "${LLM_BASE_URL:-}" ]]; then
  echo "[run] LLM_BASE_URL is not set; see the header of this script." >&2
  exit 1
fi

for SEED in $SEEDS; do
  echo "[run] seed $SEED"
  env RUN_ROOT="$OUT/seed${SEED}" \
      SKIP_DOWNLOAD=1 SKIP_INGEST=1 SKIP_BUILD=0 \
      SPLIT_SEED="$SEED" TRAIN_SEED="$SEED" \
      EVAL_PER_DATASET="${EVAL_PER_DATASET:-0}" \
      EXTRA_ROUTERS="bm25 dense cot selfask" \
    bash ./run_validation.sh
done

echo "[run] reports written under $OUT/"
