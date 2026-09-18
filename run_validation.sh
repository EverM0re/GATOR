#!/usr/bin/env bash
# Small, reproducible validation run for File_Router.
#
# Recommended server usage (multimodal OpenAI-compatible endpoint):
#   export LLM_BACKEND=openai_compatible
#   export LLM_BASE_URL=http://127.0.0.1:8000/v1
#   export LLM_API_KEY=EMPTY
#   export LLM_MODEL=Qwen3-VL-8B-Instruct
#   export LLM_VISION=1
#   bash ./run_validation.sh
#
# The script never writes LLM_API_KEY to disk. Results are placed under
# validation_runs/validation_<timestamp>/, with REPORT.md as the main handoff.
# For smoke/medium/large presets and multi-seed runs, use run_experiment.sh.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
SUBSET="${SUBSET:-30}"
DOCS="${DOCS:-8}"
EVAL_PER_DATASET="${EVAL_PER_DATASET:-8}"
EPOCHS="${EPOCHS:-20}"
DEVICE="${DEVICE:-auto}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_INGEST="${SKIP_INGEST:-0}"
REUSE_MODEL="${REUSE_MODEL:-}"
SKIP_LLM_CHECK="${SKIP_LLM_CHECK:-0}"
RUN_ROOT="${RUN_ROOT:-validation_runs}"

export TOKENIZERS_PARALLELISM="false"
export PYTHONWARNINGS="ignore"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export PYTHONUNBUFFERED="1"

# Fail before downloading/training if a configured real VLM endpoint is not
# reachable or cannot consume images. Set SKIP_LLM_CHECK=1 only for debugging.
if [[ "${LLM_BACKEND:-openai_compatible}" != "stub" && "$SKIP_LLM_CHECK" != "1" ]]; then
  "$PYTHON" -m scripts.check_vlm --config config/file_router.yaml
fi

args=(
  -m scripts.run_validation
  --config config/file_router.yaml
  --subset "$SUBSET"
  --docs "$DOCS"
  --eval-per-dataset "$EVAL_PER_DATASET"
  --epochs "$EPOCHS"
  --device "$DEVICE"
  --run-root "$RUN_ROOT"
)

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  args+=(--skip-download)
fi
if [[ "$SKIP_BUILD" == "1" ]]; then
  args+=(--skip-build)
fi
if [[ "$SKIP_INGEST" == "1" ]]; then
  args+=(--skip-ingest)
fi
if [[ -n "$REUSE_MODEL" ]]; then
  args+=(--reuse-model "$REUSE_MODEL")
fi

"$PYTHON" "${args[@]}"
