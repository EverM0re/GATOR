#!/usr/bin/env bash
# Verify the uploaded tree actually contains the changes a run depends on.
#
# A manual upload can silently deliver an older tree, and the symptom is a run
# that completes normally while producing the previous result -- baselines has
# now cost 28 minutes that way.  Check before spending the time, not after.
#
#   bash scripts/check_upload.sh
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
ok()   { echo "  [OK]   $*"; }
bad()  { echo "  [MISS] $*"; fail=1; }

echo "== code =="
grep -q 'RETRIEVAL_ROUTERS = ("bm25", "dense")' scripts/evaluate_validation.py \
  && ok "bm25 / dense arms"            || bad "bm25 / dense arms"
grep -q '"selfask": "selfask"' scripts/evaluate_validation.py \
  && ok "cot / selfask arms"           || bad "cot / selfask arms"
grep -q 'sentence_transformer' integrations/adapters.py \
  && ok "MemOS local embedder"         || bad "MemOS local embedder"
grep -q '"chunker"' integrations/adapters.py \
  && ok "MemOS chunker config"         || bad "MemOS chunker config"
grep -q 'chunk_token_counter' integrations/adapters.py \
  && ok "MemOS offline tokenizer"      || bad "MemOS offline tokenizer"
grep -q 'HF_HUB_DISABLE_IMPLICIT_TOKEN' run_wave4.sh \
  && ok "no interactive HF prompts"    || bad "no interactive HF prompts"
grep -q 'OPENAI_API_KEY' integrations/adapters.py \
  && ok "A-Mem key export"             || bad "A-Mem key export"
grep -q '_record_hardware' scripts/run_validation.py \
  && ok "hardware recording"           || bad "hardware recording"
grep -q 'TRAIN_SEED' scripts/run_validation.py \
  && ok "independent train seeds"      || bad "independent train seeds"
grep -q 'finish == "length"' file_router/encoders/vlm.py \
  && ok "empty-answer retry"           || bad "empty-answer retry"
grep -q '_is_rate_limit' file_router/encoders/vlm.py \
  && ok "429 backoff"                  || bad "429 backoff"
grep -q -- '--docs' scripts/build_unified.py \
  && ok "build_unified --docs"         || bad "build_unified --docs"
grep -q 'CONVERGE_EPOCHS' run_wave4.sh \
  && ok "converge epoch ceiling"       || bad "converge epoch ceiling"
grep -q 'answers_valid' run_wave4.sh \
  && ok "stub-run guard"               || bad "stub-run guard"

echo "== data =="
[[ -L data && -e data/unified ]] \
  && ok "data/ linked and populated"   || bad "data/ (run: bash scripts/link_external_data.sh)"
[[ -L store ]] \
  && ok "store/ linked"                || bad "store/ (run: bash scripts/link_external_data.sh)"

echo "== env =="
python3 -c "import memos, chonkie" 2>/dev/null \
  && ok "memos + chonkie importable"   || bad "memos + chonkie (pip install MemoryOS chonkie)"

echo "== wiring =="
got=$(EXTRA_ROUTERS="bm25 dense cot selfask" python3 -c "
import sys; sys.path.insert(0, 'scripts')
import evaluate_validation as ev
print(' '.join(ev.ROUTERS))" 2>/dev/null)
for arm in bm25 dense cot selfask; do
  case " $got " in *" $arm "*) ok "arm reaches evaluator: $arm";;
                   *) bad "arm reaches evaluator: $arm";;
  esac
done

echo
if [[ $fail -eq 0 ]]; then
  echo "Upload looks complete; safe to run."
else
  echo "Upload is INCOMPLETE -- re-upload before running, or the run will" >&2
  echo "finish normally and reproduce the previous result." >&2
  exit 1
fi
