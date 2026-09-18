# GATOR: Granularity-Aware Token-Optimized Routing for Efficient Multimodal Retrieval

Multimodal RAG systems decide *which* evidence to retrieve, but not *how
expensively* to read it. The same document page can enter a prompt as a one-line
caption (~140 token-equivalents), as extracted full text (~350), or as a
rendered page image (~1,540). GATOR makes that choice explicit: it treats the
granularities of one page as competing purchases of the same evidence and
selects among them under a token budget.

Because the selector reads only three fields per candidate — a relevance score,
a token cost, and a group identifier — it can be mounted on an existing
retriever or memory system without retraining or re-indexing it.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two system packages are needed for ingestion:

```bash
# macOS
brew install tesseract poppler
# Debian / Ubuntu
sudo apt-get install tesseract-ocr poppler-utils
```

## Data

```bash
bash download_data.sh
```

This downloads both corpora, builds document-grouped train/test splits, and
ingests every page into the three-tier cost ladder. Ingestion renders pages,
runs OCR where there is no text layer, and encodes both towers; it is the slow
step and makes **no LLM calls**. For a quick check, run a subset first:

```bash
SUBSET=300 DOCS=80 bash download_data.sh
```

## Running the experiments

GATOR needs an OpenAI-compatible endpoint for answer generation and for the
LLM judge. The same endpoint serves both; no other model is required.

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=Qwen3-VL-8B-Instruct
export LLM_API_KEY=...          # omit if your server needs none

bash run_experiments.sh                    # single seed
SEEDS="42 43 44" bash run_experiments.sh   # three seeds, as reported
```

Each run writes a Markdown report plus `summary.json`, `per_question.jsonl` and
`retrieval_details.jsonl` under `runs/seed<N>/validation_<timestamp>/`. No API
key is written to any of them.

## Layout

```
file_router/
  ingestion/     builds the three-tier ladder and assigns costs once, at ingest
  retrieval/     text and visual recall, fused by reciprocal rank
  router/        the scoring network and the submodular selector
  training/      label generation, losses, and the training loop
integrations/    adapters that mount the selector on external memory systems
scripts/         pipeline stages and report generation
config/          all hyperparameters (file_router.yaml)
```

The selector is the part to read first: `file_router/router/selector.py`
implements the family-mutex constraint, the page-coverage quota, and the
expensive-tier admission rule described in the paper.

## Mounting on your own system

```python
from file_router.plugin.api import CostRouter, Candidate

router = CostRouter.with_defaults()

# Three granularities for each of four retrieved pages.
candidates = []
for page, score in enumerate([0.86, 0.81, 0.76, 0.71], start=1):
    candidates += [
        Candidate(id=f"p{page}-caption", score=score - 0.06, cost=140,
                  group_key=f"page-{page}"),
        Candidate(id=f"p{page}-text",    score=score - 0.08, cost=350,
                  group_key=f"page-{page}"),
        Candidate(id=f"p{page}-image",   score=score,        cost=1540,
                  group_key=f"page-{page}"),
    ]

report = router.select(candidates)
report.selected          # one granularity per page, never two
report.selected_cost     # total token-equivalents
report.cost_saving_rate  # saving against sending the richest of each
```

Candidates sharing a `group_key` are treated as substitutes, so at most one of
them is ever selected. In the example above the router buys page images for the
two best-ranked pages and demotes the rest to captions, which is the behaviour
the budget is meant to produce.

Scores are min-max normalised per query, which is what makes the admission rule
behave identically across hosts whose scoring conventions differ.

## Configuration

Everything is in `config/file_router.yaml`. The settings that matter most:

| Key | Meaning |
| --- | --- |
| `router.max_total_cost` | Token budget per query |
| `router.min_distinct_pages` | Pages that must be covered before selection may stop |
| `router.expensive_tier_min_probability_share` | Relative bar an expensive unit must clear |
| `router.submod_cost_beta` | How strongly marginal gain is discounted by price |
| `trainer.architecture` | `linear`, `mlp1`, `mlp2` (shipped), or `mlp3` |

Environment variables override the file for a single run; see
`scripts/run_validation.py` for the full list.

## Tests

```bash
pytest tests/ -q
```

## License

MIT.
