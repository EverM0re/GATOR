# Mounting on external hosts (GATOR as a plug-in)

Core claim: **"which evidence to retrieve" and "at which granularity to read it"
are two orthogonal decisions.** Existing memory / RAG systems solve the first;
this module solves the second, so the two compose rather than compete.

How the controlled comparison is set up: the host's retriever is **not
replaced** (it *is* the control arm); we only insert the cost router between
the retrieved hits and the LLM. Any difference therefore comes from granularity
selection alone.

## 1. Download the host systems

```bash
bash integrations/setup_hosts.sh              # clone only
INSTALL=1 bash integrations/setup_hosts.sh    # clone + pip install
```

By default they are cloned into `../memory_hosts/` (**outside** the project
tree, so they are not re-uploaded on every sync). Override with `HOSTS_DIR=/path`.

## 2. Prepare the data (mind where it comes from)

```bash
python3 -m scripts.prepare_mount_data --datasets mmdocrag unidoc
```

The default is `--source auto`, which **reads from `store/<dataset>/nodes.sqlite`
first**, because the text produced during ingestion (in particular UniDoc's OCR
output) exists only in the node store and is never written back to
`data/unified/`. The store also carries the **real ingestion costs and tiers**,
which is more accurate than re-estimating them from character counts.

- Read from the store → records carry `tier` / `cost` / `group_key`, so several
  tiers of the same page are recognised by the router as *the same evidence at
  different prices*.
- Store has no text → falls back to `data/unified/` automatically and prints a
  notice.
- UniDoc still yields 0 records → that ingestion run had OCR disabled
  (`ingestion.ocr.enabled: true`).

Note: `data/mount/` is a generated artifact and is cleared when the project is
re-uploaded; this step has to be re-run afterwards.

## 3. What the four hosts actually look like

The signatures below are taken from the cloned sources, not inferred from docs:

| System | Retrieval entry point | Relevance score | Multimodal | Mounting effort |
|---|---|---|---|---|
| **mem0** | `Memory.search(q, top_k=, filters={"user_id":...})` | yes (similarity) | image→text (at ingestion only) | low |
| **A-Mem** | `AgenticMemorySystem.search(q, k)` | yes (**Chroma distance, lower is better**) | no | low |
| **RAG-Anything** | `aquery(q, mode, only_need_context=True)` | **none** | figures→text descriptions | medium |
| **MemVerse** | `POST /query` or `orchestrator.rag_retrieve()` | **none** | captions only, **no PDF support** | high (server-side only) |

Pitfalls worth knowing:

- **mem0 v2**: `search()` rejects a top-level `user_id`; it must go through
  `filters=`. The parameter is `top_k`, not `limit`.
- **A-Mem**: `score` is a distance, so its polarity is inverted relative to the
  other hosts — the adapter flips it. There is no injection point for a custom
  retriever, and the constructor calls `client.reset()`, which wipes the
  collection. Every insertion issues an LLM call.
- **RAG-Anything / MemVerse**: both wrap LightRAG and share an adapter base
  class. `only_need_context=True` returns a **formatted string with JSON
  fences** that has to be parsed, and that layer **discards the vector
  distances**, so we can only synthesise a score from the rank position. This is
  a known limitation of the experiments on these two hosts and should be stated
  when reporting them.
- **MemVerse**: there is no pip package; it is a FastAPI service that must be
  started first. `POST /query` unconditionally generates an answer, which costs
  one extra LLM call.

## 4. The host's LLM endpoint (read automatically, no extra setup)

mem0 and A-Mem both default to **api.openai.com**. The adapters default to
`local=True` and **read `base_url` / `model` / `api_key` straight from the
`llm:` section of `config/file_router.yaml`** — the same configuration the
pipeline uses, so nothing needs to be exported separately.

Precedence: environment variables > `config/file_router.yaml`. Set the
environment variables only to override temporarily.

- **mem0**: the LLM points at the configured endpoint; the embedder is swapped
  for a local sentence-transformers model (vLLM serves chat completions but no
  embeddings endpoint).
- **A-Mem**: `llm_controller.py:22` hard-codes `OpenAI(api_key=...)` with **no
  `base_url` parameter**, so the client can only be replaced after construction.
  The adapter does this and prints a WARNING if it fails.
- **A failed insertion aborts the run**: an empty memory store retrieves
  nothing, yet still produces a table showing "100% cost reduction". That is not
  a result, so the script refuses to continue.

## 5. Run a mounting experiment

```bash
python3 -m scripts.run_mount_experiment \
  --host mem0 \
  --corpus data/unified/mmdocrag/corpus_flat.jsonl \
  --questions data/unified/mmdocrag/qa.jsonl \
  --max-questions 50 --answer \
  --out mount_mem0.json
```

`--answer` generates and scores answers for both arms — **a cost reduction only
counts if quality does not drop**, otherwise the claim does not hold. Omit it to
compare cost alone.

## 6. Adding your own host

Two methods are enough:

```python
from integrations.base import HostAdapter, HostHit

class MyAdapter(HostAdapter):
    name = "my-system"

    def add(self, item): ...                      # insert into the host

    def retrieve(self, query, top_k):             # call the host's own retriever
        return [HostHit(id=..., text=..., score=..., doc_id=...)]
```

The base class handles tier expansion, cost routing, and the control-arm
computation. If the host can supply page images, override `expand_tiers()` to
add the screenshot tier.

## 7. Using the selector directly (without an adapter)

```python
from file_router.plugin import CostRouter, Candidate

router = CostRouter.with_defaults()
report = router.select([
    Candidate(id="p3_full", text=page_text, cost=340, score=0.81,
              group_key="doc1:p3", tier="fulltext"),
    Candidate(id="p3_cap",  text=caption,   cost=48,  score=0.77,
              group_key="doc1:p3", tier="caption"),
])
print(report.cost_saving_rate, [c.id for c in report.selected])
```

`group_key` is the crucial field: candidates sharing it are treated as **the
same evidence at different prices**, and at most one of them is bought.

## Verified feasibility

On the real mmdocrag corpus (25 questions, simple retriever):

```
baseline 7805 tokens -> routed 905 tokens   cost reduction 88.4%
21.9 candidates -> 5.0 kept                 routing overhead 0.16 ms
tiers kept: {'fulltext': 40}
```

What is kept is fulltext rather than the cheapest caption tier — consistent with
the conclusion of the tier ablation (fulltext F1 0.3534 / cost 892; caption
0.1850 / cost 1625).
