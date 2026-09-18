#!/usr/bin/env bash
# Clone the host memory/RAG systems File_Router mounts onto.
#
# Deliberately clones OUTSIDE File_Router: this repo gets re-uploaded to the
# server on every iteration, and these hosts are large and rarely change.
# Default target is a sibling directory; override with HOSTS_DIR.
#
#   bash integrations/setup_hosts.sh              # clone only
#   INSTALL=1 bash integrations/setup_hosts.sh    # clone + pip install
set -euo pipefail

cd "$(dirname "$0")/.."
HOSTS_DIR="${HOSTS_DIR:-$(cd .. && pwd)/memory_hosts}"
INSTALL="${INSTALL:-0}"
mkdir -p "$HOSTS_DIR"

echo "[hosts] target: $HOSTS_DIR"

clone() {
  name="$1"; url="$2"
  if [[ -d "$HOSTS_DIR/$name/.git" ]]; then
    echo "[hosts] $name already cloned; pulling"
    git -C "$HOSTS_DIR/$name" pull --ff-only || echo "[hosts] $name pull skipped"
  else
    echo "[hosts] cloning $name"
    git clone --depth 1 "$url" "$HOSTS_DIR/$name" || {
      echo "[hosts] WARNING: failed to clone $name; continuing"; return 0; }
  fi
}

clone mem0         https://github.com/mem0ai/mem0.git
clone a-mem        https://github.com/agiresearch/A-mem.git
clone rag-anything https://github.com/HKUDS/RAG-Anything.git
clone memverse     https://github.com/KnowledgeXLab/MemVerse.git

cat > "$HOSTS_DIR/HOSTS.md" <<'NOTES'
# Host systems

Cloned by `File_Router/integrations/setup_hosts.sh`. Kept outside File_Router so
re-uploading the router does not drag these along.

| System | Install | Retrieval seam | Scores? | Multimodal |
|---|---|---|---|---|
| mem0 | `pip install mem0ai` | `Memory.search(query, top_k=, filters={"user_id":...})` | yes (similarity) | image→text at ingest only |
| A-Mem | `pip install -e a-mem` | `AgenticMemorySystem.search(query, k)` | yes (Chroma **distance**, lower=better) | no (text only) |
| RAG-Anything | `pip install raganything` | `aquery(q, mode, only_need_context=True)` | **no** | figures/tables → text descriptions |
| MemVerse | server only (FastAPI) | `orchestrator.rag_retrieve()` or `POST /query` | **no** | captions; **no PDF support** |

Notes that shape the adapters:
- mem0 v2 `search()` rejects top-level `user_id`; pass `filters={"user_id": ...}` and use `top_k`, not `limit`.
- A-Mem returns Chroma *distances*, the opposite polarity of mem0's score.
- A-Mem has no retriever config hook; assign `.retriever` after construction, and note
  its constructor calls `client.reset()`, and `consolidate_memories()` re-instantiates one.
- RAG-Anything and MemVerse are both LightRAG wrappers, so one seam
  (`LightRAG.aquery(..., only_need_context=True)`) serves both. That mode returns a
  formatted string with fenced JSON blocks, which must be parsed.
- Neither LightRAG host exposes relevance scores at that layer; to get scores you must
  drop to `BaseVectorStorage.query()`.
NOTES

if [[ "$INSTALL" == "1" ]]; then
  echo "[hosts] installing python packages"
  pip install mem0ai raganything || echo "[hosts] WARNING: pip install failed"
  [[ -d "$HOSTS_DIR/a-mem" ]] && pip install -e "$HOSTS_DIR/a-mem" || true
fi

echo "[hosts] done -> $HOSTS_DIR"
echo "[hosts] see $HOSTS_DIR/HOSTS.md"
