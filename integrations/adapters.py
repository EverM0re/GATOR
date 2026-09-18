"""Concrete adapters, one per host system.

Every signature below was read from the cloned sources, not from docs:
  mem0          mem0/memory/main.py:1379   search(query, *, top_k, filters, ...)
  A-Mem         agentic_memory/memory_system.py:432  search(query, k)
  RAG-Anything  raganything/query.py:128   aquery(query, mode, **kwargs)
  MemVerse      orchestrator.py:218        rag_retrieve(query, mode)

Imports are deferred into the constructors so this module can be imported (and
tested) without any host installed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from .base import HostAdapter, HostHit


DEFAULT_CONFIG_PATH = "config/file_router.yaml"


def llm_settings(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, str]:
    """Resolve the answer endpoint the same way the rest of the project does.

    Precedence is environment first, then `llm:` in the project YAML.  Reading
    the YAML matters because that is where these values are already configured;
    requiring the caller to re-export them as environment variables is a second
    place to get wrong.
    """
    settings = {"base_url": "", "model": "", "api_key": ""}
    try:
        from file_router.config import load_config

        llm = load_config(config_path).llm
        settings["base_url"] = str(getattr(llm, "base_url", "") or "")
        settings["model"] = str(getattr(llm, "model", "") or "")
        settings["api_key"] = str(getattr(llm, "api_key", "") or "")
    except Exception as exc:  # noqa: BLE001
        print(f"[hosts] could not read {config_path}: {repr(exc)[:120]}")

    settings["base_url"] = (os.environ.get("LLM_BASE_URL")
                            or settings["base_url"] or "http://127.0.0.1:8000/v1")
    settings["model"] = (os.environ.get("LLM_MODEL")
                         or settings["model"] or "gpt-4o-mini")
    settings["api_key"] = (os.environ.get("LLM_API_KEY")
                           or os.environ.get("OPENAI_API_KEY")
                           or settings["api_key"] or "EMPTY")
    return settings


def local_mem0_config(base_url: str = "", model: str = "",
                      api_key: str = "",
                      embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                      embedding_dims: int = 384,
                      collection_name: str = "file_router_eval",
                      qdrant_path: str = "",
                      config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Point mem0 at the project's own LLM endpoint instead of api.openai.com.

    mem0 defaults to OpenAI for BOTH the LLM and the embedder, so an
    unconfigured Memory() reaches out to OpenAI even when everything else in the
    experiment runs against a local vLLM.  The embedder stays on local
    sentence-transformers because vLLM serves a chat model, not embeddings.
    """
    resolved = llm_settings(config_path)
    base_url = base_url or resolved["base_url"]
    model = model or resolved["model"]
    api_key = api_key or resolved["api_key"]
    # A run-local store, so re-running does not append to a collection built
    # with different settings (which is how the dimension mismatch persists).
    qdrant_path = qdrant_path or os.path.join(
        tempfile.gettempdir(), f"mem0_{collection_name}_{embedding_dims}")
    return {
        "llm": {"provider": "openai",
                "config": {"model": model, "api_key": api_key,
                           "openai_base_url": base_url}},
        "embedder": {"provider": "huggingface",
                     "config": {"model": embed_model,
                                "embedding_dims": embedding_dims}},
        # The vector store defaults to 1536 (OpenAI's size) independently of the
        # embedder, so a local 384-dim model fails with a shape mismatch on the
        # first insert.  Both have to be told the dimension.
        "vector_store": {"provider": "qdrant",
                         "config": {"embedding_model_dims": embedding_dims,
                                    "collection_name": collection_name,
                                    "path": qdrant_path}},
    }


class Mem0Adapter(HostAdapter):
    """mem0 v2.

    Two v2 API details that silently break naive usage: `search()` rejects a
    top-level `user_id` (it must go in `filters`), and the limit parameter is
    `top_k`, not `limit`.
    """

    name = "mem0"

    def __init__(self, *, user_id: str = "file_router_eval",
                 config: Optional[Dict[str, Any]] = None,
                 local: bool = True, config_path: str = DEFAULT_CONFIG_PATH,
                 **kwargs):
        super().__init__(**kwargs)
        from mem0 import Memory

        self.user_id = user_id
        self.config_path = config_path
        if config is None and local:
            config = local_mem0_config(config_path=config_path)
        self.memory = Memory.from_config(config) if config else Memory()

    def add(self, item: Dict[str, Any]) -> None:
        # infer=False stores the text verbatim instead of running LLM fact
        # extraction, so ingest is deterministic and comparable across runs.
        # tier/cost/group_key ride along so retrieval can price candidates with
        # the pipeline's own ingestion costs rather than re-estimating them.
        self.memory.add(
            [{"role": "user", "content": item["text"]}],
            user_id=self.user_id,
            metadata={"doc_id": item.get("doc_id", ""),
                      "page": item.get("page"),
                      "tier": item.get("tier"),
                      "cost": item.get("cost"),
                      "group_key": item.get("group_key")},
            infer=False,
        )

    def retrieve(self, query: str, top_k: int) -> List[HostHit]:
        response = self.memory.search(
            query, top_k=top_k, filters={"user_id": self.user_id})
        results = response.get("results", response) if isinstance(
            response, dict) else response
        hits = []
        for r in results:
            meta = r.get("metadata") or {}
            hits.append(HostHit(
                id=str(r.get("id")),
                text=r.get("memory") or "",
                score=float(r.get("score") or 0.0),
                doc_id=str(meta.get("doc_id") or ""),
                metadata=meta,
            ))
        return hits


class AMemAdapter(HostAdapter):
    """A-Mem (agentic_memory).

    `search()` returns a ChromaDB *distance* in its `score` field -- lower is
    better, the opposite of every other host here.  Converting it is the whole
    reason this adapter cannot be shared with mem0.
    """

    name = "a-mem"

    def __init__(self, *, model_name: str = "all-MiniLM-L6-v2",
                 llm_backend: str = "openai", llm_model: str = "",
                 local: bool = True, config_path: str = DEFAULT_CONFIG_PATH,
                 **kwargs):
        super().__init__(**kwargs)
        from agentic_memory.memory_system import AgenticMemorySystem

        self.config_path = config_path
        resolved = llm_settings(config_path)
        llm_model = llm_model or resolved["model"]
        # A-Mem reads OPENAI_API_KEY inside its own constructor
        # (llm_controller.py:21) and raises before returning if it is unset, so
        # the post-construction redirect below never gets to run.  Our key lives
        # in the project config, not that variable, so export it first.
        if resolved["api_key"] and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = resolved["api_key"]
        self.memory = AgenticMemorySystem(
            model_name=model_name, llm_backend=llm_backend,
            llm_model=llm_model)
        if local and llm_backend == "openai":
            self._point_llm_at_local_endpoint()

    def _point_llm_at_local_endpoint(self) -> None:
        """Redirect A-Mem's LLM client to the local OpenAI-compatible endpoint.

        llm_controller.py:22 constructs `OpenAI(api_key=...)` with no base_url,
        so there is no config path to a local server -- the client has to be
        replaced after construction.  Without this, ingest calls api.openai.com
        (A-Mem calls an LLM on every add_note, with no infer=False escape).
        """
        try:
            from openai import OpenAI

            resolved = llm_settings(self.config_path)
            base_url, api_key = resolved["base_url"], resolved["api_key"]
            controller = getattr(self.memory, "llm_controller", None)
            target = getattr(controller, "llm", None) if controller else None
            if target is not None and hasattr(target, "client"):
                target.client = OpenAI(api_key=api_key, base_url=base_url)
                print(f"[a-mem] LLM redirected to {base_url}")
            else:
                print("[a-mem] WARNING: could not find llm_controller.llm.client; "
                      "ingest may call OpenAI directly")
        except Exception as exc:  # noqa: BLE001
            print(f"[a-mem] WARNING: local LLM redirect failed: {repr(exc)[:120]}")

    def add(self, item: Dict[str, Any]) -> None:
        # add_note() always calls an LLM (analysis + evolution); there is no
        # infer=False equivalent, so ingest here is not free.
        self.memory.add_note(item["text"])

    def retrieve(self, query: str, top_k: int) -> List[HostHit]:
        results = self.memory.search(query, k=top_k)
        if not results:
            return []
        distances = [float(r.get("score") or 0.0) for r in results]
        worst = max(distances) or 1.0
        hits = []
        for r, distance in zip(results, distances):
            # Map distance -> similarity in [0, 1]; only the ordering and the
            # relative spread matter downstream.
            hits.append(HostHit(
                id=str(r.get("id")),
                text=r.get("content") or "",
                score=1.0 - (distance / worst if worst else 0.0),
                metadata={"context": r.get("context"),
                          "keywords": r.get("keywords")},
            ))
        return hits


_FENCE = re.compile(r"-----(.+?)-----\s*```json\s*(.*?)```", re.DOTALL)


class LightRAGAdapter(HostAdapter):
    """Shared base for RAG-Anything and MemVerse, which both wrap LightRAG.

    `only_need_context=True` returns retrieved context WITHOUT generating an
    answer, but as a formatted string containing fenced JSON blocks rather than
    structured results -- so it has to be parsed back out.

    LightRAG drops vector distances during context assembly, so there is no
    relevance score at this layer.  Rank order is the only signal available;
    scores are synthesised from position, which is stated here because it
    weakens the comparison and should be reported as a limitation.
    """

    name = "lightrag"

    @staticmethod
    def parse_context(context: str) -> List[HostHit]:
        hits: List[HostHit] = []
        for section, blob in _FENCE.findall(context or ""):
            try:
                rows = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(rows, list):
                continue
            for rank, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                text = (row.get("content") or row.get("description") or "")
                if not text:
                    continue
                hits.append(HostHit(
                    id=f"{section.strip()}:{row.get('id', rank)}",
                    text=text,
                    # Rank-derived, NOT a real relevance score. See class docstring.
                    score=1.0 / (1.0 + rank),
                    doc_id=str(row.get("file_path") or ""),
                    metadata={"section": section.strip(), "rank": rank},
                ))
        return hits


class RAGAnythingAdapter(LightRAGAdapter):
    name = "rag-anything"

    def __init__(self, rag: Any, *, mode: str = "mix", **kwargs):
        super().__init__(**kwargs)
        self.rag = rag
        self.mode = mode

    def add(self, item: Dict[str, Any]) -> None:
        content = [{"type": "text", "text": item["text"],
                    "page_idx": item.get("page", 0)}]
        if item.get("image_path"):
            content.append({"type": "image", "img_path": item["image_path"],
                            "image_caption": [], "page_idx": item.get("page", 0)})
        self._await(self.rag.insert_content_list(
            content, file_path=item.get("doc_id", "doc")))

    def retrieve(self, query: str, top_k: int) -> List[HostHit]:
        # vlm_enhanced=False keeps the retrieve-only path deterministic: aquery
        # otherwise auto-routes to the VLM path when a vision model is set.
        context = self._await(self.rag.aquery(
            query, mode=self.mode, only_need_context=True,
            vlm_enhanced=False))
        return self.parse_context(context)[:top_k]

    @staticmethod
    def _await(coro):
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "RAGAnythingAdapter needs a sync context; call from outside an event loop")


class MemVerseAdapter(LightRAGAdapter):
    """MemVerse over HTTP.

    MemVerse ships no importable package -- it is a FastAPI app -- so this talks
    to a running server.  `POST /query` always generates an answer, which costs
    an LLM call we do not need; `rag_memory` in the response carries the
    retrieved context we actually want.
    """

    name = "memverse"

    def __init__(self, base_url: str = "http://127.0.0.1:8000",
                 *, mode: str = "hybrid", timeout: float = 120.0, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout

    def add(self, item: Dict[str, Any]) -> None:
        import requests

        files = {}
        if item.get("image_path"):
            files["image"] = open(item["image_path"], "rb")
        try:
            requests.post(f"{self.base_url}/insert",
                          data={"query": item["text"]}, files=files or None,
                          timeout=self.timeout).raise_for_status()
        finally:
            for handle in files.values():
                handle.close()

    def retrieve(self, query: str, top_k: int) -> List[HostHit]:
        import requests

        response = requests.post(
            f"{self.base_url}/query",
            data={"query": query, "mode": self.mode, "use_pm": False},
            timeout=self.timeout)
        response.raise_for_status()
        return self.parse_context(
            response.json().get("rag_memory") or "")[:top_k]


class MemOSAdapter(HostAdapter):
    """MemOS (MemTensor/MemOS, distributed on PyPI as `MemoryOS`).

    Mounting is unusually direct here: `Searcher.retrieve()` already returns
    `list[(item, score)]` and `post_retrieve()` already consumes that shape, so
    the router sits between them without adapting either side.

    `mode` must stay "fast". The "fine" path sends the query through
    `TaskGoalParser._parse_fine`, which calls an LLM to rewrite it -- that would
    place a generation step inside the retrieval we are measuring, and make the
    cost comparison meaningless.
    """

    name = "memos"
    RETRIEVAL_MODE = "fast"

    def __init__(self, *, user_id: str = "file_router_eval",
                 cube_name: str = "file_router_eval",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_token_counter: str = "word",
                 embedding_dims: int = 384,
                 **kwargs):
        super().__init__(**kwargs)
        from memos.configs.mem_os import MOSConfig
        from memos.mem_os.main import MOS

        self.user_id = user_id
        self.cube_name = cube_name
        settings = llm_settings()
        # The embedder is a separate service from the chat model: the local
        # vLLM serves one VLM and no embedding endpoint.  Use the same local
        # sentence-transformers model the rest of the pipeline already uses
        # (and that the mem0 adapter already mounts this way), so MemOS needs no
        # hosted embedding API.  MEMOS_EMBED_MODEL overrides it.
        embedding_model = os.environ.get("MEMOS_EMBED_MODEL", embedding_model)
        config = MOSConfig(
            user_id=user_id,
            chat_model={"backend": "openai",
                        "config": {"model_name_or_path": settings["model"],
                                   "api_key": settings["api_key"],
                                   "api_base": settings["base_url"]}},
            # MOSConfig validates mem_reader strictly and `chunker` is required
            # rather than defaulted; omitting it fails at construction, before
            # any work starts.
            mem_reader={"backend": "simple_struct",
                        "config": {"llm": {"backend": "openai",
                                           "config": {
                                               "model_name_or_path": settings["model"],
                                               "api_key": settings["api_key"],
                                               "api_base": settings["base_url"]}},
                                   "embedder": {"backend": "sentence_transformer",
                                                "config": {
                                                    "model_name_or_path":
                                                        embedding_model}},
                                   # "gpt2" makes the chunker fetch a tokenizer
                                   # from HuggingFace, and an unauthenticated
                                   # fetch turns into an interactive username
                                   # prompt that hangs the run forever.  A
                                   # plain word counter needs no download and
                                   # only decides chunk boundaries, which the
                                   # router never sees -- it scores whole
                                   # candidates returned by the host.
                                   "chunker": {"backend": "sentence",
                                               "config": {
                                                   "tokenizer_or_token_counter":
                                                       chunk_token_counter,
                                                   "chunk_size": 512,
                                                   "chunk_overlap": 128,
                                                   "min_sentences_per_chunk": 1}}}},
            enable_textual_memory=True,
        )
        self.mos = MOS(config)
        self.mos.create_user(user_id=user_id)
        # register_mem_cube treats a bare name as a remote repo id and tries to
        # `git clone https://huggingface.co/datasets/<name>`, which fails for a
        # cube that only exists locally.  Build the cube here and hand over a
        # real directory so nothing is fetched.
        self._cube_dir = os.path.join(
            tempfile.gettempdir(), f"memos_cube_{cube_name}")
        # init_from_dir reads config.json, so the directory existing is not the
        # condition that matters -- a half-built cube from an earlier attempt
        # would be skipped here and then fail on the missing file.
        if not os.path.isfile(os.path.join(self._cube_dir, "config.json")):
            shutil.rmtree(self._cube_dir, ignore_errors=True)
            from memos.configs.mem_cube import GeneralMemCubeConfig
            from memos.mem_cube.general import GeneralMemCube

            cube_cfg = GeneralMemCubeConfig.model_validate({
                "user_id": user_id,
                "cube_id": cube_name,
                "text_mem": {
                    "backend": "general_text",
                    "config": {
                        "extractor_llm": config.chat_model.model_dump(),
                        "embedder": config.mem_reader.config.embedder.model_dump(),
                        "vector_db": {
                            "backend": "qdrant",
                            "config": {
                                "collection_name": cube_name,
                                "vector_dimension": embedding_dims,
                                "distance_metric": "cosine",
                                "path": os.path.join(self._cube_dir, "qdrant"),
                            },
                        },
                    },
                },
                "act_mem": {"backend": "uninitialized", "config": {}},
                "para_mem": {"backend": "uninitialized", "config": {}},
            })
            os.makedirs(self._cube_dir, exist_ok=True)
            # Write config.json ourselves: dump() does not reliably emit it,
            # and init_from_dir treats its absence as a fatal error.
            cube_cfg.to_json_file(
                os.path.join(self._cube_dir, "config.json"))
            try:
                GeneralMemCube(cube_cfg).dump(self._cube_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"[memos] cube dump skipped: {repr(exc)[:120]}")
        self.mos.register_mem_cube(self._cube_dir, mem_cube_id=cube_name,
                                   user_id=user_id)

    def add(self, item: Dict[str, Any]) -> None:
        self.mos.add(
            memory_content=item["text"],
            user_id=self.user_id,
            mem_cube_id=self.cube_name,
            metadata={"doc_id": item.get("doc_id", ""),
                      "page": item.get("page"),
                      "tier": item.get("tier"),
                      "cost": item.get("cost"),
                      "group_key": item.get("group_key")},
        )

    def retrieve(self, query: str, top_k: int) -> List[HostHit]:
        response = self.mos.search(
            query=query, user_id=self.user_id, top_k=top_k,
            mode=self.RETRIEVAL_MODE,
        )
        results = (response.get("text_mem") or []) if isinstance(
            response, dict) else response
        # search() nests hits under one entry per cube.
        flat = []
        for entry in results:
            flat.extend(entry.get("memories", []) if isinstance(entry, dict)
                        else [entry])
        hits = []
        for r in flat[:top_k]:
            meta = (getattr(r, "metadata", None) or {})
            if not isinstance(meta, dict):
                meta = getattr(meta, "__dict__", {}) or {}
            hits.append(HostHit(
                id=str(getattr(r, "id", "") or meta.get("id", "")),
                text=str(getattr(r, "memory", "") or getattr(r, "content", "")),
                score=float(meta.get("relativity") or
                            getattr(r, "score", 0.0) or 0.0),
                doc_id=str(meta.get("doc_id") or ""),
                metadata=meta,
            ))
        return hits


ADAPTERS = {
    "mem0": Mem0Adapter,
    "a-mem": AMemAdapter,
    "rag-anything": RAGAnythingAdapter,
    "memverse": MemVerseAdapter,
    "memos": MemOSAdapter,
}
