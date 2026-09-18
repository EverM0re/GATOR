"""Download SMALL SUBSETS of the 5 datasets into data/raw + data/resources.

Subset-first: we only fetch what the first `--subset` QA actually reference, so
this runs on a laptop. To go full-scale, raise --subset (and --docs) or edit
config.datasets.

Layout produced:
    data/raw/<dataset>/...            # raw QA files (parquet/jsonl)
    data/resources/<dataset>/...      # page images (png/jpg) referenced by QA

Usage:
    python -m scripts.download_data --config config/file_router.yaml --subset 60
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
import zipfile

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_router.config import load_config          # noqa: E402
from file_router.utils import banner, dbg, info, warn  # noqa: E402


def _hf_download(repo, filename, repo_type="dataset"):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type=repo_type)


# --------------------------------------------------------------------------- UniDoc
def download_unidoc(cfg, subset_qa, subset_docs):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    domain = cfg.datasets.unidoc_domain
    repo = "Salesforce/UniDoc-Bench"
    raw_dir = os.path.join(cfg.paths.raw_dir, "unidoc")
    res_dir = os.path.join(cfg.paths.resources_dir, "unidoc")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    info(f"[unidoc] downloading {domain} QA parquet...")
    pqt = _hf_download(repo, f"data/{domain}-00000-of-00001.parquet")
    all_rows = pq.read_table(pqt).to_pylist()
    t = []
    selected_docs = set()
    for row in all_rows:
        paths = (row.get("gt_image_paths") or row.get("longdoc_image_paths") or [])
        parts = paths[0].split("/") if paths else []
        doc_key = parts[2] if len(parts) > 2 else (paths[0] if paths else "")
        if (doc_key not in selected_docs and subset_docs and
                len(selected_docs) >= subset_docs):
            continue
        if doc_key:
            selected_docs.add(doc_key)
        t.append(row)
        if subset_qa and len(t) >= subset_qa:
            break
    # save raw QA
    import json
    with open(os.path.join(raw_dir, "qa.json"), "w") as f:
        json.dump(t, f)

    # collect the page images we actually need (gt + a few longdoc pages)
    needed = set()
    docs = set()
    for r in t:
        for p in (r.get("gt_image_paths") or []):
            needed.add(p); docs.add(p.split("/")[2] if len(p.split("/")) > 2 else p)
        # cap longdoc pages per doc to keep it small
        for p in (r.get("longdoc_image_paths") or [])[:8]:
            needed.add(p)
    info(f"[unidoc] need {len(needed)} page images across ~{len(docs)} docs")
    ok = 0
    for rel in sorted(needed):
        try:
            src = hf_hub_download(repo, rel, repo_type="dataset")
            dst = os.path.join(res_dir, os.path.basename(rel))
            if not os.path.exists(dst):
                import shutil; shutil.copy(src, dst)
            ok += 1
            dbg(f"[unidoc] image {rel} -> {os.path.basename(dst)}")
        except Exception as e:  # noqa: BLE001
            warn(f"[unidoc] missing image {rel}: {repr(e)[:80]}")
    info(f"[unidoc] fetched {ok}/{len(needed)} images, {len(t)} QA")


# --------------------------------------------------------------------------- MMDocRAG
def download_mmdocrag(cfg, subset_qa, subset_docs):
    import json
    repo = "MMDocIR/MMDocRAG"
    raw_dir = os.path.join(cfg.paths.raw_dir, "mmdocrag")
    res_dir = os.path.join(cfg.paths.resources_dir, "mmdocrag")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    info("[mmdocrag] downloading dev_20.jsonl...")
    jl = _hf_download(repo, "dev_20.jsonl")
    rows = []
    selected_docs = set()
    with open(jl) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                doc_key = row.get("doc_name", f"doc{row.get('q_id')}")
                if (doc_key not in selected_docs and subset_docs and
                        len(selected_docs) >= subset_docs):
                    continue
                selected_docs.add(doc_key)
                rows.append(row)
            if subset_qa and len(rows) >= subset_qa:
                break
    with open(os.path.join(raw_dir, "qa.json"), "w") as f:
        json.dump(rows, f)

    # image quotes reference images/<name>.jpg inside images.zip — extract only needed
    needed = set()
    for r in rows:
        for iq in (r.get("img_quotes") or []):
            p = iq.get("img_path")
            if p:
                needed.add(p)
    info(f"[mmdocrag] need {len(needed)} quote images; downloading images.zip (one-time)...")
    try:
        zp = _hf_download(repo, "images.zip")
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
            ok = 0
            for rel in needed:
                cand = rel if rel in names else (rel.lstrip("./"))
                # try a few normalizations
                match = None
                for n in (rel, cand, os.path.basename(rel)):
                    hits = [x for x in names if x.endswith(n)]
                    if hits:
                        match = hits[0]; break
                if match is None:
                    warn(f"[mmdocrag] no zip entry for {rel}")
                    continue
                data = z.read(match)
                dst = os.path.join(res_dir, os.path.basename(rel))
                with open(dst, "wb") as out:
                    out.write(data)
                ok += 1
                dbg(f"[mmdocrag] extracted {match} -> {os.path.basename(dst)}")
            info(f"[mmdocrag] extracted {ok}/{len(needed)} images, {len(rows)} QA")
    except Exception as e:  # noqa: BLE001
        warn(f"[mmdocrag] images.zip failed ({repr(e)[:80]}); text-only fallback")


# --------------------------------------------------------------------------- ViDoRe
def download_vidore(cfg, subset_qa, subset_docs):
    import pyarrow.parquet as pq
    repo = "vidore/colpali_train_set"
    raw_dir = os.path.join(cfg.paths.raw_dir, "vidore")
    res_dir = os.path.join(cfg.paths.resources_dir, "vidore")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    info("[vidore] downloading test parquet (500 rows, images embedded)...")
    pqt = _hf_download(repo, "data/test-00000-of-00001.parquet")
    rows = pq.read_table(pqt).to_pylist()[: max(subset_qa, 30)]
    import json
    meta = []
    ok = 0
    for i, r in enumerate(rows):
        img = r.get("image")
        fn = f"vidore_{i:04d}.jpg"
        if isinstance(img, dict) and img.get("bytes"):
            with open(os.path.join(res_dir, fn), "wb") as out:
                out.write(img["bytes"])
            ok += 1
        meta.append({"idx": i, "query": r.get("query"), "answer": r.get("answer"),
                     "image_file": fn, "source": r.get("source")})
    with open(os.path.join(raw_dir, "qa.json"), "w") as f:
        json.dump(meta, f)
    info(f"[vidore] wrote {ok} images, {len(meta)} queries")


# --------------------------------------------------------------------------- REAL-MM-RAG
def download_real_mm_rag(cfg, subset_qa, subset_docs):
    import pyarrow.parquet as pq
    repo = cfg.datasets.real_mm_rag_repo
    raw_dir = os.path.join(cfg.paths.raw_dir, "real_mm_rag")
    res_dir = os.path.join(cfg.paths.resources_dir, "real_mm_rag")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    info(f"[real_mm_rag] downloading {repo} test parquet...")
    pqt = _hf_download(repo, "data/test-00000-of-00002.parquet")
    rows = pq.read_table(pqt).to_pylist()[: max(subset_qa, 30)]
    import json
    meta = []
    ok = 0
    for i, r in enumerate(rows):
        img = r.get("image")
        fn = f"real_{i:04d}.png"
        if isinstance(img, dict) and img.get("bytes"):
            with open(os.path.join(res_dir, fn), "wb") as out:
                out.write(img["bytes"])
            ok += 1
        meta.append({"idx": i, "query": r.get("query"), "answer": r.get("answer"),
                     "image_file": fn, "image_filename": r.get("image_filename")})
    with open(os.path.join(raw_dir, "qa.json"), "w") as f:
        json.dump(meta, f)
    info(f"[real_mm_rag] wrote {ok} images, {len(meta)} queries")


# --------------------------------------------------------------------------- LoCoMo-Refined
def download_locomo_refined(cfg, subset_qa, subset_docs):
    import json
    import urllib.request
    raw_dir = os.path.join(cfg.paths.raw_dir, "locomo_refined")
    os.makedirs(raw_dir, exist_ok=True)
    url = ("https://raw.githubusercontent.com/mem-eval-suite/LoCoMo_refined/"
           "main/data/raw/locomo_refined.json")
    dst = os.path.join(raw_dir, "locomo_refined.json")
    info("[locomo_refined] downloading from GitHub...")
    try:
        urllib.request.urlretrieve(url, dst)
        with open(dst) as f:
            data = json.load(f)
        n = len(data) if isinstance(data, list) else "?"
        info(f"[locomo_refined] saved ({n} records)")
    except Exception as e:  # noqa: BLE001
        warn(f"[locomo_refined] download failed: {repr(e)[:120]}")


_DISPATCH = {
    "unidoc": download_unidoc,
    "mmdocrag": download_mmdocrag,
    "vidore": download_vidore,
    "real_mm_rag": download_real_mm_rag,
    "locomo_refined": download_locomo_refined,
}


def _existing_qa_count(cfg, dataset: str) -> int:
    """How many QA rows are already downloaded for this dataset."""
    path = os.path.join(cfg.paths.unified_dir, dataset, "qa.jsonl")
    if not os.path.exists(path):
        return 0
    try:
        with open(path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _would_downgrade(cfg, targets, subset_qa: int):
    """Datasets whose on-disk data is larger than what this run would fetch."""
    if not subset_qa:          # 0 means "everything"; never a downgrade
        return []
    out = []
    for name in targets:
        existing = _existing_qa_count(cfg, name)
        # Small slack: per-dataset caps mean the counts never match exactly.
        if existing > subset_qa * 1.2:
            out.append((name, existing, subset_qa))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/file_router.yaml")
    ap.add_argument("--subset", type=int, default=None, help="max QA per dataset")
    ap.add_argument("--docs", type=int, default=None, help="max docs per dataset")
    ap.add_argument("--only", nargs="*", default=None, help="only these datasets")
    ap.add_argument("--force", action="store_true",
                    help="download even if it would shrink existing data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    subset_qa = args.subset if args.subset is not None else cfg.datasets.subset_qa
    subset_docs = args.docs if args.docs is not None else cfg.datasets.subset_docs

    targets = (args.only if args.only
               else list(cfg.datasets.enabled) + list(cfg.datasets.retrieval_only))

    # Downloading OVERWRITES data/raw, and the scale check runs afterwards, so a
    # smoke run silently destroys a medium corpus and only reports it once the
    # damage is done.  Refuse to shrink existing data unless asked explicitly.
    downgrades = _would_downgrade(cfg, targets, subset_qa)
    if downgrades and not args.force:
        for name, existing, requested in downgrades:
            warn(f"[{name}] on disk has {existing} QA; this run wants "
                 f"{requested}")
        warn("refusing to overwrite larger data with a smaller subset. "
             "Re-run with --force to download anyway, or raise datasets.subset_qa.")
        return

    banner(f"DOWNLOAD subset_qa={subset_qa} docs={subset_docs} datasets={targets}")
    for name in targets:
        fn = _DISPATCH.get(name)
        if fn is None:
            warn(f"unknown dataset '{name}', skipping")
            continue
        try:
            fn(cfg, subset_qa, subset_docs)
        except Exception as e:  # noqa: BLE001
            warn(f"[{name}] download failed: {repr(e)[:160]}")
    info("download stage done.")


if __name__ == "__main__":
    main()
