"""Verify each store's visual index matches the configured query encoder.

A mismatch disables visual retrieval for an entire run while the run still
completes, so this is checked before spending hours rather than discovered in a
log afterwards.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_router.config import load_config  # noqa: E402
from file_router.encoders import VisualEncoder  # noqa: E402


def _find_probe_image(cfg):
    """Any ingested page image; used to confirm both encoders share a space."""
    for root in (Path(cfg.paths.resources_dir), Path(cfg.paths.data_dir)):
        if not root.exists():
            continue
        for pattern in ("**/*.png", "**/*.jpg"):
            for path in root.glob(pattern):
                return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/file_router.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    encoder = VisualEncoder(cfg.encoders.visual)
    requested = cfg.encoders.visual.backend
    if encoder.backend != requested:
        print(f"[FAIL] visual encoder fell back: requested {requested!r}, "
              f"got {encoder.backend!r}. Visual retrieval will be meaningless.")
        return 2

    probe, _ = encoder.encode_query("dimension probe")
    query_dim = int(np.asarray(probe).reshape(-1).shape[0])
    declared = int(getattr(getattr(encoder, "_model", None), "config", None)
                   and encoder._model.config.projection_dim or 0)
    print(f"[INFO] encoder={encoder.backend} query_dim={query_dim} "
          f"config.projection_dim={declared or 'n/a'}")
    if declared and query_dim != declared:
        print(f"[WARN] the text tower returns {query_dim}-d while the config "
              f"declares {declared}-d. If this fires, the encoder is not "
              f"extracting the projected embedding for this transformers "
              f"version.")

    # Matching dimensions are necessary but not sufficient: text-to-image
    # retrieval also requires both towers to share a space.  If only one tower
    # changed, rebuilding the index cannot fix retrieval.
    probe_image = _find_probe_image(cfg)
    if probe_image is not None:
        image_dim = encoder._measure_image_dim(str(probe_image))
        text_dim = encoder._measure_text_dim()
        print(f"[INFO] text_tower={text_dim} image_tower={image_dim} "
              f"(probe: {probe_image.name})")
        if image_dim and text_dim and image_dim != text_dim:
            print(f"[FAIL] the two towers disagree ({text_dim}-d text vs "
                  f"{image_dim}-d image), so text-to-image retrieval cannot "
                  f"work at all. Rebuilding the index will NOT help.")
            return 2
    else:
        print("[WARN] no page image found to probe the image tower")

    problems = 0
    checked = 0
    store_root = Path(cfg.paths.store_dir)
    for pooled_path in sorted(store_root.glob("*/visual_pooled.npy")):
        dataset = pooled_path.parent.name
        pooled = np.load(pooled_path)
        if pooled.shape[0] == 0:
            print(f"[SKIP] {dataset}: empty visual index")
            continue
        checked += 1
        if pooled.shape[1] != query_dim:
            print(f"[FAIL] {dataset}: index is {pooled.shape[1]}-d "
                  f"({pooled.shape[0]} vectors) but queries are {query_dim}-d")
            problems += 1
        else:
            print(f"[PASS] {dataset}: {pooled.shape[0]} vectors, "
                  f"{pooled.shape[1]}-d")

    if problems:
        print(f"[FAIL] {problems} store(s) mismatch the query encoder. "
              "Rebuild them with SKIP_INGEST=0 before trusting any result.")
        return 2
    if not checked:
        print("[WARN] no non-empty visual index found; visual retrieval is "
              "inactive for every dataset")
        return 1
    print(f"[PASS] {checked} visual index(es) match the query encoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
