"""Config loader — reads config/file_router.yaml into nested SimpleNamespaces.

Same shape as cnem_v's loader, but does NOT eagerly create per-dataset store
dirs (those are created per dataset at ingest time).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import yaml


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str = "config/file_router.yaml") -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _to_ns(raw)
    os.makedirs(cfg.paths.store_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
    os.makedirs(cfg.paths.router_model_dir, exist_ok=True)
    return cfg
