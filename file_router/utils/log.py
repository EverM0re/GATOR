"""Unified logger for File_Router.

Design goals (per DESIGN.md §5):
  - One logger, level controlled by env LOG_LEVEL (DEBUG = verbose, INFO = quiet).
  - All *temporary debugging* output goes through `dbg(...)`, which prefixes
    every line with `[DBG]`. To strip debug noise later you can either:
        export LOG_LEVEL=INFO          # silences it at runtime, OR
        grep -v '\\[DBG\\]'  / delete dbg() calls   # remove at source
    Business-critical messages use `info/warn/error` and are never prefixed
    with [DBG], so removing debug output never touches program logic.
"""

from __future__ import annotations

import logging
import os
import sys

_LOGGER_NAME = "file_router"
_DBG_PREFIX = "[DBG] "


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


_log = get_logger()


def dbg(msg: str) -> None:
    """Removable debug line. Prefixed with [DBG]; only shown when LOG_LEVEL=DEBUG."""
    _log.debug(_DBG_PREFIX + msg)


def info(msg: str) -> None:
    _log.info(msg)


def warn(msg: str) -> None:
    _log.warning(msg)


def error(msg: str) -> None:
    _log.error(msg)


def banner(title: str) -> None:
    """A visible section header (always shown)."""
    bar = "=" * 70
    _log.info(bar)
    _log.info(title)
    _log.info(bar)
