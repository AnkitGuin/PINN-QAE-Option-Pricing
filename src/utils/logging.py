"""Lightweight structured logging helpers.

Experiment scripts log to both stdout and a run log file.  Per-epoch training
logs are written as CSV by the PINN trainer (``src/pinn/training.py``); this
module only provides ordinary application logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stdout and optionally to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        logger.addHandler(stream)
        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
            logger.addHandler(fh)
    return logger
