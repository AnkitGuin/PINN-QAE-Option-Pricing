"""Serialisation helpers for results (JSON / CSV) with stable formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class _Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(path: str | Path, payload: Dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, cls=_Encoder)
    return str(path)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_rows_csv(path: str | Path, rows: List[Dict[str, Any]], columns: List[str] | None = None) -> str:
    """Save a list of flat dictionaries as CSV (used for training histories)."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(",".join(columns or []) + "\n", encoding="utf-8")
        return str(path)
    columns = columns or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    return str(path)


def read_rows_csv(path: str | Path) -> List[Dict[str, Any]]:
    import csv

    with open(path, "r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]
