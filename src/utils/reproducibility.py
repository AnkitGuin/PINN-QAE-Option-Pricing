"""Centralised seeding and environment recording for reproducibility.

Every stochastic component (data generation, network initialisation, collocation
sampling, Monte Carlo, quantum simulator) derives its seed from a single master
seed stored in the configuration, so runs are bit-for-bit reproducible on the
same platform.  All manifests record the seeds actually used plus the software
versions and hardware description.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any, Dict, Optional


def derive_seed(master_seed: int, *parts: Any) -> int:
    """Deterministically derive a sub-seed from a master seed and labels.

    Uses sha256 over the master seed and the label parts so that sub-seeds for
    different purposes (e.g. ``('torch_init', 3)``) never collide in practice,
    while remaining fully reproducible.
    """
    payload = json.dumps({"master_seed": int(master_seed), "parts": list(parts)}, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np

    np.random.seed(seed % (2**32))
    try:  # torch is optional for pure classical runs
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a declared dependency
        pass


def environment_manifest(seed: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Machine-readable description of the software/hardware environment."""
    info: Dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "seed": seed,
    }
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except ImportError:  # pragma: no cover
        info["numpy"] = None
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        info["scipy"] = None
    try:
        import pandas

        info["pandas"] = pandas.__version__
    except ImportError:  # pragma: no cover
        info["pandas"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover
        # ImportError, and the platform torch+qiskit static-TLS conflict
        # (torch cannot load in a process where qiskit-aer is loaded);
        # recorded rather than raised so manifests never fail experiments
        info["torch"] = None
        info["torch_load_error"] = type(exc).__name__
    try:
        import qiskit

        info["qiskit"] = qiskit.__version__
    except ImportError:  # pragma: no cover
        info["qiskit"] = None
    try:
        import qiskit_aer

        info["qiskit_aer"] = qiskit_aer.__version__
    except ImportError:  # pragma: no cover
        info["qiskit_aer"] = None
    if extra:
        info.update(extra)
    return info


def write_manifest(path: str | os.PathLike, manifest: Dict[str, Any]) -> str:
    """Write a manifest as pretty JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
    return str(path)


def dataset_version(dataset_path: str | os.PathLike, config_hash_value: str) -> str:
    """Stable dataset version identifier: config hash + file fingerprint."""
    dataset_path = Path(dataset_path)
    fingerprint = hashlib.sha256()
    if dataset_path.exists():
        fingerprint.update(dataset_path.read_bytes())
    else:
        fingerprint.update(b"missing")
    fingerprint.update(config_hash_value.encode("utf-8"))
    return fingerprint.hexdigest()[:16]
