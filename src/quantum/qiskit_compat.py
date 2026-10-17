"""Qiskit compatibility helpers.

Handles the qiskit-aer ``libgomp`` static-TLS import issue seen on some Linux
platforms by preloading the bundled libgomp with RTLD_GLOBAL before qiskit_aer
is imported.  Import this module BEFORE importing qiskit_aer.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sysconfig
from typing import Optional

_PRELOADED = False


def preload_libgomp() -> Optional[str]:
    """Preload qiskit-aer's bundled libgomp; returns the path or None."""
    global _PRELOADED
    if _PRELOADED:
        return None
    candidates: list[str] = []
    for path in sysconfig.get_paths().values():
        if isinstance(path, str):
            candidates.extend(glob.glob(os.path.join(path, "qiskit_aer.libs", "libgomp*.so*")))
    for lib in candidates:
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            _PRELOADED = True
            return lib
        except OSError:
            continue
    return None


def import_qiskit():
    """Import and return (qiskit, qiskit_aer) with the compatibility shim."""
    preload_libgomp()
    import qiskit
    try:
        import qiskit_aer  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "qiskit-aer is required for the quantum module. "
            f"Import failed after libgomp preload attempt: {exc}"
        ) from exc
    return qiskit
