"""Timing utilities for calibration / training / quantum runtime accounting."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict


@contextmanager
def timed(label: str = "", seconds_dict: Dict[str, float] | None = None):
    """Context manager measuring wall-clock seconds.

    Usage::

        with timed("ls_fit", times) as t:
            ...
        # times["ls_fit"] == elapsed seconds
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if seconds_dict is not None:
            seconds_dict[label] = elapsed


class Timer:
    """Simple stopwatch."""

    def __init__(self) -> None:
        self._start: float | None = None
        self.elapsed: float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is not None:
            self.elapsed += time.perf_counter() - self._start
            self._start = None
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def human_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, sec = divmod(seconds, 60.0)
    return f"{int(minutes)}m {sec:.1f}s"
