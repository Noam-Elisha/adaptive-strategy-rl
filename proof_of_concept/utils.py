"""Logging and general utilities."""

import csv
import os
import time
from typing import Dict, List


class CSVLogger:
    """Append-mode CSV logger — writes a header on first call, then rows."""

    def __init__(self, path: str, fieldnames: List[str]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: Dict):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


class Timer:
    """Simple wall-clock timer for FPS measurement."""

    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self):
        self._start = time.perf_counter()
