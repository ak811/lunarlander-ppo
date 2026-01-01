from __future__ import annotations

import csv
import json
from typing import Any, Dict, Optional


class CSVWriter:
    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames = None

    def write(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


class JSONLWriter:
    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "w", encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass
