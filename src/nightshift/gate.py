"""One model, one job at a time: editing a draft and writing a post share this gate."""

from __future__ import annotations

import threading


class Gate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holder: str | None = None

    @property
    def holder(self) -> str | None:
        with self._lock:
            return self._holder

    def acquire(self, holder: str) -> bool:
        with self._lock:
            if self._holder is not None:
                return False
            self._holder = holder
            return True

    def release(self) -> None:
        with self._lock:
            self._holder = None
