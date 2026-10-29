"""Run one draft through a model right now, in a background thread.

The nightly worker is the main path. This is for a short draft you do not want to wait a night
for. One run at a time: the model on this machine can only serve one request at a time anyway.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from .chunking import EDITABLE_KINDS, split
from .masking import MaskingError, mask
from .pipeline import Editor, process_document
from .results import never_answered, result_payload
from .storage import Store


class RunnerError(ValueError):
    """The run cannot start. The message is safe to show to the user."""


class Busy(RunnerError):
    pass


class _Cancelled(Exception):
    pass


@dataclass(frozen=True)
class RunState:
    slug: str
    model: str
    total: int
    done: int
    state: str  # running, done, cancelled or failed
    started: float
    finished: float = 0.0
    error: str = ""

    @property
    def running(self) -> bool:
        return self.state == "running"


class Runner:
    def __init__(
        self,
        store: Store,
        make_editor: Callable[[str], Editor],
        models: Sequence[str],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.models = list(models)
        self._make_editor = make_editor
        self._clock = clock
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._state: RunState | None = None
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> RunState | None:
        with self._lock:
            return self._state

    def start(self, slug: str, model: str = "") -> RunState:
        model = model or (self.models[0] if self.models else "")
        if model not in self.models:
            raise RunnerError("that model is not available")
        text = self.store.read(slug)
        if text is None:
            raise RunnerError("no such draft")
        try:
            total = sum(1 for c in split(mask(text).text) if c.kind in EDITABLE_KINDS)
        except MaskingError as exc:
            raise RunnerError(str(exc)) from exc
        with self._lock:
            if self._state is not None and self._state.running:
                raise Busy(f"{self._state.slug} is being edited right now")
            self._cancel.clear()
            self._state = RunState(slug, model, total, 0, "running", self._clock())
            started = self._state
        self._thread = threading.Thread(target=self._work, args=(slug, model, text), daemon=True)
        self._thread.start()
        return started

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float = 10.0) -> None:
        """Block until the current run ends. Used by tests."""
        if self._thread is not None:
            self._thread.join(timeout)

    def _update(self, **changes: object) -> None:
        with self._lock:
            if self._state is not None:
                self._state = replace(self._state, **changes)  # type: ignore[arg-type]

    def _finish(self, state: str, error: str = "") -> None:
        self._update(state=state, error=error, finished=self._clock())

    def _work(self, slug: str, model: str, text: str) -> None:
        try:
            inner = self._make_editor(model)

            def editor(block: str) -> str:
                if self._cancel.is_set():
                    raise _Cancelled
                try:
                    return inner(block)
                finally:
                    with self._lock:
                        if self._state is not None:
                            self._state = replace(self._state, done=self._state.done + 1)

            result = process_document(text, editor)
        except Exception as exc:
            self._finish("failed", f"{type(exc).__name__}: {exc}"[:160])
            return
        if self._cancel.is_set():
            self._finish("cancelled")
            return
        reason = never_answered(result)
        if reason:
            self._finish("failed", reason[:160])
            return
        self.store.write_result(slug, model, result_payload(model, text, result))
        self._finish("done")
