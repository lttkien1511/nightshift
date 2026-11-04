"""Write a blog post from a brief in the background: first an outline, then the post.

The text arrives piece by piece, so the page can show it growing. One job at a time, shared
with Run now through the gate, because the model on this machine serves one request at a time.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from .compose import (
    BriefError,
    Inputs,
    build_inputs,
    check_post,
    clean_output,
    ensure_frame,
    ensure_points,
    flag_pairs,
    parse_voices,
    preview,
    restore,
    tidy_post,
    trim_cut_off,
)
from .gate import Gate
from .runner import Busy, RunnerError
from .storage import Store

# system prompt, user prompt, token cap, a callback for each piece (return False to stop)
Stream = Callable[[str, str, int, Callable[[str], object]], str]
PARTIAL_KEEP = 4000


@dataclass(frozen=True)
class PostRun:
    slug: str
    kind: str  # outline, write or voices
    model: str
    state: str  # running, done, cancelled or failed
    started: float
    finished: float = 0.0
    chars: int = 0
    partial: str = ""
    error: str = ""

    @property
    def running(self) -> bool:
        return self.state == "running"


class Composer:
    def __init__(
        self,
        store: Store,
        make_stream: Callable[[str], Stream],
        models: Sequence[str],
        gate: Gate,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.models = list(models)
        self.gate = gate
        self._make_stream = make_stream
        self._clock = clock
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._state: PostRun | None = None
        self._thread: threading.Thread | None = None
        self._inputs: Inputs | None = None
        self._raw = ""

    @property
    def state(self) -> PostRun | None:
        with self._lock:
            return self._state

    def start(self, slug: str, kind: str, model: str = "") -> PostRun:
        if kind not in {"outline", "write", "voices"}:
            raise RunnerError("unknown step")
        model = model or (self.models[0] if self.models else "")
        if model not in self.models:
            raise RunnerError("that model is not available")
        record = self.store.post(slug)
        if record is None:
            raise RunnerError("no such blog post")
        try:
            inputs = build_inputs(record, kind)
        except BriefError as exc:
            raise RunnerError(str(exc)) from exc
        with self._lock:
            if self._state is not None and self._state.running:
                raise Busy(f"{self._state.slug} is being written right now")
            if not self.gate.acquire(f"writing {slug}"):
                raise Busy(f"{self.gate.holder} right now")
            self._cancel.clear()
            self._inputs, self._raw = inputs, ""
            self._state = PostRun(slug, kind, model, "running", self._clock())
            started = self._state
        self._thread = threading.Thread(
            target=self._work, args=(slug, kind, model, inputs), daemon=True
        )
        self._thread.start()
        return started

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float = 10.0) -> None:
        """Block until the current job ends. Used by tests."""
        if self._thread is not None:
            self._thread.join(timeout)

    def _finish(self, state: str, error: str = "") -> None:
        with self._lock:
            if self._state is not None:
                self._state = replace(self._state, state=state, error=error, finished=self._clock())
        self.gate.release()

    def _on_text(self, piece: str) -> bool:
        with self._lock:
            self._raw += piece
            if self._state is not None and self._inputs is not None:
                shown = preview(self._raw, self._inputs)[-PARTIAL_KEEP:]
                self._state = replace(
                    self._state, chars=self._state.chars + len(piece), partial=shown
                )
        return not self._cancel.is_set()

    def _work(self, slug: str, kind: str, model: str, inputs: Inputs) -> None:
        try:
            raw = self._make_stream(model)(
                inputs.system, inputs.prompt, inputs.num_predict, self._on_text
            )
            if self._cancel.is_set():
                self._finish("cancelled")
                return
            text, redacted, invented = restore(clean_output(raw), inputs)
            if not text.strip():
                raise RunnerError("the model wrote nothing")
            fields: dict[str, object] = {"model": model, "error": ""}
            if kind == "outline":
                outline = ensure_points(
                    ensure_frame(text, inputs.language), inputs.points, inputs.language
                )
                fields.update(outline=outline, stage="outline")
            elif kind == "voices":
                voices = parse_voices(text)
                if not voices:
                    raise RunnerError("the model did not write usable samples; try again")
                fields.update(voices=voices)
            else:
                text, repeated = tidy_post(text, inputs.language)
                text, cut = trim_cut_off(text)
                checks = check_post(text, inputs, redacted, invented)
                if cut:
                    checks.insert(
                        0,
                        {
                            "level": "warn",
                            "text": "The model ran out of room and the last sentence was cut. "
                            "Ask for a medium length, or finish the ending yourself.",
                        },
                    )
                if repeated:
                    checks.append(
                        {"level": "info", "text": f"Removed {repeated} repeated sentence(s)."}
                    )
                fields.update(
                    post=text,
                    stage="done",
                    checks=checks,
                    flags=flag_pairs(text, inputs.source, inputs.mode == "expand"),
                )
            self.store.update_post(slug, **fields)
        except Exception as exc:
            self._finish("failed", f"{type(exc).__name__}: {exc}"[:160])
            return
        self._finish("done")
