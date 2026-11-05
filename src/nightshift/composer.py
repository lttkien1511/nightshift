"""Write a blog post from a brief in the background: an outline, then the post.

Each step can be given to several models, one after the other. Every answer is kept as a
candidate the person can pick parts from. The text being built is replaced by an answer only
when asked to, or when there is nothing yet. One job at a time, shared with Run now through the
gate, because the model on this machine serves one request at a time.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
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
MAX_MODELS = 4
MAX_CANDIDATES = 12


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
    queue: tuple[str, ...] = ()  # every model of this job, in order
    position: int = 0  # which of them is running

    @property
    def running(self) -> bool:
        return self.state == "running"


def _with_candidate(record: Mapping[str, object], kind: str, answer: dict[str, object]) -> dict:
    """The candidates of a post with one more answer, keeping only the newest few."""
    found = record.get("candidates")
    book = (
        {k: list(v) for k, v in found.items() if isinstance(v, list)}
        if isinstance(found, dict)
        else {}
    )
    book[kind] = [*book.get(kind, []), {"id": uuid.uuid4().hex[:8], **answer}][-MAX_CANDIDATES:]
    return book


class Composer:
    def __init__(
        self,
        store: Store,
        make_stream: Callable[[str], Stream],
        models: Sequence[str] | Callable[[], Sequence[str]],
        gate: Gate,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self._models = models
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
    def models(self) -> list[str]:
        """The models a blog post may use. They can change while the server runs."""
        return list(self._models() if callable(self._models) else self._models)

    @property
    def state(self) -> PostRun | None:
        with self._lock:
            return self._state

    def start(
        self,
        slug: str,
        kind: str,
        model: str = "",
        models: Sequence[str] = (),
        replace: bool = True,
    ) -> PostRun:
        """Start a step with one model or several. ``replace`` lets the answer take the place
        of the outline or post being built; with several models it should be False, so the
        answers wait as candidates."""
        if kind not in {"outline", "write", "voices"}:
            raise RunnerError("unknown step")
        available = self.models
        chosen = list(dict.fromkeys(m for m in (*models, model) if m)) or available[:1]
        if not chosen or any(m not in available for m in chosen):
            raise RunnerError("that model is not available")
        if len(chosen) > MAX_MODELS:
            raise RunnerError(f"pick at most {MAX_MODELS} models")
        if kind == "voices":
            chosen = chosen[:1]
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
            self._state = PostRun(
                slug, kind, chosen[0], "running", self._clock(), queue=tuple(chosen)
            )
            started = self._state
        self._thread = threading.Thread(
            target=self._work, args=(slug, kind, chosen, inputs, replace), daemon=True
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

    def _work(
        self, slug: str, kind: str, chosen: list[str], inputs: Inputs, replace_text: bool
    ) -> None:
        failures: list[str] = []
        done = 0
        for position, model in enumerate(chosen):
            if position:
                with self._lock:
                    self._raw = ""
                    self._state = PostRun(
                        slug,
                        kind,
                        model,
                        "running",
                        self._clock(),
                        queue=tuple(chosen),
                        position=position,
                    )
            try:
                outcome = self._run_one(slug, kind, model, inputs, replace_text)
            except Exception as exc:
                label = f"{model}: " if len(chosen) > 1 else ""
                failures.append(f"{label}{type(exc).__name__}: {exc}")
                continue
            if outcome == "cancelled":
                self._finish("cancelled")
                return
            done += 1
        error = "; ".join(failures)[:300]
        self._finish("done" if done else "failed", error)

    def _run_one(self, slug: str, kind: str, model: str, inputs: Inputs, replace_text: bool) -> str:
        began = self._clock()
        raw = self._make_stream(model)(
            inputs.system, inputs.prompt, inputs.num_predict, self._on_text
        )
        if self._cancel.is_set():
            return "cancelled"
        text, redacted, invented = restore(clean_output(raw), inputs)
        if not text.strip():
            raise RunnerError("the model wrote nothing")
        record = self.store.post(slug) or {}
        seconds = round(self._clock() - began)
        fields: dict[str, object] = {"error": ""}
        if kind == "voices":
            voices = parse_voices(text)
            if not voices:
                raise RunnerError("the model did not write usable samples; try again")
            fields.update(model=model, voices=voices)
        elif kind == "outline":
            outline = ensure_points(
                ensure_frame(text, inputs.language), inputs.points, inputs.language
            )
            answer: dict[str, object] = {"model": model, "text": outline, "seconds": seconds}
            fields["candidates"] = _with_candidate(record, "outline", answer)
            if replace_text or not str(record.get("outline", "")).strip():
                fields.update(model=model, outline=outline, stage="outline")
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
            flags = flag_pairs(text, inputs.source, inputs.mode == "expand")
            answer = {
                "model": model,
                "text": text,
                "seconds": seconds,
                "checks": checks,
                "flags": flags,
            }
            fields["candidates"] = _with_candidate(record, "write", answer)
            if replace_text or not str(record.get("post", "")).strip():
                fields.update(model=model, post=text, stage="done", checks=checks, flags=flags)
        self.store.update_post(slug, **fields)
        return "done"
