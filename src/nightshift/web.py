"""A small web UI and API for queueing drafts. Authentication is left to the ingress."""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse

from . import __version__, ui
from .merge import MergeError, merge
from .ollama import OllamaConfig, OllamaEditor
from .runner import Busy, Runner, RunnerError
from .status import snapshot
from .storage import DraftTooLarge, QueueFull, StorageError, Store, digest, model_file

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_STATIC = Path(__file__).parent / "static"
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
    )
}


def store_from_env() -> Store:
    return Store(
        root=Path(os.environ.get("NIGHTSHIFT_DATA_DIR", "/data")),
        max_bytes=int(os.environ.get("NIGHTSHIFT_MAX_BYTES", "1000000")),
        max_queue=int(os.environ.get("NIGHTSHIFT_MAX_QUEUE", "20")),
    )


def runner_from_env(store: Store) -> Runner:
    """Run now needs the model list and the Ollama address; with no models it stays hidden."""
    raw = os.environ.get("NIGHTSHIFT_RUN_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    base = OllamaConfig(
        url=os.environ.get("NIGHTSHIFT_OLLAMA_URL", OllamaConfig.url),
        timeout=float(os.environ.get("NIGHTSHIFT_RUN_TIMEOUT", "300")),
        num_ctx=int(os.environ.get("NIGHTSHIFT_OLLAMA_NUM_CTX", OllamaConfig.num_ctx)),
    )

    def make_editor(model: str) -> OllamaEditor:
        voice = store.voice_path.read_text(encoding="utf-8") if store.voice_path.is_file() else ""
        cfg = OllamaConfig(url=base.url, model=model, timeout=base.timeout, num_ctx=base.num_ctx)
        return OllamaEditor(cfg, voice)

    return Runner(store, make_editor, models)


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if origin is not None:
        return urlparse(origin).netloc == request.headers.get("host", "")
    return request.headers.get("sec-fetch-site", "same-origin") in {"same-origin", "none"}


def create_app(store: Store | None = None, runner: Runner | None = None) -> FastAPI:
    store = store or store_from_env()
    store.ensure()
    runner = runner or runner_from_env(store)
    body_limit = store.max_bytes + 16_384
    monitor_url = os.environ.get("NIGHTSHIFT_MONITOR_URL", "")
    app = FastAPI(title="nightshift", version=__version__)

    def render(title: str, main: str, active: str = "") -> HTMLResponse:
        drafts = store.drafts()
        run = runner.state
        strip = ui.strip_html(snapshot(store, __version__), monitor_url, run)
        full = len(drafts) >= store.max_queue
        running = run.slug if run is not None and run.running else ""
        html = ui.page(title, main, drafts, strip, active=active, queue_full=full, running=running)
        return HTMLResponse(html, headers=_PAGE_HEADERS)

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method not in _SAFE_METHODS:
            if not _same_origin(request):
                return Response("cross-site request refused", 403, media_type="text/plain")
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > body_limit:
                return Response("request body too large", 413, media_type="text/plain")
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status.json")
    def status_json() -> dict[str, object]:
        return snapshot(store, __version__)

    @app.get("/static/ui.css")
    def stylesheet() -> FileResponse:
        return FileResponse(_STATIC / "ui.css", media_type="text/css")

    @app.get("/static/ui.js")
    def script() -> FileResponse:
        return FileResponse(_STATIC / "ui.js", media_type="text/javascript")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return render("nightshift", ui.new_draft_main(store.max_bytes, bool(runner.models)))

    @app.get("/drafts")
    def list_drafts() -> list[dict[str, object]]:
        return [
            {"slug": d.slug, "size": d.size, "has_results": d.has_results} for d in store.drafts()
        ]

    @app.post("/drafts")
    def create_draft(
        name: Annotated[str, Form()],
        text: Annotated[str, Form()],
        run: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        try:
            slug = store.save(name, text)
        except DraftTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except QueueFull as exc:
            raise HTTPException(429, str(exc)) from exc
        except StorageError as exc:
            raise HTTPException(400, str(exc)) from exc
        if run:
            # The draft is queued either way; the page explains why nothing is running.
            with contextlib.suppress(RunnerError):
                runner.start(slug)
        return RedirectResponse(f"/drafts/{slug}", status_code=303)

    @app.get("/drafts/{slug}", response_class=HTMLResponse)
    def show_draft(slug: str, model: str = "", show: str = "changed") -> HTMLResponse:
        text = store.read(slug)
        if text is None:
            raise HTTPException(404, "no such draft")
        panel = ui.run_panel(slug, runner.models, runner.state, time.time())
        main = ui.draft_main(
            slug, text, store.results(slug), model=model, show=show, run_html=panel
        )
        return render(slug, main, active=slug)

    @app.post("/drafts/{slug}/run")
    def run_now(slug: str, model: Annotated[str, Form()] = "") -> RedirectResponse:
        try:
            runner.start(slug, model)
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        except RunnerError as exc:
            status = 404 if "no such draft" in str(exc) else 400
            raise HTTPException(status, str(exc)) from exc
        return RedirectResponse(f"/drafts/{slug}", status_code=303)

    @app.post("/drafts/{slug}/cancel")
    def cancel_run(slug: str) -> RedirectResponse:
        state = runner.state
        if state is not None and state.running and state.slug == slug:
            runner.cancel()
        return RedirectResponse(f"/drafts/{slug}", status_code=303)

    @app.get("/drafts/{slug}/run.json")
    def run_state(slug: str) -> dict[str, object]:
        state = runner.state
        if state is None or state.slug != slug:
            return {"state": "idle"}
        return {
            "state": state.state,
            "model": state.model,
            "done": state.done,
            "total": state.total,
            "error": state.error,
        }

    @app.get("/drafts/{slug}/results")
    def draft_results(slug: str) -> list[dict[str, object]]:
        if store.read(slug) is None:
            raise HTTPException(404, "no such draft")
        return store.results(slug)

    @app.get("/drafts/{slug}/merged")
    def merged(request: Request, slug: str, model: str) -> PlainTextResponse:
        text = store.read(slug)
        if text is None:
            raise HTTPException(404, "no such draft")
        result = next(
            (r for r in store.results(slug) if model_file(str(r.get("model"))) == model), None
        )
        if result is None:
            raise HTTPException(404, "no results for that model")
        if result.get("draft_sha256") != digest(text):
            raise HTTPException(409, "the draft changed after this run")
        try:
            keep = {int(value) for value in request.query_params.getlist("keep")}
        except ValueError as exc:
            raise HTTPException(400, "keep must be block numbers") from exc
        blocks = [b for b in result.get("blocks", []) if isinstance(b, dict)]
        try:
            body = merge(text, blocks, keep)
        except MergeError as exc:
            raise HTTPException(409, str(exc)) from exc
        headers = {"Content-Disposition": f'attachment; filename="{slug}-edited.md"'}
        return PlainTextResponse(body, media_type="text/markdown; charset=utf-8", headers=headers)

    @app.post("/drafts/{slug}/delete")
    def delete_draft(slug: str) -> RedirectResponse:
        if not store.delete(slug):
            raise HTTPException(404, "no such draft")
        return RedirectResponse("/", status_code=303)

    return app
