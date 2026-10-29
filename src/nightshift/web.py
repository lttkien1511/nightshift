"""A small web UI and API for queueing drafts. Authentication is left to the ingress."""

from __future__ import annotations

import html
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from . import __version__
from .storage import DraftTooLarge, QueueFull, StorageError, Store

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    )
}
_STYLE = (
    ":root{color-scheme:light dark}"
    "body{font:16px/1.5 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem}"
    "input,textarea{width:100%;box-sizing:border-box;font:inherit;padding:.4rem}"
    "textarea{font-family:ui-monospace,monospace}"
    "table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:.3rem .2rem}"
    "pre{white-space:pre-wrap;word-break:break-word;padding:.8rem;border:1px solid #8884}"
    "form.inline{display:inline}.muted{opacity:.65}"
)


def store_from_env() -> Store:
    return Store(
        root=Path(os.environ.get("NIGHTSHIFT_DATA_DIR", "/data")),
        max_bytes=int(os.environ.get("NIGHTSHIFT_MAX_BYTES", "1000000")),
        max_queue=int(os.environ.get("NIGHTSHIFT_MAX_QUEUE", "20")),
    )


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if origin is not None:
        return urlparse(origin).netloc == request.headers.get("host", "")
    return request.headers.get("sec-fetch-site", "same-origin") in {"same-origin", "none"}


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style>{body}</html>"
    )


def _delete_form(slug: str) -> str:
    action = f"/drafts/{html.escape(slug)}/delete"
    return f'<form class="inline" method="post" action="{action}"><button>Delete</button></form>'


def _index_page(store: Store) -> str:
    rows = "".join(
        "<tr>"
        f'<td><a href="/drafts/{html.escape(d.slug)}">{html.escape(d.slug)}</a></td>'
        f"<td>{d.size} bytes</td>"
        f"<td>{'results ready' if d.has_results else 'waiting'}</td>"
        f"<td>{_delete_form(d.slug)}</td></tr>"
        for d in store.drafts()
    )
    table = (
        f"<table><tr><th>Draft</th><th>Size</th><th>Status</th><th></th></tr>{rows}</table>"
        if rows
        else '<p class="muted">No drafts yet.</p>'
    )
    form = (
        '<form method="post" action="/drafts">'
        '<p><label>Name<input name="name" required maxlength="80"></label></p>'
        '<p><label>Markdown<textarea name="text" rows="12" required></textarea></label></p>'
        "<button>Queue draft</button></form>"
    )
    return _page(
        "nightshift",
        "<h1>nightshift</h1>"
        '<p class="muted">Drafts wait here until the nightly worker has processed them.</p>'
        f"<h2>New draft</h2>{form}<h2>Queue</h2>{table}",
    )


def _draft_page(slug: str, text: str, has_results: bool) -> str:
    status = "Results are ready." if has_results else "No results yet."
    return _page(
        slug,
        f"<p><a href='/'>&larr; All drafts</a></p><h1>{html.escape(slug)}</h1>"
        f'<p class="muted">{status}</p><pre>{html.escape(text)}</pre>{_delete_form(slug)}',
    )


def create_app(store: Store | None = None) -> FastAPI:
    store = store or store_from_env()
    store.ensure()
    body_limit = store.max_bytes + 16_384
    app = FastAPI(title="nightshift", version=__version__)

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

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_index_page(store), headers=_PAGE_HEADERS)

    @app.get("/drafts")
    def list_drafts() -> list[dict[str, object]]:
        return [
            {"slug": d.slug, "size": d.size, "has_results": d.has_results} for d in store.drafts()
        ]

    @app.post("/drafts")
    def create_draft(
        name: Annotated[str, Form()], text: Annotated[str, Form()]
    ) -> RedirectResponse:
        try:
            store.save(name, text)
        except DraftTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except QueueFull as exc:
            raise HTTPException(429, str(exc)) from exc
        except StorageError as exc:
            raise HTTPException(400, str(exc)) from exc
        return RedirectResponse("/", status_code=303)

    @app.get("/drafts/{slug}", response_class=HTMLResponse)
    def show_draft(slug: str) -> HTMLResponse:
        text = store.read(slug)
        if text is None:
            raise HTTPException(404, "no such draft")
        has_results = any(d.slug == slug and d.has_results for d in store.drafts())
        return HTMLResponse(_draft_page(slug, text, has_results), headers=_PAGE_HEADERS)

    @app.post("/drafts/{slug}/delete")
    def delete_draft(slug: str) -> RedirectResponse:
        if not store.delete(slug):
            raise HTTPException(404, "no such draft")
        return RedirectResponse("/", status_code=303)

    return app
