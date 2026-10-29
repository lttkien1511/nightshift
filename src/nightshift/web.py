"""A small web UI and API for queueing drafts. Authentication is left to the ingress."""

from __future__ import annotations

import hashlib
import html
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from . import __version__
from .diffing import word_diff
from .status import collect, render, snapshot
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
    "ins{background:#2a92;text-decoration:none}del{background:#d334}"
    "ol.flow{list-style:none;padding:0;display:grid;gap:.6rem;"
    "grid-template-columns:repeat(auto-fit,minmax(9.5rem,1fr))}"
    "li.node{border:1px solid #8886;border-left-width:6px;border-radius:6px;padding:.5rem .6rem;"
    "font-size:.85rem;line-height:1.35}"
    "li.node b{display:block;font-size:.95rem}"
    ".tag{display:inline-block;font-size:.72rem;margin:.1rem 0 .2rem;padding:0 .35rem;"
    "border:1px solid #8886;border-radius:4px}"
    "li.ok{border-left-color:#2a9d4a}li.warn{border-left-color:#d19a1c}"
    "li.bad{border-left-color:#d33}li.idle{border-left-color:#8886}"
    "nav{margin:0 0 1rem}nav a{margin-right:1rem}"
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


def _page(title: str, body: str, head: str = "") -> str:
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"{head}<title>{html.escape(title)}</title><style>{_STYLE}</style>{body}</html>"
    )


def _status_page(store: Store) -> str:
    deploy, nightly = collect(
        store,
        os.environ.get("NIGHTSHIFT_OLLAMA_URL", "http://ollama.nightshift.svc.cluster.local:11434"),
        actions_url=os.environ.get(
            "NIGHTSHIFT_ACTIONS_URL", "https://github.com/lttkien1511/nightshift/actions"
        ),
        argocd_url=os.environ.get("NIGHTSHIFT_ARGOCD_URL", ""),
        version=__version__,
    )
    return _page(
        "nightshift status",
        "<nav><a href='/'>&larr; Drafts</a></nav><h1>Pipeline status</h1>"
        + render(deploy, nightly),
        head='<meta http-equiv="refresh" content="30">',
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
        "<nav><a href='/status'>Pipeline status</a></nav>"
        '<p class="muted">Drafts wait here until the nightly worker has processed them.</p>'
        f"<h2>New draft</h2>{form}<h2>Queue</h2>{table}",
    )


def _diff_html(original: str, edited: str) -> str:
    parts = []
    for op, chunk in word_diff(original, edited):
        escaped = html.escape(chunk)
        if op == "delete":
            parts.append(f"<del>{escaped}</del>")
        elif op == "insert":
            parts.append(f"<ins>{escaped}</ins>")
        else:
            parts.append(escaped)
    return "".join(parts)


def _block_html(block: dict[str, object]) -> str:
    status = str(block.get("status", ""))
    if status not in {"edited", "rejected"}:
        return ""
    original = str(block.get("original", ""))
    shown = str(block.get("result" if status == "edited" else "proposal") or original)
    note = ""
    if status == "rejected":
        reasons = block.get("reasons")
        listed = "; ".join(str(r) for r in reasons) if isinstance(reasons, list) else ""
        note = f'<p class="muted">Rejected, original kept: {html.escape(listed)}</p>'
    return f"<pre>{_diff_html(original, shown)}</pre>{note}"


def _result_html(result: dict[str, object], current: str) -> str:
    stale = result.get("draft_sha256") != hashlib.sha256(current.encode("utf-8")).hexdigest()
    stats = result.get("stats")
    summary = (
        ", ".join(f"{count} {status}" for status, count in sorted(stats.items()))
        if isinstance(stats, dict)
        else ""
    )
    blocks = result.get("blocks")
    body = "".join(_block_html(b) for b in blocks if isinstance(b, dict)) if blocks else ""
    model = html.escape(str(result.get("model", "model")))
    created = html.escape(str(result.get("created", "")))
    stale_note = (
        '<p class="muted">The draft changed after this run; wait for the next one.</p>'
        if stale
        else ""
    )
    body = body or '<p class="muted">No changes proposed.</p>'
    return (
        f"<h2>{model}</h2>"
        f'<p class="muted">{html.escape(summary)} &middot; {created}</p>'
        f"{stale_note}{body}"
    )


def _draft_page(slug: str, text: str, results: list[dict[str, object]]) -> str:
    status = f"{len(results)} model run(s) ready." if results else "No results yet."
    shown = "".join(_result_html(r, text) for r in results)
    return _page(
        slug,
        f"<p><a href='/'>&larr; All drafts</a></p><h1>{html.escape(slug)}</h1>"
        f'<p class="muted">{status}</p>{shown}'
        f"<h2>Original</h2><pre>{html.escape(text)}</pre>{_delete_form(slug)}",
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

    @app.get("/status.json")
    def status_json() -> dict[str, object]:
        return snapshot(store, __version__)

    @app.get("/status", response_class=HTMLResponse)
    def status() -> HTMLResponse:
        return HTMLResponse(_status_page(store), headers=_PAGE_HEADERS)

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
        return HTMLResponse(_draft_page(slug, text, store.results(slug)), headers=_PAGE_HEADERS)

    @app.get("/drafts/{slug}/results")
    def draft_results(slug: str) -> list[dict[str, object]]:
        if store.read(slug) is None:
            raise HTTPException(404, "no such draft")
        return store.results(slug)

    @app.post("/drafts/{slug}/delete")
    def delete_draft(slug: str) -> RedirectResponse:
        if not store.delete(slug):
            raise HTTPException(404, "no such draft")
        return RedirectResponse("/", status_code=303)

    return app
