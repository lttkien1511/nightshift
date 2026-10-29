"""HTML for the web UI. Everything that came from a user or a model is escaped in this file."""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote, urlparse

from .diffing import word_diff
from .runner import RunState
from .storage import Draft, digest, model_file

SHOW_MODES = ("changed", "rejected", "all")
LABELS = {
    "edited": "Edited",
    "rejected": "Rejected",
    "unchanged": "Unchanged",
    "skipped": "Skipped",
    "error": "Error",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _count(value: object) -> int:
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0


def split_diff(original: str, edited: str) -> tuple[str, str]:
    """The same word diff twice: deletions on the left, insertions on the right."""
    left: list[str] = []
    right: list[str] = []
    for op, chunk in word_diff(original, edited):
        text = esc(chunk)
        if op == "delete":
            left.append(f"<del>{text}</del>")
        elif op == "insert":
            right.append(f"<ins>{text}</ins>")
        else:
            left.append(text)
            right.append(text)
    return "".join(left), "".join(right)


def _chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{text}</span>'


def strip_html(snap: dict[str, Any], monitor_url: str = "", run: RunState | None = None) -> str:
    queue = snap.get("queue") or {}
    worker = snap.get("worker") or {}
    chips = [
        _chip(f"{_count(queue.get('waiting'))} waiting"),
        _chip(f"{_count(queue.get('ready'))} with results"),
    ]
    if run is not None and run.running:
        chips.append(_chip(f"editing {esc(run.slug)} {run.done}/{run.total}", "run"))
    finished = worker.get("finished") or worker.get("started")
    if worker.get("state") == "running":
        chips.append(_chip("worker running", "run"))
    elif finished:
        when = esc(finished)
        chips.append(_chip(f'worker ran <time data-rel datetime="{when}">{when}</time>'))
    else:
        chips.append(_chip("worker has not run yet"))
    if _count(worker.get("failed")):
        chips.append(_chip(f"{_count(worker.get('failed'))} failed", "bad"))
    if urlparse(monitor_url).scheme in {"http", "https"}:
        chips.append(
            f'<a class="chip link" href="{esc(monitor_url)}" rel="noopener noreferrer">Monitor</a>'
        )
    return "".join(chips)


def _sidebar(drafts: Sequence[Draft], active: str, full: bool, running: str = "") -> str:
    items = []
    for d in drafts:
        state = "editing" if d.slug == running else ("ready" if d.has_results else "waiting")
        current = ' aria-current="page"' if d.slug == active else ""
        items.append(
            f'<li><a href="/drafts/{esc(d.slug)}"{current}><span class="t">{esc(d.slug)}</span>'
            f'<span class="m">{d.size:,} bytes</span>'
            f'<span class="badge {state}">{state}</span></a></li>'
        )
    listing = "".join(items) or '<li class="empty">No drafts yet.</li>'
    note = '<p class="note">The queue is full. Delete a draft to add another.</p>' if full else ""
    return f'<a class="new-draft" href="/">New draft</a>{note}<ul class="queue">{listing}</ul>'


def page(
    title: str,
    main: str,
    drafts: Sequence[Draft],
    strip: str,
    active: str = "",
    queue_full: bool = False,
    running: str = "",
) -> str:
    # While the page you are looking at is being edited, reload it every few seconds.
    refresh = '<meta http-equiv="refresh" content="3">' if running and running == active else ""
    theme = (
        '<div class="theme" role="group" aria-label="Colour theme">'
        '<button type="button" data-mode="auto">Auto</button>'
        '<button type="button" data-mode="light">Light</button>'
        '<button type="button" data-mode="dark">Dark</button></div>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f"{refresh}<title>{esc(title)}</title>"
        '<link rel="stylesheet" href="/static/ui.css">'
        '<script src="/static/ui.js"></script></head><body>'
        '<header class="top"><a class="brand" href="/">nightshift</a>'
        f'<div class="strip">{strip}</div>'
        f'<div class="tools"><span id="phase" class="phase"></span>{theme}</div></header>'
        '<div class="sky" aria-hidden="true"></div>'
        '<div class="layout"><aside class="side">'
        f"{_sidebar(drafts, active, queue_full, running)}</aside>"
        f"<main>{main}</main></div></body></html>"
    )


def new_draft_main(max_bytes: int, can_run: bool = False) -> str:
    now_button = (
        '<button class="secondary" name="run" value="1">Queue and run now</button>'
        if can_run
        else ""
    )
    return (
        "<h1>New draft</h1>"
        '<p class="lead">Paste Markdown or choose a file. The nightly worker edits the prose and '
        "leaves code, links and front matter alone. Tomorrow you read the diff and keep what "
        "you like.</p>"
        '<form class="card" method="post" action="/drafts">'
        '<label>Name<input name="name" id="name" required maxlength="80" '
        'autocomplete="off"></label>'
        '<label>Markdown file <span class="hint">(optional, .md or .txt)</span>'
        '<input type="file" id="file" accept=".md,.markdown,.txt,text/markdown,text/plain"></label>'
        '<label>Markdown<textarea name="text" id="text" rows="18" required></textarea></label>'
        f'<p class="counter" data-max="{_count(max_bytes)}"><span id="count">0</span> of '
        f"{_count(max_bytes):,} bytes</p>"
        '<div class="buttons"><button class="primary">Queue draft</button>'
        f"{now_button}</div></form>"
    )


def _elapsed(seconds: float) -> str:
    minutes, rest = divmod(max(int(seconds), 0), 60)
    return f"{minutes}:{rest:02d}"


def run_panel(slug: str, models: Sequence[str], state: RunState | None, now: float) -> str:
    """The Run now card: a button, or the progress of the run that is going on."""
    if not models:
        return ""
    base = f"/drafts/{esc(slug)}"
    mine = state is not None and state.slug == slug
    if state is not None and state.running:
        if not mine:
            return (
                '<section class="card runcard"><b>Another draft is being edited.</b> '
                f'<span class="hint"><a href="/drafts/{esc(state.slug)}">{esc(state.slug)}</a> '
                "has to finish first, because the model handles one draft at a time."
                "</span></section>"
            )
        total = max(state.total, 1)
        return (
            '<section class="card runcard" aria-live="polite"><div class="runhead">'
            f"<b>Editing with {esc(state.model)}</b>"
            f'<span class="hint">block {state.done} of {state.total} '
            f"&middot; {_elapsed(now - state.started)} elapsed</span></div>"
            f'<progress max="{total}" value="{min(state.done, total)}"></progress>'
            f'<form method="post" action="{base}/cancel"><button>Cancel</button></form>'
            '<p class="hint">This page reloads by itself and shows the result when it is done.</p>'
            "</section>"
        )
    notice = ""
    if mine and state is not None and state.state == "failed":
        notice = f'<p class="banner">The last run failed: {esc(state.error)}</p>'
    elif mine and state is not None and state.state == "cancelled":
        notice = '<p class="hint">The last run was cancelled.</p>'
    select = ""
    if len(models) > 1:
        options = "".join(f'<option value="{esc(m)}">{esc(m)}</option>' for m in models)
        select = f'<label class="inline">Model<select name="model">{options}</select></label>'
    return (
        f'<section class="card runcard">{notice}'
        f'<form class="runform" method="post" action="{base}/run">'
        f'{select}<button class="primary">Run now</button>'
        '<span class="hint">Edit this draft right away instead of waiting for tonight. '
        "A short draft takes a few minutes.</span></form></section>"
    )


def _block(block: dict[str, Any], stale: bool) -> str:
    status = str(block.get("status", ""))
    original = str(block.get("original", ""))
    shown = block.get("proposal") if status == "rejected" else block.get("result")
    proposed = str(shown or original)
    label = LABELS.get(status, status)
    index = _count(block.get("index"))
    head = (
        f'<span class="badge {esc(status)}">{esc(label)}</span>'
        f'<span class="meta">block {index} &middot; {esc(block.get("kind", ""))}</span>'
    )
    if status == "edited":
        disabled = " disabled" if stale else ""
        head += (
            f'<label class="keep"><input type="checkbox" name="keep" value="{index}" '
            f"checked{disabled}> keep this edit</label>"
        )
    if status in {"edited", "rejected"}:
        left, right = split_diff(original, proposed)
        body = (
            f'<div class="pair"><pre class="old">{left}</pre><pre class="new">{right}</pre></div>'
        )
    else:
        body = f'<pre class="plain">{esc(original)}</pre>'
    reasons = block.get("reasons")
    why = ""
    if status == "rejected" and isinstance(reasons, list) and reasons:
        why = f'<p class="why">Original kept: {esc("; ".join(str(r) for r in reasons))}</p>'
    return (
        f'<article class="block {esc(status)}" id="b{index}">'
        f"<header>{head}</header>{body}{why}</article>"
    )


def _visible(block: dict[str, Any], show: str) -> bool:
    status = block.get("status")
    if show == "all":
        return True
    if show == "rejected":
        return status == "rejected"
    return status in {"edited", "rejected"}


def _top_reasons(blocks: Sequence[dict[str, Any]]) -> str:
    found: Counter[str] = Counter()
    for block in blocks:
        reasons = block.get("reasons")
        if block.get("status") == "rejected" and isinstance(reasons, list):
            found.update(str(r).split(":")[0] for r in reasons)
    if not found:
        return ""
    top = ", ".join(f"{esc(name)} ({n})" for name, n in found.most_common(3))
    return f'<p class="hint">Most common reasons for rejecting an edit: {top}.</p>'


def draft_main(
    slug: str,
    text: str,
    results: Sequence[dict[str, Any]],
    model: str = "",
    show: str = "changed",
    run_html: str = "",
) -> str:
    show = show if show in SHOW_MODES else "changed"
    base = f"/drafts/{esc(slug)}"
    delete = (
        f'<form class="inline" method="post" action="{base}/delete" '
        'data-confirm="Delete this draft and all of its results?">'
        '<button class="danger">Delete</button></form>'
    )
    title = f'<div class="titlebar"><h1>{esc(slug)}</h1>{delete}</div>'
    original = (
        '<details class="card"><summary>Original draft</summary>'
        f'<pre class="plain">{esc(text)}</pre></details>'
    )
    if not results:
        return (
            f'{title}{run_html}<div class="card empty">No results yet. The worker runs every '
            f"night; this page updates once it has been through the draft.</div>{original}"
        )

    by_file = {model_file(str(r.get("model", ""))): r for r in results}
    chosen_key = model if model in by_file else next(iter(by_file))
    result = by_file[chosen_key]
    stale = result.get("draft_sha256") != digest(text)
    blocks = [b for b in result.get("blocks", []) if isinstance(b, dict)]
    stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}

    def tab(key: str, r: dict[str, Any]) -> str:
        current = ' aria-current="true"' if key == chosen_key else ""
        href = f"{base}?model={quote(key)}&amp;show={show}"
        return f'<a class="tab" href="{href}"{current}>{esc(r.get("model", key))}</a>'

    links = "".join(tab(k, r) for k, r in by_file.items())
    tabs = f'<nav class="tabs" aria-label="Models">{links}</nav>'
    summary = "".join(
        f'<span class="stat"><b>{_count(n)}</b> {esc(LABELS.get(str(s), s).lower())}</span>'
        for s, n in sorted(stats.items())
    )
    created = esc(result.get("created", ""))
    summary_html = (
        f'<section class="card"><div class="stats">{summary}</div>{_top_reasons(blocks)}'
        f'<p class="hint">Run <time data-rel datetime="{created}">{created}</time></p></section>'
    )
    stale_html = (
        '<p class="banner">You changed this draft after the run. These edits no longer match it, '
        "so merging is switched off until the next run.</p>"
        if stale
        else ""
    )

    def filter_link(mode: str, name: str) -> str:
        current = ' aria-current="true"' if mode == show else ""
        href = f"{base}?model={quote(chosen_key)}&amp;show={mode}"
        return f'<a class="seg" href="{href}"{current}>{name}</a>'

    filters = (
        '<nav class="filters" aria-label="Which blocks to show">'
        + filter_link("changed", "Changed")
        + filter_link("rejected", "Rejected")
        + filter_link("all", "All blocks")
        + "</nav>"
    )
    shown = [b for b in blocks if _visible(b, show)]
    blocks_html = "".join(_block(b, stale) for b in shown) or (
        '<div class="card empty">Nothing to show with this filter.</div>'
    )
    editable = sum(1 for b in blocks if b.get("status") == "edited")
    actions = ""
    if editable and not stale:
        actions = (
            '<div class="actions"><span id="kept"></span>'
            '<button type="button" data-keep="all">Select all</button>'
            '<button type="button" data-keep="none">Select none</button>'
            '<button class="primary">Download merged .md</button></div>'
        )
    form = (
        f'<form method="get" action="{base}/merged" id="merge">'
        f'<input type="hidden" name="model" value="{esc(chosen_key)}">{actions}{blocks_html}</form>'
    )
    return f"{title}{run_html}{tabs}{summary_html}{stale_html}{filters}{form}{original}"
