"""A live picture of the whole pipeline, built only from what the web pod can see."""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

from .ollama import OllamaInfo, list_models
from .storage import Store, digest

# Ho Chi Minh City has no daylight saving, so a fixed offset is exact.
LOCAL = timezone(timedelta(hours=7))
RUN_HOUR = 1
STALLED_AFTER = timedelta(hours=7)

STATES = {
    "ok": "OK",
    "warn": "Check",
    "bad": "Down",
    "idle": "Not measured",
}


@dataclass(frozen=True)
class Node:
    title: str
    state: str
    lines: tuple[str, ...]
    href: str = ""


def next_run(now: datetime) -> datetime:
    """The next 01:00 local time after ``now``."""
    local = now.astimezone(LOCAL)
    candidate = local.replace(hour=RUN_HOUR, minute=0, second=0, microsecond=0)
    return candidate if candidate > local else candidate + timedelta(days=1)


def _parse(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _ago(then: datetime, now: datetime) -> str:
    seconds = int((now - then).total_seconds())
    if seconds < 90:
        return "just now"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes} min ago"
    hours = minutes // 60
    return f"{hours} h ago" if hours < 48 else f"{hours // 24} days ago"


def queue_counts(store: Store) -> tuple[int, int, int]:
    """How many drafts are waiting, have results, and have results older than the draft."""
    waiting = ready = stale = 0
    for draft in store.drafts():
        text = store.read(draft.slug) or ""
        results = store.results(draft.slug)
        if not results:
            waiting += 1
            continue
        ready += 1
        if any(r.get("draft_sha256") != digest(text) for r in results):
            stale += 1
    return waiting, ready, stale


def snapshot(store: Store, version: str) -> dict[str, object]:
    """Machine-readable state for the separate monitor app. Contains no draft text."""
    waiting, ready, stale = queue_counts(store)
    return {
        "version": version,
        "queue": {"waiting": waiting, "ready": ready, "stale": stale},
        "worker": store.worker_status() or {},
    }


def _queue(store: Store) -> tuple[Node, Node]:
    waiting, ready, stale = queue_counts(store)
    queue = Node(
        "Queue",
        "ok",
        (f"{waiting} waiting for the next run", f"{waiting + ready} drafts in the inbox"),
    )
    lines = (f"{ready} with results", f"{stale} out of date" if stale else "none out of date")
    return queue, Node("Results", "warn" if stale else "ok", lines)


def _ollama(info: OllamaInfo, expected: list[str]) -> Node:
    if not info.reachable:
        return Node("Ollama", "bad", ("not reachable from the web pod", info.error[:80]))
    names = [name for name, _ in info.models]
    missing = [m for m in expected if m not in names]
    if not names:
        return Node("Ollama", "warn", ("running, but no model is installed",))
    lines = [f"{len(names)} model(s): " + ", ".join(names)]
    if missing:
        lines.append("missing: " + ", ".join(missing))
    return Node("Ollama", "warn" if missing else "ok", tuple(lines))


def _worker(status: dict[str, object] | None, now: datetime) -> Node:
    if status is None:
        return Node("Worker", "idle", ("has never run", f"next run {_when(next_run(now))}"))
    started = _parse(status.get("started"))
    if status.get("state") == "running":
        if started and now - started > STALLED_AFTER:
            return Node("Worker", "bad", ("a run started but never reported back",))
        return Node("Worker", "warn", ("running now",))
    finished = _parse(status.get("finished")) or started
    when = _ago(finished, now) if finished else "at an unknown time"
    done, failed = int(status.get("done") or 0), int(status.get("failed") or 0)
    summary = f"{done} done, {int(status.get('skipped') or 0)} skipped, {failed} failed"
    state = "ok"
    if failed:
        state = "bad" if not done else "warn"
    elif status.get("timed_out"):
        state = "warn"
    lines = [f"last run {when}", summary]
    if status.get("timed_out"):
        lines.append("stopped early: time budget used up")
    lines.append(f"next run {_when(next_run(now))}")
    return Node("Worker", state, tuple(lines))


def _when(moment: datetime) -> str:
    return moment.astimezone(LOCAL).strftime("%d/%m %H:%M") + " (UTC+7)"


def collect(
    store: Store,
    ollama_url: str,
    now: datetime | None = None,
    actions_url: str = "",
    argocd_url: str = "",
    version: str = "",
) -> tuple[list[Node], list[Node]]:
    """Return the deploy flow and the nightly flow, each as an ordered list of nodes."""
    now = now or datetime.now(UTC)
    status = store.worker_status()
    expected = [str(m) for m in status.get("models", [])] if status else []
    queue, results = _queue(store)
    deploy = [
        Node("You push to main", "idle", ("commit and push are done by you",)),
        Node("GitHub Actions", "idle", ("test, build image, bot bumps the tag",), actions_url),
        Node("Argo CD", "idle", ("syncs the new tag into the cluster",), argocd_url),
        Node("This web pod", "ok", ("answering this request", f"version {version}")),
    ]
    nightly = [
        queue,
        _ollama(list_models(ollama_url), expected),
        _worker(status, now),
        results,
        Node("You, in the morning", "idle", ("read the diff, keep what you like",)),
    ]
    return deploy, nightly


def render(deploy: list[Node], nightly: list[Node]) -> str:
    def card(node: Node) -> str:
        title = html.escape(node.title)
        if node.href:
            title = f'<a href="{html.escape(node.href, quote=True)}">{title}</a>'
        body = "".join(f"<br>{html.escape(line)}" for line in node.lines)
        return (
            f'<li class="node {node.state}"><b>{title}</b>'
            f'<span class="tag">{STATES[node.state]}</span>{body}</li>'
        )

    def flow(name: str, nodes: list[Node]) -> str:
        return f'<h2>{name}</h2><ol class="flow">{"".join(card(n) for n in nodes)}</ol>'

    return (
        flow("Deploy: every push to main", deploy)
        + flow("Nightly: queue, edit, review", nightly)
        + '<p class="muted">Refreshes every 30 seconds. "Not measured" means the web pod '
        "cannot see that step from inside the cluster.</p>"
    )
