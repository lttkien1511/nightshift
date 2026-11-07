"""A small web UI and API for queueing drafts. Authentication is left to the ingress."""

from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse

from . import __version__, ui, ui_posts
from .compose import (
    GRAINS,
    LIMITS,
    POST_LIMIT,
    VOICE_FIELDS,
    BriefError,
    gather,
    normalize_brief,
    review_post,
)
from .composer import Composer, Stream
from .gate import Gate
from .merge import MergeError, merge
from .ollama import OllamaConfig, OllamaEditor, generate_stream, list_models
from .runner import Busy, Runner, RunnerError
from .status import snapshot
from .storage import (
    DraftTooLarge,
    NameTaken,
    QueueFull,
    StorageError,
    Store,
    digest,
    model_file,
    slugify,
)

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


COMPOSE_CTX = 8192  # a post needs room for the material, the outline and the answer


def composer_from_env(store: Store, gate: Gate) -> Composer:
    """Blog posts have their own models: NIGHTSHIFT_POST_MODELS, else whatever Ollama has
    downloaded. Without any models configured there are no buttons."""
    listed = _names(os.environ.get("NIGHTSHIFT_POST_MODELS", ""))
    fallback = _names(os.environ.get("NIGHTSHIFT_RUN_MODELS", ""))
    url = os.environ.get("NIGHTSHIFT_OLLAMA_URL", OllamaConfig.url)
    seen: dict[str, object] = {"at": -60.0, "found": []}

    def models() -> list[str]:
        if listed or not fallback:
            return listed
        if time.monotonic() - float(seen["at"]) > 30:  # type: ignore[arg-type]
            seen.update(at=time.monotonic(), found=list_models(url))
        found = seen["found"]
        return list(found) if isinstance(found, list) and found else fallback

    silence = float(os.environ.get("NIGHTSHIFT_RUN_TIMEOUT", "300"))
    context = int(os.environ.get("NIGHTSHIFT_COMPOSE_CTX", str(COMPOSE_CTX)))

    def make_stream(model: str) -> Stream:
        voice = store.voice_path.read_text(encoding="utf-8") if store.voice_path.is_file() else ""
        cfg = OllamaConfig(url=url, model=model, timeout=silence, temperature=0.5)

        def stream(
            system: str, prompt: str, num_predict: int, on_text: Callable[[str], object]
        ) -> str:
            styled = system
            if voice.strip():
                styled += f"\n\nSamples of the author's own voice, for style only:\n{voice.strip()}"
            return generate_stream(
                cfg,
                styled,
                prompt,
                on_text,
                num_predict=num_predict,
                num_ctx=context,
                think=False if model.startswith("qwen3") else None,  # no thinking aloud
            )

        return stream

    rest = float(os.environ.get("NIGHTSHIFT_MODEL_REST_SECONDS", "0") or 0)
    return Composer(store, make_stream, models, gate, rest_seconds=rest)


def _names(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


async def _form_with_lists(request: Request) -> tuple[dict[str, str], dict[str, list[str]]]:
    """The form fields, with the model choices (several checkboxes of one name) kept apart."""
    form = await request.form()
    lists = {
        name: [v for v in form.getlist(name) if isinstance(v, str)]
        for name in ("models_outline", "models_write")
    }
    values = {k: v for k, v in form.items() if isinstance(v, str) and k not in lists}
    return values, lists


async def _form(request: Request) -> dict[str, str]:
    form = await request.form()
    return {key: value for key, value in form.items() if isinstance(value, str)}


def _lf(text: str) -> str:
    """Browsers send form text with CRLF line breaks; drafts are stored with plain LF."""
    return text.replace("\r\n", "\n")


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


def create_app(
    store: Store | None = None,
    runner: Runner | None = None,
    composer: Composer | None = None,
) -> FastAPI:
    store = store or store_from_env()
    store.ensure()
    runner = runner or runner_from_env(store)
    composer = composer or composer_from_env(store, runner.gate)
    body_limit = store.max_bytes + 16_384
    monitor_url = os.environ.get("NIGHTSHIFT_MONITOR_URL", "")
    app = FastAPI(title="nightshift", version=__version__)

    def render(title: str, main: str, active: str = "", post_active: str = "") -> HTMLResponse:
        drafts = store.drafts()
        run, writing = runner.state, composer.state
        strip = ui.strip_html(snapshot(store, __version__), monitor_url, run, writing)
        running = run.slug if run is not None and run.running else ""
        post_running = writing.slug if writing is not None and writing.running else ""
        html = ui.page(
            title,
            main,
            drafts,
            strip,
            active=active,
            queue_full=len(drafts) >= store.max_queue,
            running=running,
            posts=store.posts(),
            post_active=post_active,
            post_running=post_running,
        )
        return HTMLResponse(html, headers=_PAGE_HEADERS)

    def post_or_404(slug: str) -> dict[str, object]:
        record = store.post(slug)
        if record is None:
            raise HTTPException(404, "no such blog post")
        return record

    def writing_now(slug: str) -> bool:
        state = composer.state
        return state is not None and state.running and state.slug == slug

    def start_job(slug: str, kind: str, picked: list[str]) -> None:
        try:
            composer.start(slug, kind, models=picked, replace=len(picked) <= 1)
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        except RunnerError as exc:
            raise HTTPException(400, str(exc)) from exc
        if picked and kind in {"outline", "write"}:
            remember_models(slug, kind, picked)

    def only_known(picked: list[str]) -> list[str]:
        known = composer.models
        return [m for m in dict.fromkeys(picked) if m in known]

    def remember_models(slug: str, kind: str, picked: list[str]) -> None:
        """Keep the models chosen for a step with the post, and offer them for the next post."""
        record = post_or_404(slug)
        kept = record.get("models")
        store.update_post(slug, models={**(kept if isinstance(kept, dict) else {}), kind: picked})
        store.save_blog_voice({f"pick_{kind}": ",".join(picked)})

    def saved_picks(record: dict[str, object] | None) -> dict[str, list[str]]:
        """The models to tick: those of this post, or else those of the last post."""
        kept = record.get("models") if record else None
        if isinstance(kept, dict):
            return {
                k: [m for m in v if isinstance(m, str)]
                for k, v in kept.items()
                if isinstance(v, list)
            }
        last = store.blog_voice()
        return {k: _names(last.get(f"pick_{k}", "")) for k in ("outline", "write")}

    def brief_page(
        values: dict[str, str] | None,
        action: str,
        error: str = "",
        existing: bool = False,
        post_active: str = "",
        status: int = 200,
        picks: dict[str, list[str]] | None = None,
    ) -> HTMLResponse:
        main = ui_posts.brief_form(
            values,
            action,
            bool(composer.models),
            error,
            existing,
            composer.models,
            picks if picks is not None else saved_picks(None),
        )
        response = render("Blog post brief", main, post_active=post_active)
        response.status_code = status
        return response

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
        # Not no-referrer: with that policy a browser sends "Origin: null" on every form POST,
        # and the same-origin check above would refuse the page's own forms.
        response.headers["Referrer-Policy"] = "same-origin"
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
            slug = store.save(name, _lf(text))
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

    @app.get("/drafts/{slug}/edit", response_class=HTMLResponse)
    def edit_form(slug: str) -> HTMLResponse:
        text = store.read(slug)
        if text is None:
            raise HTTPException(404, "no such draft")
        main = ui.edit_main(slug, text, store.max_bytes, bool(runner.models))
        return render(f"Edit {slug}", main, active=slug)

    @app.post("/drafts/{slug}/edit")
    def save_edit(
        slug: str,
        name: Annotated[str, Form()],
        text: Annotated[str, Form()],
        run: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        current = store.read(slug)
        if current is None:
            raise HTTPException(404, "no such draft")
        state = runner.state
        if state is not None and state.running and state.slug == slug:
            raise HTTPException(409, "the model is editing this draft right now; cancel that first")
        new_text = _lf(text)
        try:
            new_slug = slugify(name)
            store.validate(new_text)
            if new_slug != slug:
                store.rename(slug, new_slug)
            if new_text != _lf(current):
                store.save(new_slug, new_text)
        except NameTaken as exc:
            raise HTTPException(409, str(exc)) from exc
        except DraftTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except StorageError as exc:
            raise HTTPException(400, str(exc)) from exc
        if run:
            with contextlib.suppress(RunnerError):
                runner.start(new_slug)
        return RedirectResponse(f"/drafts/{new_slug}", status_code=303)

    @app.get("/posts/new", response_class=HTMLResponse)
    def new_post_form() -> HTMLResponse:
        return brief_page(store.blog_voice() or None, "/posts")

    @app.post("/posts")
    async def create_post(request: Request) -> Response:
        values, lists = await _form_with_lists(request)
        action = values.pop("action", "save")
        picks = {
            "outline": only_known(lists["models_outline"]),
            "write": only_known(lists["models_write"]),
        }
        try:
            brief = normalize_brief(values)
            slug = store.create_post(brief["topic"], brief)
            store.save_blog_voice({k: brief[k] for k in VOICE_FIELDS})
        except BriefError as exc:
            return brief_page(values, "/posts", str(exc), status=400, picks=picks)
        except QueueFull as exc:
            raise HTTPException(429, str(exc)) from exc
        if picks["outline"] or picks["write"]:
            store.update_post(slug, models={k: v for k, v in picks.items() if v})
            store.save_blog_voice({f"pick_{k}": ",".join(v) for k, v in picks.items() if v})
        if action in {"outline", "write"}:
            # The post is saved either way; its page says why nothing is running.
            chosen = picks[action]
            with contextlib.suppress(RunnerError):
                composer.start(slug, action, models=chosen, replace=len(chosen) <= 1)
        return RedirectResponse(f"/posts/{slug}", status_code=303)

    @app.get("/posts/{slug}", response_class=HTMLResponse)
    def show_post(slug: str, grain: str = "paragraph") -> HTMLResponse:
        record = post_or_404(slug)
        grain = grain if grain in GRAINS else "paragraph"
        main = ui_posts.post_main(
            record, composer.state, composer.models, time.time(), grain, saved_picks(record)
        )
        return render(str(record.get("title", slug)), main, post_active=slug)

    @app.get("/posts/{slug}/edit", response_class=HTMLResponse)
    def edit_post_form(slug: str) -> HTMLResponse:
        record = post_or_404(slug)
        brief = record.get("brief")
        values = {k: str(v) for k, v in brief.items()} if isinstance(brief, dict) else {}
        return brief_page(
            values,
            f"/posts/{slug}/edit",
            existing=True,
            post_active=slug,
            picks=saved_picks(record),
        )

    @app.post("/posts/{slug}/edit")
    async def save_post_brief(slug: str, request: Request) -> Response:
        post_or_404(slug)
        if writing_now(slug):
            raise HTTPException(409, "the model is writing this post right now; cancel that first")
        values, lists = await _form_with_lists(request)
        values.pop("action", None)
        picks = {
            "outline": only_known(lists["models_outline"]),
            "write": only_known(lists["models_write"]),
        }
        try:
            brief = normalize_brief(values)
        except BriefError as exc:
            return brief_page(
                values,
                f"/posts/{slug}/edit",
                str(exc),
                existing=True,
                post_active=slug,
                status=400,
                picks=picks,
            )
        store.update_post(slug, brief=brief, title=brief["topic"])
        if picks["outline"] or picks["write"]:
            store.update_post(slug, models={k: v for k, v in picks.items() if v})
        store.save_blog_voice({k: brief[k] for k in VOICE_FIELDS})
        return RedirectResponse(f"/posts/{slug}", status_code=303)

    @app.post("/posts/{slug}/outline")
    def suggest_outline(
        slug: str,
        model: Annotated[str, Form()] = "",
        models: Annotated[list[str] | None, Form()] = None,
    ) -> RedirectResponse:
        post_or_404(slug)
        start_job(slug, "outline", [*(models or []), *([model] if model else [])])
        return RedirectResponse(f"/posts/{slug}", status_code=303)

    @app.post("/posts/{slug}/voices")
    def try_voices(slug: str, model: Annotated[str, Form()] = "") -> RedirectResponse:
        post_or_404(slug)
        start_job(slug, "voices", [model] if model else [])
        return RedirectResponse(f"/posts/{slug}", status_code=303)

    @app.post("/posts/{slug}/voice")
    def choose_voice(slug: str, choice: Annotated[int, Form()] = -1) -> RedirectResponse:
        record = post_or_404(slug)
        if writing_now(slug):
            raise HTTPException(409, "the model is writing this post right now; cancel that first")
        voices = record.get("voices")
        if not isinstance(voices, list) or not 0 <= choice < len(voices):
            raise HTTPException(400, "pick one of the voices")
        picked = voices[choice]
        brief = record.get("brief")
        if not isinstance(picked, str) or not isinstance(brief, dict):
            raise HTTPException(400, "pick one of the voices")
        brief = {**brief, "voice_sample": picked[: LIMITS["voice_sample"]]}
        store.update_post(slug, brief=brief)
        store.save_blog_voice({k: str(brief.get(k, "")) for k in VOICE_FIELDS})
        return RedirectResponse(f"/posts/{slug}", status_code=303)

    @app.post("/posts/{slug}/outline/save")
    def save_outline(slug: str, outline: Annotated[str, Form()] = "") -> RedirectResponse:
        record = post_or_404(slug)
        if writing_now(slug):
            raise HTTPException(409, "the model is writing this post right now; cancel that first")
        text = _lf(outline).strip()
        if len(text) > LIMITS["outline"]:
            raise HTTPException(400, f"the outline is longer than {LIMITS['outline']:,} characters")
        fields: dict[str, object] = {"outline": text}
        if text and record.get("stage") == "brief":
            fields["stage"] = "outline"
        store.update_post(slug, **fields)
        return RedirectResponse(f"/posts/{slug}", status_code=303)

    @app.post("/posts/{slug}/write")
    def write_post(
        slug: str,
        model: Annotated[str, Form()] = "",
        models: Annotated[list[str] | None, Form()] = None,
        outline: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        post_or_404(slug)
        if outline is not None:
            outline = _lf(outline).strip()
            if len(outline) > LIMITS["outline"]:
                raise HTTPException(
                    400, f"the outline is longer than {LIMITS['outline']:,} characters"
                )
            store.update_post(slug, outline=outline)
        start_job(slug, "write", [*(models or []), *([model] if model else [])])
        return RedirectResponse(f"/posts/{slug}", status_code=303)

    @app.post("/posts/{slug}/gather")
    async def gather_parts(slug: str, request: Request) -> RedirectResponse:
        """Add the parts picked from the models' answers to the outline or post being built."""
        record = post_or_404(slug)
        if writing_now(slug):
            raise HTTPException(409, "the model is writing this post right now; cancel that first")
        form = await request.form()
        kind, grain = str(form.get("kind", "")), str(form.get("grain", "paragraph"))
        if kind not in {"outline", "write"} or grain not in GRAINS:
            raise HTTPException(400, "unknown step")
        picks = [v for v in form.getlist("pick") if isinstance(v, str)]
        field = "outline" if kind == "outline" else "post"
        try:
            text = gather(record.get("candidates"), str(record.get(field, "")), kind, grain, picks)
        except BriefError as exc:
            raise HTTPException(400, str(exc)) from exc
        limit = LIMITS["outline"] if kind == "outline" else POST_LIMIT
        if len(text) > limit:
            raise HTTPException(400, f"the {field} would be longer than {limit:,} characters")
        save_working(slug, record, kind, text)
        return RedirectResponse(f"/posts/{slug}?grain={grain}#build-{kind}", status_code=303)

    def save_working(slug: str, record: dict[str, object], kind: str, text: str) -> None:
        """Keep the outline or post a person built, and look it over again if it is the post."""
        if kind == "outline":
            fields: dict[str, object] = {"outline": text}
            if text and record.get("stage") == "brief":
                fields["stage"] = "outline"
        elif not text:
            fields = {
                "post": "",
                "checks": [],
                "flags": [],
                "stage": "outline" if record.get("outline") else "brief",
            }
        else:
            try:
                checks, flags = review_post(record, text)
            except BriefError:
                checks, flags = [], []
            fields = {
                "post": text,
                "checks": checks,
                "flags": flags,
                "stage": "done",
                "model": "picks",
            }
        store.update_post(slug, **fields)

    @app.post("/posts/{slug}/post/save")
    def save_post_text(slug: str, post: Annotated[str, Form()] = "") -> RedirectResponse:
        record = post_or_404(slug)
        if writing_now(slug):
            raise HTTPException(409, "the model is writing this post right now; cancel that first")
        text = _lf(post).strip()
        if len(text) > POST_LIMIT:
            raise HTTPException(400, f"the post is longer than {POST_LIMIT:,} characters")
        save_working(slug, record, "write", text)
        return RedirectResponse(f"/posts/{slug}#build-write", status_code=303)

    @app.post("/posts/{slug}/candidates/clear")
    def clear_candidates(slug: str, kind: Annotated[str, Form()] = "") -> RedirectResponse:
        record = post_or_404(slug)
        if writing_now(slug):
            raise HTTPException(409, "the model is writing this post right now; cancel that first")
        found = record.get("candidates")
        if kind not in {"outline", "write"} or not isinstance(found, dict):
            raise HTTPException(400, "unknown step")
        store.update_post(slug, candidates={k: v for k, v in found.items() if k != kind})
        return RedirectResponse(f"/posts/{slug}", status_code=303)

    @app.post("/posts/{slug}/cancel")
    def cancel_post(slug: str) -> RedirectResponse:
        if writing_now(slug):
            composer.cancel()
        return RedirectResponse(f"/posts/{slug}", status_code=303)

    @app.post("/posts/{slug}/delete")
    def delete_post(slug: str) -> RedirectResponse:
        post_or_404(slug)
        if writing_now(slug):
            raise HTTPException(409, "the model is writing this post right now; cancel that first")
        store.delete_post(slug)
        return RedirectResponse("/", status_code=303)

    @app.get("/posts/{slug}/post.md")
    def download_post(slug: str) -> PlainTextResponse:
        record = post_or_404(slug)
        text = str(record.get("post", ""))
        if not text:
            raise HTTPException(404, "this post has not been written yet")
        headers = {"Content-Disposition": f'attachment; filename="{slug}.md"'}
        return PlainTextResponse(
            text + "\n", media_type="text/markdown; charset=utf-8", headers=headers
        )

    @app.post("/posts/{slug}/to-draft")
    def post_to_draft(slug: str) -> RedirectResponse:
        record = post_or_404(slug)
        text = str(record.get("post", ""))
        if not text:
            raise HTTPException(400, "this post has not been written yet")
        base = slugify(str(record.get("title", slug)))[:70] or slug
        name, number = base, 1
        while store.read(name) is not None:
            number += 1
            name = f"{base}-{number}"
        try:
            new_slug = store.save(name, text + "\n")
        except QueueFull as exc:
            raise HTTPException(429, str(exc)) from exc
        except StorageError as exc:
            raise HTTPException(400, str(exc)) from exc
        return RedirectResponse(f"/drafts/{new_slug}", status_code=303)

    @app.post("/posts/{slug}/send-to-dev")
    def send_to_dev(slug: str) -> RedirectResponse:
        """Push the finished post to portfolio-dev as a new draft blog post,
        so Kien can tweak images and copy there before anything goes public.
        A direct in-cluster call (ClusterIP, not the public hostname) guarded
        by a shared token - see NIGHTSHIFT_PORTFOLIO_DEV_URL/_TOKEN."""
        record = post_or_404(slug)
        text = str(record.get("post", ""))
        if not text:
            raise HTTPException(400, "this post has not been written yet")
        url = os.environ.get("NIGHTSHIFT_PORTFOLIO_DEV_URL", "")
        token = os.environ.get("NIGHTSHIFT_PORTFOLIO_DEV_TOKEN", "")
        if not url or not token:
            raise HTTPException(503, "portfolio-dev integration is not configured")
        title = str(record.get("title", slug))
        payload = json.dumps({"title": title, "markdown": text}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "X-Ingest-Token": token},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise HTTPException(502, f"portfolio-dev refused it ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise HTTPException(502, f"could not reach portfolio-dev: {exc.reason}") from exc
        dev_slug = body.get("post", {}).get("slug", "")
        store.update_post(slug, sent_to_dev=time.time(), sent_to_dev_slug=dev_slug)
        return RedirectResponse(f"/posts/{slug}", status_code=303)

    @app.get("/posts/{slug}/run.json")
    def post_state(slug: str) -> dict[str, object]:
        post_or_404(slug)
        state = composer.state
        if state is None or state.slug != slug:
            return {"state": "idle"}
        return {
            "state": state.state,
            "kind": state.kind,
            "model": state.model,
            "chars": state.chars,
            "resting_until": state.resting_until,
            "error": state.error,
        }

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
