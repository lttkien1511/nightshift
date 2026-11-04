"""HTML for the blog post side of the UI: the brief form, the outline and the finished post."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from .compose import LENGTHS, LIMITS, TONES, uncovered_points
from .composer import PostRun
from .storage import PostItem

STAGE_LABEL = {"brief": "brief", "outline": "outline", "done": "written"}
TONE_LABEL = {
    "conversational": "Conversational",
    "professional": "Professional",
    "technical": "Technical and concise",
    "reflective": "Reflective",
}
LENGTH_LABEL = {"short": "Short, about 300 words", "medium": "Medium, about 600 words"}
LANGUAGE_LABEL = {"vi": "Vietnamese", "en": "English"}
MODE_LABEL = {
    "expand": "Expand: explain and reflect in my voice",
    "close": "Stay close to my notes",
}
SENTENCE_LABEL = {"short": "Short", "mixed": "A mix", "long": "Longer and flowing"}
HUMOR_LABEL = {"none": "None", "light": "A little", "playful": "Playful"}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def posts_section(posts: Sequence[PostItem], active: str, running: str) -> str:
    items = []
    for item in posts:
        state = "writing" if item.slug == running else item.stage
        label = "writing" if state == "writing" else STAGE_LABEL.get(state, state)
        current = ' aria-current="page"' if item.slug == active else ""
        items.append(
            f'<li><a href="/posts/{esc(item.slug)}"{current}>'
            f'<span class="t">{esc(item.title)}</span>'
            f'<span class="badge {esc(state)}">{esc(label)}</span></a></li>'
        )
    listing = "".join(items) or '<li class="empty">No blog posts yet.</li>'
    return (
        '<h2 class="sidehead">Blog posts</h2>'
        '<a class="new-draft alt" href="/posts/new">New blog post</a>'
        f'<ul class="queue posts">{listing}</ul>'
    )


def _spans(text: str, flags: object) -> list[tuple[int, int]]:
    """The flagged spans that are well formed, in order and without overlaps."""
    found: list[tuple[int, int]] = []
    if isinstance(flags, list):
        for pair in flags:
            if (
                isinstance(pair, list)
                and len(pair) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in pair)
            ):
                start, end = pair
                if 0 <= start < end <= len(text):
                    found.append((start, end))
    kept: list[tuple[int, int]] = []
    for start, end in sorted(found):
        if not kept or start >= kept[-1][1]:
            kept.append((start, end))
    return kept


def marked(text: str, flags: object) -> str:
    """The post as HTML with the flagged parts highlighted."""
    pieces: list[str] = []
    position = 0
    for start, end in _spans(text, flags):
        pieces.append(esc(text[position:start]))
        pieces.append(
            '<mark class="odd" title="May say more than your material does">'
            f"{esc(text[start:end])}</mark>"
        )
        position = end
    pieces.append(esc(text[position:]))
    return "".join(pieces)


def _select(name: str, options: Mapping[str, str], chosen: str) -> str:
    items = "".join(
        f'<option value="{esc(key)}"{" selected" if key == chosen else ""}>{esc(label)}</option>'
        for key, label in options.items()
    )
    return f'<select name="{esc(name)}" id="{esc(name)}">{items}</select>'


def _model_select(models: Sequence[str]) -> str:
    if len(models) < 2:
        return ""
    options = "".join(f'<option value="{esc(m)}">{esc(m)}</option>' for m in models)
    return f'<label class="inline">Model<select name="model">{options}</select></label>'


def brief_form(
    values: Mapping[str, str] | None,
    action: str,
    can_write: bool,
    error: str = "",
    existing: bool = False,
) -> str:
    """The brief: what the post is about, for whom, in what voice, and what must stay out."""
    v = {
        "tone": "conversational",
        "length": "short",
        "language": "vi",
        "mode": "expand",
        "voice_sentences": "mixed",
        "voice_humor": "light",
        **(values or {}),
    }
    note = f'<p class="banner">{esc(error)}</p>' if error else ""
    tones = {k: TONE_LABEL[k] for k in TONES}
    lengths = {k: LENGTH_LABEL[k] for k in LENGTHS}

    def text(name: str, label: str, hint: str = "", required: bool = False) -> str:
        need = " required" if required else ""
        return (
            f'<label>{label} <span class="hint">{hint}</span>'
            f'<input name="{name}" id="{name}" maxlength="{LIMITS[name]}" autocomplete="off"'
            f' value="{esc(v.get(name, ""))}"{need}></label>'
        )

    def area(name: str, label: str, hint: str, rows: int) -> str:
        return (
            f'<label>{label} <span class="hint">{hint}</span>'
            f'<textarea name="{name}" id="{name}" rows="{rows}" maxlength="{LIMITS[name]}">'
            f"\n{esc(v.get(name, ''))}</textarea></label>"
        )

    if existing:
        buttons = '<button class="primary">Save the brief</button>'
    elif can_write:
        buttons = (
            '<button class="primary" name="action" value="outline">Suggest an outline</button>'
            '<button class="secondary" name="action" value="write">Write the post now</button>'
            '<button name="action" value="save">Save for later</button>'
        )
    else:
        buttons = '<button class="primary" name="action" value="save">Save the brief</button>'
    heading = "Edit the brief" if existing else "New blog post"
    lead = (
        "Tell the model what to write. The more it is given, the less it has to guess. "
        "It writes in your name only from the material below and marks what is missing."
    )
    return (
        f"<h1>{heading}</h1>"
        f'<p class="lead">{lead}</p>{note}'
        f'<form class="card" method="post" action="{esc(action)}">'
        + text("topic", "Topic or title", "what the post is about", required=True)
        + '<div class="grid2">'
        + text("audience", "Who reads it", "for example a recruiter, or someone new to Kubernetes")
        + f"<label>Tone {_select('tone', tones, v['tone'])}</label>"
        + f"<label>Length {_select('length', lengths, v['length'])}</label>"
        + f"<label>Language {_select('language', LANGUAGE_LABEL, v['language'])}</label>"
        + "</div>"
        + '<fieldset class="voice"><legend>Your voice</legend><div class="grid2">'
        + f"<label>How far may it go {_select('mode', MODE_LABEL, v['mode'])}</label>"
        + text(
            "voice_address", "What you call yourself and the reader", "for example mình - các bạn"
        )
        + "<label>Sentence length "
        + _select("voice_sentences", SENTENCE_LABEL, v["voice_sentences"])
        + "</label>"
        + f"<label>Humor {_select('voice_humor', HUMOR_LABEL, v['voice_humor'])}</label>"
        + "</div>"
        + text("voice_avoid", "Words and phrases you never use", "separated by commas")
        + text("tone_note", "Anything else about the voice", "optional")
        + area(
            "voice_sample",
            "A sample of your voice",
            "optional: a paragraph the way you write or talk. Not sure? Save the brief, then "
            "let the model try three voices and pick the closest",
            4,
        )
        + "</fieldset>"
        + area(
            "story",
            "Your own story",
            "what did you see, think or feel? Was anything surprising or funny? What should "
            "the reader take away? The model uses this in the first person and cannot make "
            "it up for you",
            6,
        )
        + area("points", "Key points", "one per line", 5)
        + area(
            "material",
            "Material",
            "your notes, commands, links, numbers: the only source of facts "
            f"(up to {LIMITS['material']:,} characters)",
            14,
        )
        + area(
            "avoid",
            "Never mention",
            "one per line; the model never sees these. Write <code>term =&gt; stand-in</code> "
            "to choose what replaces it, otherwise it becomes [hidden]",
            4,
        )
        + f'<div class="buttons">{buttons}</div></form>'
    )


def _summary(record: Mapping[str, Any]) -> str:
    brief: Mapping[str, str] = record["brief"]
    rows = [
        ("Topic", brief.get("topic", "")),
        ("Reader", brief.get("audience", "")),
        (
            "Tone",
            TONE_LABEL.get(brief.get("tone", ""), "")
            + (f"; {brief['tone_note']}" if brief.get("tone_note") else ""),
        ),
        ("Length", LENGTH_LABEL.get(brief.get("length", ""), "")),
        ("Mode", MODE_LABEL.get(brief.get("mode", "expand"), "")),
        (
            "Voice",
            "; ".join(
                part
                for part in (
                    brief.get("voice_address", ""),
                    SENTENCE_LABEL.get(brief.get("voice_sentences", ""), ""),
                    HUMOR_LABEL.get(brief.get("voice_humor", ""), ""),
                    "has a voice sample" if brief.get("voice_sample") else "",
                )
                if part
            ),
        ),
        ("Your story", f"{len(brief['story']):,} characters" if brief.get("story") else ""),
        ("Language", LANGUAGE_LABEL.get(brief.get("language", ""), "")),
        ("Key points", brief.get("points", "")),
        (
            "Never mention",
            f"{len([t for t in brief.get('avoid', '').splitlines() if t.strip()])} term(s)",
        ),
    ]
    body = "".join(
        f"<tr><th>{esc(label)}</th><td>{esc(value)}</td></tr>" for label, value in rows if value
    )
    material = len(brief.get("material", ""))
    return (
        '<details class="card"><summary>The brief</summary>'
        f'<table class="brief">{body}<tr><th>Material</th><td>{material:,} characters</td></tr>'
        "</table></details>"
    )


def _elapsed(seconds: float) -> str:
    minutes, rest = divmod(max(int(seconds), 0), 60)
    return f"{minutes}:{rest:02d}"


def _run_card(slug: str, run: PostRun | None, now: float) -> str:
    base = f"/posts/{esc(slug)}"
    if run is None:
        return ""
    if run.running and run.slug != slug:
        return (
            '<section class="card runcard"><b>Another blog post is being written.</b> '
            f'<span class="hint"><a href="/posts/{esc(run.slug)}">{esc(run.slug)}</a> has to '
            "finish first, because the model does one job at a time.</span></section>"
        )
    if run.slug != slug:
        return ""
    if run.running:
        what = {"outline": "outline", "voices": "three voices"}.get(run.kind, "post")
        shown = esc(run.partial) if run.partial else "Waiting for the first words..."
        return (
            '<section class="card runcard" aria-live="polite"><div class="runhead">'
            f"<b>Writing the {what} with {esc(run.model)}</b>"
            f'<span class="hint">{run.chars:,} characters &middot; '
            f"{_elapsed(now - run.started)} elapsed</span></div>"
            f'<pre class="pstream">{shown}</pre>'
            f'<form method="post" action="{base}/cancel"><button>Cancel</button></form>'
            '<p class="hint">This page reloads by itself. The first words can take up to a '
            "minute while the model loads.</p></section>"
        )
    if run.state == "failed":
        return f'<p class="banner">The last attempt failed: {esc(run.error)}</p>'
    if run.state == "cancelled":
        return '<p class="hint">The last attempt was cancelled.</p>'
    return ""


def _outline_card(slug: str, record: Mapping[str, Any], models: Sequence[str], busy: bool) -> str:
    base = f"/posts/{esc(slug)}"
    outline = str(record.get("outline", ""))
    picker = _model_select(models)
    if not models or busy:
        return ""
    brief = record.get("brief")
    points = str(brief.get("points", "")) if isinstance(brief, dict) else ""
    missing = uncovered_points(points, outline) if outline else []
    note = (
        '<div class="banner"><b>These key points do not seem to be in the outline:</b><ul>'
        + "".join(f"<li>{esc(point)}</li>" for point in missing)
        + "</ul>Add them below, or the post may leave them out.</div>"
        if missing
        else ""
    )
    if not outline:
        return (
            '<section class="card"><h2>Outline</h2>'
            '<p class="hint">Let the model draft the structure first (opening, sections, '
            "closing), so you can steer the post before it is written. About a minute.</p>"
            f'<form class="runform" method="post" action="{base}/outline">{picker}'
            '<button class="primary">Suggest an outline</button></form>'
            f'<form class="runform" method="post" action="{base}/write">{picker}'
            '<button class="secondary">Skip it and write the post now</button></form></section>'
        )
    return (
        '<section class="card"><h2>Outline</h2>'
        '<p class="hint">Edit it freely. The post follows this outline.</p>'
        f"{note}"
        f'<form method="post" action="{base}/write">'
        f'<textarea name="outline" class="outline" rows="8" maxlength="{LIMITS["outline"]}">'
        f"\n{esc(outline)}</textarea>"
        f'<div class="runform">{picker}'
        '<button class="primary">Write the post from this outline</button>'
        f'<button formaction="{base}/outline/save">Save outline</button>'
        "</div></form>"
        f'<form class="runform" method="post" action="{base}/outline" '
        'data-confirm="Replace this outline with a new one from the model? '
        'Changes you have not saved are lost.">'
        f"{picker}<button>Suggest a new outline</button></form></section>"
    )


def _voices_card(slug: str, record: Mapping[str, Any], models: Sequence[str], busy: bool) -> str:
    if not models or busy:
        return ""
    base = f"/posts/{esc(slug)}"
    found = record.get("voices")
    voices = [v for v in found if isinstance(v, str)] if isinstance(found, list) else []
    brief = record.get("brief")
    sample = str(brief.get("voice_sample", "")) if isinstance(brief, dict) else ""
    use = (
        f'<details><summary>The voice sample in use</summary><p class="sample">{esc(sample)}</p>'
        "</details>"
        if sample
        else ""
    )
    choose = ""
    if voices:
        options = "".join(
            f'<label class="choice"><input type="radio" name="choice" value="{n}"'
            f"{' required' if n == 0 else ''}><span>{esc(text)}</span></label>"
            for n, text in enumerate(voices)
        )
        choose = (
            f'<form method="post" action="{base}/voice">{options}'
            '<div class="buttons"><button class="primary">Use this voice</button></div></form>'
        )
    picker = _model_select(models)
    label = "Try three new voices" if voices else "Try three voices"
    return (
        '<section class="card"><h2>Find your voice</h2>'
        '<p class="hint">Not sure how you sound? The model writes the opening of your post three '
        "ways. Pick the one closest to you and the post follows it. About two minutes.</p>"
        f"{choose}{use}"
        f'<form class="runform" method="post" action="{base}/voices">{picker}'
        f"<button>{label}</button></form></section>"
    )


def _post_card(slug: str, record: Mapping[str, Any], models: Sequence[str], busy: bool) -> str:
    post = str(record.get("post", ""))
    if not post:
        return ""
    base = f"/posts/{esc(slug)}"
    checks = record.get("checks") or []
    count = len(_spans(post, record.get("flags")))
    legend = (
        f'<p class="hint"><mark class="odd">{count} highlighted part(s)</mark> may say more '
        "than your material does. Read each one, then delete it or rewrite it.</p>"
        if count
        else ""
    )
    items = "".join(
        f'<li class="{esc(c.get("level", "info"))}">{esc(c.get("text", ""))}</li>'
        for c in checks
        if isinstance(c, dict)
    )
    verdict = (
        f'<ul class="pcheck">{items}</ul>'
        if items
        else '<p class="hint">Nothing suspicious found. Read it anyway before you publish.</p>'
    )
    again = (
        f'<form class="inline" method="post" action="{base}/write">'
        f'<input type="hidden" name="outline" value="{esc(record.get("outline", ""))}">'
        f"{_model_select(models)}<button>Write it again</button></form>"
        if models and not busy
        else ""
    )
    return (
        '<section class="card"><h2>The post</h2>'
        f'<p class="hint">Written by {esc(record.get("model", "the model"))}. '
        "These checks look for numbers and names the material does not contain, and for "
        "anything that looks private. They are not a substitute for reading it.</p>"
        f"{verdict}"
        f"{legend}"
        f'<div class="ppost">{marked(post, record.get("flags"))}</div>'
        '<div class="buttons">'
        f'<a class="btn" href="{base}/post.md">Download .md</a>'
        f'<form class="inline" method="post" action="{base}/to-draft">'
        '<button class="secondary">Save as a draft to polish</button></form>'
        f"{again}</div></section>"
    )


def post_main(
    record: Mapping[str, Any],
    run: PostRun | None,
    models: Sequence[str],
    now: float,
) -> str:
    slug = str(record["slug"])
    base = f"/posts/{esc(slug)}"
    busy = run is not None and run.running
    delete = (
        f'<form class="inline" method="post" action="{base}/delete" '
        'data-confirm="Delete this blog post, its outline and the written text?">'
        '<button class="danger">Delete</button></form>'
    )
    title = (
        f'<div class="titlebar"><h1>{esc(record.get("title", slug))}</h1>'
        f'<div class="titlebtns"><a class="btn" href="{base}/edit">Edit the brief</a>'
        f"{delete}</div></div>"
    )
    return (
        title
        + _run_card(slug, run, now)
        + _post_card(slug, record, models, busy)
        + _voices_card(slug, record, models, busy)
        + _outline_card(slug, record, models, busy)
        + _summary(record)
    )
