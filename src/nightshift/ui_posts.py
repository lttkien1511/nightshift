"""HTML for the blog post side of the UI: the brief form, the outline and the finished post."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from .compose import LENGTHS, LIMITS, POST_LIMIT, TONES, pieces, uncovered_points
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


def sent_to_dev_note(record: Mapping[str, Any]) -> str:
    sent = record.get("sent_to_dev")
    if not sent:
        return ""
    when = datetime.fromtimestamp(float(sent), UTC).strftime("%H:%M %d/%m")
    return f'<p class="hint">Sent to Portfolio Dev as a draft at {esc(when)} UTC.</p>'


SIDE_LIMIT = 8


def side_row(href: str, label: str, state: str, title: str, current: bool, query: str) -> str:
    """One line of the sidebar: a dot for the state and the name, cut short if it is long."""
    page = ' aria-current="page"' if current else ""
    return (
        f'<li data-q="{esc(query.lower())}"><a href="{esc(href)}"{page} title="{esc(title)}">'
        f'<i class="dot {esc(state)}" role="img" aria-label="{esc(state)}"></i>'
        f'<span class="t">{esc(label)}</span></a></li>'
    )


def side_group(name: str, title: str, rows: Sequence[tuple[str, bool]], empty: str) -> str:
    """A group that can be folded. Only the newest rows show; the rest sit behind "Show all"."""
    shown, rest = rows[:SIDE_LIMIT], rows[SIDE_LIMIT:]
    listing = "".join(row for row, _ in shown) or f'<li class="empty">{esc(empty)}</li>'
    more = ""
    if rest:
        opened = " open" if any(current for _, current in rest) else ""
        more = (
            f'<details class="rest"{opened}><summary>'
            f'<span class="s-closed">Show all ({len(rows)})</span>'
            '<span class="s-open">Show fewer</span></summary>'
            f'<ul class="queue">{"".join(row for row, _ in rest)}</ul></details>'
        )
    return (
        f'<details class="group" data-group="{esc(name)}" open><summary>{esc(title)} '
        f'<span class="count">{len(rows)}</span></summary>'
        f'<ul class="queue">{listing}</ul>{more}</details>'
    )


def posts_group(posts: Sequence[PostItem], active: str, running: str) -> str:
    rows = []
    for item in posts:
        state = "writing" if item.slug == running else item.stage
        label = "writing" if state == "writing" else STAGE_LABEL.get(state, state)
        rows.append(
            (
                side_row(
                    f"/posts/{item.slug}",
                    item.title,
                    state,
                    f"{item.title} · {label}",
                    item.slug == active,
                    f"{item.title} {item.slug}",
                ),
                item.slug == active,
            )
        )
    return side_group("posts", "Blog posts", rows, "No blog posts yet.")


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


def _boxes(models: Sequence[str], chosen: Sequence[str], name: str) -> str:
    ticked = [m for m in chosen if m in models] or list(models[:1])
    return "".join(
        f'<label class="pick"><input type="checkbox" name="{esc(name)}" value="{esc(m)}"'
        f"{' checked' if m in ticked else ''}> {esc(m)}</label>"
        for m in models
    )


def _model_picks(models: Sequence[str], chosen: Sequence[str]) -> str:
    """Tick the models that take this step. With one model there is nothing to choose."""
    if len(models) < 2:
        return ""
    return (
        '<fieldset class="picks"><legend>Models</legend>'
        f"{_boxes(models, chosen, 'models')}"
        '<span class="hint">Tick several to compare them. They run one after another.</span>'
        "</fieldset>"
    )


def brief_form(
    values: Mapping[str, str] | None,
    action: str,
    can_write: bool,
    error: str = "",
    existing: bool = False,
    models: Sequence[str] = (),
    picks: Mapping[str, Sequence[str]] | None = None,
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
    chosen = picks or {}
    chooser = (
        '<fieldset class="voice"><legend>Models</legend>'
        '<p class="hint">Tick the models that take part. With several, they run one after '
        "another and you pick the best parts of each.</p>"
        '<div class="grid2"><div><b>For the outline</b><br>'
        f"{_boxes(models, chosen.get('outline', ()), 'models_outline')}</div>"
        "<div><b>For the post</b><br>"
        f"{_boxes(models, chosen.get('write', ()), 'models_write')}</div></div></fieldset>"
        if len(models) > 1
        else ""
    )
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
        + chooser
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


def _turn(run: PostRun) -> str:
    if len(run.queue) < 2:
        return ""
    return f" (model {run.position + 1} of {len(run.queue)})"


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
    if run.running and run.resting_until > now:
        return (
            '<section class="card runcard" aria-live="polite"><div class="runhead">'
            f"<b>Resting before {esc(run.model)}{_turn(run)}</b>"
            f'<span class="hint">{_elapsed(run.resting_until - now)} left</span></div>'
            '<p class="hint">The machine rests between models, so it can cool down and free '
            "its memory. The next model starts by itself.</p>"
            f'<form method="post" action="{base}/cancel"><button>Cancel</button></form>'
            "</section>"
        )
    if run.running:
        what = {"outline": "outline", "voices": "three voices"}.get(run.kind, "post")
        shown = esc(run.partial) if run.partial else "Waiting for the first words..."
        return (
            '<section class="card runcard" aria-live="polite"><div class="runhead">'
            f"<b>Writing the {what} with {esc(run.model)}{_turn(run)}</b>"
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
    if run.state == "done" and run.error:
        return f'<p class="banner">Some models failed: {esc(run.error)}</p>'
    return ""


def _candidates(slug: str, record: Mapping[str, Any], kind: str, grain: str) -> str:
    """The answers of the models, in parts a person can tick and add to their own text."""
    found = record.get("candidates")
    book = found.get(kind) if isinstance(found, dict) else None
    answers = [a for a in book or [] if isinstance(a, dict) and a.get("id")]
    if not answers:
        return ""
    base = f"/posts/{esc(slug)}"
    mine = str(record.get("outline" if kind == "outline" else "post", ""))
    noun = "outline" if kind == "outline" else "post"
    cards = []
    for answer in reversed(answers):
        text = str(answer.get("text", ""))
        spans = _spans(text, answer.get("flags"))
        items = []
        for number, (start, end, part) in enumerate(pieces(text, kind, grain)):
            added = part in mine
            odd = any(a < end and b > start for a, b in spans)
            classes = "piece" + (" added" if added else "") + (" odd" if odd else "")
            tag = (
                '<span class="tag">in your text</span>'
                if added
                else '<span class="tag" title="May say more than your material does">check</span>'
                if odd
                else ""
            )
            items.append(
                f'<label class="{classes}"><input type="checkbox" name="pick" '
                f'value="{esc(answer["id"])}.{number}"><span>{esc(part)}</span>{tag}</label>'
            )
        seconds = answer.get("seconds")
        took = f" &middot; {_elapsed(float(seconds))}" if isinstance(seconds, int | float) else ""
        cards.append(
            f"<details open><summary>{esc(answer.get('model', 'model'))}{took} &middot; "
            f"{len(items)} part(s)</summary>{''.join(items)}</details>"
        )
    if kind == "write":
        other = "paragraph" if grain == "sentence" else "sentence"
        toggle = (
            f'<a class="grain" href="/posts/{esc(slug)}?grain={other}#build-ideas">'
            f"Show {other}s instead</a>"
        )
    else:
        toggle = ""
    return (
        '<div class="cands"><h3>Ideas from the models</h3>'
        f'<p class="hint">Tick the parts you like and add them to your {noun}. '
        f"Parts marked <b>check</b> may say more than your material does. {toggle}</p>"
        f'<form method="post" action="{base}/gather">'
        f'<input type="hidden" name="kind" value="{kind}">'
        f'<input type="hidden" name="grain" value="{esc(grain)}">'
        f"{''.join(cards)}"
        f'<div class="buttons"><button class="primary">Add the ticked parts to my {noun}'
        "</button></div></form>"
        f'<form class="inline" method="post" action="{base}/candidates/clear" '
        f'data-confirm="Forget these answers? Your {noun} stays as it is.">'
        f'<input type="hidden" name="kind" value="{kind}">'
        '<button class="secondary">Clear these answers</button></form></div>'
    )


def _outline_card(
    slug: str,
    record: Mapping[str, Any],
    models: Sequence[str],
    busy: bool,
    picks: Mapping[str, Sequence[str]],
    grain: str,
) -> str:
    base = f"/posts/{esc(slug)}"
    outline = str(record.get("outline", ""))
    if not models or busy:
        return ""
    ask_outline = _model_picks(models, picks.get("outline", ()))
    ask_write = _model_picks(models, picks.get("write", ()))
    ideas = _candidates(slug, record, "outline", grain)
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
            '<section class="card" id="build-outline"><h2>Outline</h2>'
            '<p class="hint">Let the model draft the structure first (opening, sections, '
            "closing), so you can steer the post before it is written. About a minute each.</p>"
            f'<form class="runform" method="post" action="{base}/outline">{ask_outline}'
            '<button class="primary">Suggest an outline</button></form>'
            f'<form class="runform" method="post" action="{base}/write">{ask_write}'
            '<button class="secondary">Skip it and write the post now</button></form>'
            f"{ideas}</section>"
        )
    return (
        '<section class="card" id="build-outline"><h2>Outline</h2>'
        '<p class="hint">This is your outline. Edit it freely, or add the parts you like from '
        "the models below. The post follows it.</p>"
        f"{note}"
        f'<form method="post" action="{base}/write">'
        f'<textarea name="outline" class="outline" rows="8" maxlength="{LIMITS["outline"]}">'
        f"\n{esc(outline)}</textarea>"
        f"{ask_write}"
        '<div class="runform">'
        '<button class="primary">Write the post from this outline</button>'
        f'<button formaction="{base}/outline/save">Save outline</button>'
        "</div></form>"
        f'<form class="runform" method="post" action="{base}/outline" '
        'data-confirm="Ask for a new outline? With one model it replaces the outline above '
        "(changes you have not saved are lost). With several, their answers appear below "
        'for you to pick from.">'
        f"{ask_outline}<button>Suggest a new outline</button></form>{ideas}</section>"
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


def _post_card(
    slug: str,
    record: Mapping[str, Any],
    models: Sequence[str],
    busy: bool,
    picks: Mapping[str, Sequence[str]],
    grain: str,
) -> tuple[str, str]:
    """The post card (left column) and the checks and ideas that go beside it (right)."""
    post = str(record.get("post", ""))
    found = "" if busy else _candidates(slug, record, "write", grain)
    ideas = f'<section class="card" id="build-ideas">{found}</section>' if found else ""
    if not post and not ideas:
        return "", ""
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
        f"{_model_picks(models, picks.get('write', ()))}<button>Write it again</button></form>"
        if models and not busy
        else ""
    )
    editor = (
        '<details class="edit"><summary>Edit or build the text</summary>'
        f'<form method="post" action="{base}/post/save">'
        f'<textarea name="post" class="outline" rows="16" maxlength="{POST_LIMIT}">'
        f"\n{esc(post)}</textarea>"
        '<div class="buttons"><button class="primary">Save the text</button></div></form>'
        "</details>"
        if not busy
        else ""
    )
    who = record.get("model")
    byline = (
        "Put together from the parts you picked."
        if who == "picks"
        else f"Written by {esc(who or 'the model')}."
    )
    if not post:
        return (
            '<section class="card" id="build-write"><h2>The post</h2>'
            '<p class="hint">Nothing is here yet. Tick parts from the ideas to build it, or '
            f"write it yourself.</p>{editor}</section>",
            ideas,
        )
    checked = (
        '<section class="card" id="build-checks"><h2>Checks</h2>'
        '<p class="hint">These look for numbers and names the material does not contain, and '
        "for anything that looks private. They are not a substitute for reading it.</p>"
        f"{verdict}</section>"
    )
    written = (
        '<section class="card" id="build-write"><h2>The post</h2>'
        f'<p class="hint">{byline}</p>'
        f"{legend}"
        f'<div class="ppost">{marked(post, record.get("flags"))}</div>'
        f"{editor}"
        '<div class="buttons">'
        f'<a class="btn" href="{base}/post.md">Download .md</a>'
        f'<form class="inline" method="post" action="{base}/to-draft">'
        '<button class="secondary">Save as a draft to polish</button></form>'
        f'<form class="inline" method="post" action="{base}/send-to-dev">'
        '<button class="secondary">Send to Portfolio Dev</button></form>'
        f"{sent_to_dev_note(record)}"
        f"{again}</div></section>"
    )
    return written, checked + ideas


def post_main(
    record: Mapping[str, Any],
    run: PostRun | None,
    models: Sequence[str],
    now: float,
    grain: str = "paragraph",
    picks: Mapping[str, Sequence[str]] | None = None,
) -> str:
    chosen = picks or {}
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
    written, beside = _post_card(slug, record, models, busy, chosen, grain)
    voices = _voices_card(slug, record, models, busy)
    outline = _outline_card(slug, record, models, busy, chosen, grain)
    if str(record.get("post", "")):
        left, right = written, beside + voices + outline
    else:  # no text yet: the outline is the work in hand, so it takes the wide column
        left, right = outline + written, beside + voices
    return (
        title
        + _run_card(slug, run, now)
        + f'<div class="postgrid"><div class="postcol">{left}</div>'
        f'<div class="helpcol">{right}{_summary(record)}</div></div>'
    )
