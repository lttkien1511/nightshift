"""Turn a brief into prompts for a blog post, and check what comes back. No I/O in this file."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from .masking import PLACEHOLDER_RE, MaskingError, mask, unmask

TONES = {
    "conversational": "warm and conversational, as if telling a friend what happened",
    "professional": "professional and clear, without jargon for its own sake",
    "technical": "technical and concise, precise about the details",
    "reflective": "reflective, sharing what was learned and why it matters",
}
# language -> name for the prompt, and the marker used where the material is missing something
LANGUAGES = {"vi": "Vietnamese", "en": "English"}
TODO_MARKER = {"vi": "[cần bổ sung: ...]", "en": "[TODO: ...]"}
HIDDEN_LABEL = {"vi": "[đã ẩn]", "en": "[hidden]"}
# length -> (target words, token cap for the answer)
LENGTHS = {"short": (300, 900), "medium": (600, 1800)}
OUTLINE_TOKENS = 600
OUTLINE_PARTS = {"vi": ("Mở bài", "Kết bài"), "en": ("Opening", "Closing")}

LIMITS = {
    "topic": 200,
    "audience": 200,
    "tone_note": 300,
    "points": 3000,
    "material": 12000,
    "avoid": 2000,
    "outline": 3000,
}
MAX_TERMS = 50
MAX_TERM = 80

HIDE_RE = re.compile(r"\[\[HIDE_\d+\]\]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_TODO = re.compile(r"\[(?:cần bổ sung|TODO)[^\]]*\]", re.IGNORECASE)
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_INTERNAL_HOST = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.(?:lab|local|internal|corp|lan)\b", re.I)
_TOKENISH = re.compile(r"\b[A-Za-z0-9+/_-]{32,}\b")


class BriefError(ValueError):
    """The brief cannot be used. The message is safe to show to the user."""


def normalize_brief(form: Mapping[str, str]) -> dict[str, str]:
    """Clean the form fields and refuse what the model could not read."""

    def field(name: str) -> str:
        return str(form.get(name, "")).replace("\r\n", "\n").strip()

    brief = {
        "topic": field("topic"),
        "audience": field("audience"),
        "tone": field("tone") or "conversational",
        "tone_note": field("tone_note"),
        "length": field("length") or "short",
        "language": field("language") or "vi",
        "points": field("points"),
        "material": field("material"),
        "avoid": field("avoid"),
    }
    if not brief["topic"]:
        raise BriefError("say what the post is about")
    if brief["tone"] not in TONES:
        raise BriefError("unknown tone")
    if brief["length"] not in LENGTHS:
        raise BriefError("unknown length")
    if brief["language"] not in LANGUAGES:
        raise BriefError("unknown language")
    for name, limit in LIMITS.items():
        if len(brief.get(name, "")) > limit:
            raise BriefError(f"{name.replace('_', ' ')} is longer than {limit:,} characters")
    parse_terms(brief["avoid"])
    return brief


def parse_terms(text: str) -> list[tuple[str, str]]:
    """One term per line, optionally ``term => what to write instead``. Longest first."""
    found: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        term, _, replacement = line.partition("=>")
        term, replacement = term.strip(), replacement.strip()
        if not term:
            continue
        if len(term) > MAX_TERM:
            raise BriefError(f"a term to hide is longer than {MAX_TERM} characters")
        found.setdefault(term.lower(), (term, replacement))
    if len(found) > MAX_TERMS:
        raise BriefError(f"more than {MAX_TERMS} terms to hide")
    return sorted(found.values(), key=lambda pair: len(pair[0]), reverse=True)


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def hide_terms(text: str, terms: list[tuple[str, str]]) -> tuple[str, dict[str, int]]:
    """Replace each term by ``[[HIDE_n]]``. Returns the text and token -> term index."""
    tokens: dict[str, int] = {}
    for index, (term, _) in enumerate(terms):
        token = f"[[HIDE_{index + 1}]]"
        text, count = _term_pattern(term).subn(token, text)
        if count:
            tokens[token] = index
    return text, tokens


@dataclass(frozen=True)
class Inputs:
    system: str
    prompt: str
    num_predict: int
    originals: dict[str, str]
    terms: list[tuple[str, str]]
    language: str
    target_words: int
    source: str  # everything the model was given, for the fact check


def _system(kind: str, brief: Mapping[str, str]) -> str:
    language = LANGUAGES[brief["language"]]
    protected = (
        "Tokens like [[CODE_1]], [[URL_2]], [[LINK_3]], [[HTML_4]], [[BLOCK_5]] and "
        "[[HIDE_6]] stand "
        "for content you must not see. Copy each one exactly where it belongs, never explain or "
        "guess what is behind it, and never invent new ones. "
    )
    if kind == "outline":
        opening, closing = OUTLINE_PARTS[brief["language"]]
        if brief.get("points", "").strip():
            source = (
                "Build the structure from the key points: turn each one into a section, and you "
                "may merge, split or reorder them so the story flows naturally. "
            )
        else:
            source = "Choose the sections yourself from the material. "
        return (
            "You help plan a short blog post for a personal portfolio blog. "
            "Reply with the structure of the post as a list, one item per line, each starting "
            "with '- ' and written as 'Heading: one line on what this part says'. "
            f"The first item is '{opening}' (how the post begins) and the last is '{closing}' "
            f"(how it ends); between them come 3 to 6 sections. {source}"
            "Use only what the brief and the material say; do not add facts. "
            f"{protected}Write in {language}. No title, no introduction, no commentary."
        )
    marker = TODO_MARKER[brief["language"]]
    return (
        "You are a careful ghostwriter for a personal technical blog on a portfolio website. "
        "Write one blog post from the brief and the material. "
        "If an outline is given, follow its order and cover every part; its first and last "
        "items say how to open and how to close. "
        "Use ONLY facts, names, numbers, places, dates and commands that appear in the material. "
        "Never invent them and never write quotations. "
        f"When the post needs something the material does not give, write {marker} instead of "
        f"making it up. {protected}"
        f"Write in {language}, in a tone that is {TONES[brief['tone']]}. "
        "Reply with Markdown only: start with a single '# ' title line, then paragraphs, with "
        "'## ' headings only if they help. No front matter, no commentary before or after."
    )


def build_inputs(record: Mapping[str, object], kind: str) -> Inputs:
    """Everything the model will see. Hidden terms and code are replaced before it is built."""
    brief: Mapping[str, str] = record["brief"]  # type: ignore[assignment]
    terms = parse_terms(brief.get("avoid", ""))
    words = LENGTHS[brief["length"]][0]
    parts = [f"Topic: {brief['topic']}"]
    if brief.get("audience"):
        parts.append(f"Audience: {brief['audience']}")
    if brief.get("tone_note"):
        parts.append(f"Extra notes on the tone: {brief['tone_note']}")
    if kind == "write":
        parts.append(f"Length: about {words} words")
    if brief.get("points"):
        parts.append(f"Key points to cover:\n{brief['points']}")
    outline = str(record.get("outline", "")).strip()
    if kind == "write" and outline:
        parts.append(f"Outline to follow:\n{outline}")
    if brief.get("material"):
        parts.append(f"Material (the only source of facts):\n{brief['material']}")
    source = "\n\n".join(parts)
    hidden, _ = hide_terms(source, terms)
    try:
        masked = mask(hidden)
    except MaskingError as exc:
        raise BriefError(str(exc)) from exc
    return Inputs(
        system=_system(kind, brief),
        prompt=masked.text,
        num_predict=OUTLINE_TOKENS if kind == "outline" else LENGTHS[brief["length"]][1],
        originals=masked.originals,
        terms=terms,
        language=brief["language"],
        target_words=words,
        source=source,
    )


def clean_output(text: str) -> str:
    """Drop a wrapping code fence some models add around the whole answer."""
    body = text.strip()
    if body.startswith("```") and body.rstrip().endswith("```") and body.count("```") == 2:
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    return body.strip()


def restore(text: str, inputs: Inputs) -> tuple[str, int, int]:
    """Put the real code and links back; hidden terms become their stand-ins.

    Returns the text, how many raw hidden terms the model wrote on its own and had to be
    redacted, and how many tokens it invented that stand for nothing.
    """
    text = unmask(text, inputs.originals)
    for index, (_term, replacement) in enumerate(inputs.terms):
        label = replacement or HIDDEN_LABEL[inputs.language]
        text = text.replace(f"[[HIDE_{index + 1}]]", label)
    redacted = 0
    for term, replacement in inputs.terms:
        label = replacement or HIDDEN_LABEL[inputs.language]
        text, count = _term_pattern(term).subn(lambda _m, label=label: label, text)
        redacted += count
    invented = len(PLACEHOLDER_RE.findall(text)) + len(HIDE_RE.findall(text))
    text = HIDE_RE.sub(HIDDEN_LABEL[inputs.language], PLACEHOLDER_RE.sub("", text))
    return text, redacted, invented


def preview(text: str, inputs: Inputs) -> str:
    """The text so far, for the page while the model is still writing.

    Real code and links are put back and hidden terms already show their stand-ins, so the
    page never shows more than the finished post will.
    """
    text = re.sub(r"\[\[?[A-Z_0-9]*$", "", text)  # a token still being typed
    text = unmask(text, inputs.originals)
    for index, (_term, replacement) in enumerate(inputs.terms):
        text = text.replace(f"[[HIDE_{index + 1}]]", replacement or HIDDEN_LABEL[inputs.language])
    for term, replacement in inputs.terms:
        label = replacement or HIDDEN_LABEL[inputs.language]
        text = _term_pattern(term).sub(lambda _m, label=label: label, text)
    return text


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def check_post(post: str, inputs: Inputs, redacted: int, invented: int) -> list[dict[str, str]]:
    """What a person should look at before publishing. Warnings first."""
    found: list[dict[str, str]] = []
    known = _numbers(inputs.source)
    new_numbers = sorted(n for n in _numbers(post) - known if not (n.isdigit() and int(n) <= 10))
    if new_numbers:
        found.append(
            {
                "level": "warn",
                "text": "Numbers in the post that the material does not contain: "
                + ", ".join(new_numbers[:12]),
            }
        )
    sensitive = (
        _IPV4.findall(post)
        + _EMAIL.findall(post)
        + _INTERNAL_HOST.findall(post)
        + _TOKENISH.findall(post)
    )
    if sensitive:
        shown = ", ".join(dict.fromkeys(s for s in sensitive))[:200]
        found.append({"level": "warn", "text": f"Looks private, check before publishing: {shown}"})
    if redacted:
        found.append(
            {
                "level": "warn",
                "text": f"The model wrote {redacted} hidden term(s) itself; redacted.",
            }
        )
    if invented:
        found.append({"level": "info", "text": f"Removed {invented} token(s) the model made up."})
    todos = len(_TODO.findall(post))
    if todos:
        found.append({"level": "warn", "text": f"{todos} place(s) marked as missing material."})
    words = len(post.split())
    target = inputs.target_words
    if words < target * 0.5 or words > target * 1.5:
        found.append({"level": "info", "text": f"{words} words; the target was about {target}."})
    if not post.lstrip().startswith("# "):
        found.append({"level": "info", "text": "The post has no title line."})
    return sorted(found, key=lambda item: item["level"] != "warn")
