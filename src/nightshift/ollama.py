"""A minimal Ollama client for the nightly worker. Standard library only."""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass


class OllamaError(RuntimeError):
    """The server was unreachable, too slow, or answered something unusable."""


@dataclass(frozen=True)
class OllamaConfig:
    url: str = "http://ollama:11434"
    model: str = "qwen2.5:7b"
    timeout: float = 900.0
    keep_alive: str = "5m"
    num_ctx: int = 4096
    temperature: float = 0.3


SYSTEM_PROMPT = (
    "You are a careful copy editor for a personal technical blog. "
    "You receive ONE paragraph, list or quote at a time. Improve clarity, rhythm and "
    "word choice, fix grammar and typos, and keep the author's voice and meaning. "
    "Reply in the same language as the input. "
    "Tokens that look like [[CODE_1]], [[URL_2]], [[LINK_3]], [[HTML_4]] or [[BLOCK_5]] are "
    "placeholders for content you must not see: copy every one of them exactly, in the same "
    "order, and never add new code, URLs or placeholders. "
    "Keep numbers, versions and durations unchanged. "
    "Keep list markers and quote markers and the number of lines. "
    "Return only the edited text, with no commentary, no quotes around it and no code fences."
)


def build_system(voice: str = "") -> str:
    """The system prompt, followed by the author's own style samples when there are any."""
    sample = voice.strip()
    if not sample:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\nSamples of the author's own voice, for style only:\n{sample}"


def _post(cfg: OllamaConfig, path: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        cfg.url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout) as response:  # noqa: S310
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        raise OllamaError(f"ollama answered HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaError(f"ollama is unreachable or too slow: {exc}") from exc
    except ValueError as exc:
        raise OllamaError("ollama sent a reply that is not JSON") from exc
    if not isinstance(body, dict):
        raise OllamaError("ollama sent an unexpected reply")
    return body


def generate(cfg: OllamaConfig, system: str, prompt: str) -> str:
    body = _post(
        cfg,
        "/api/generate",
        {
            "model": cfg.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "keep_alive": cfg.keep_alive,
            "options": {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx},
        },
    )
    text = body.get("response")
    if not isinstance(text, str) or not text.strip():
        raise OllamaError("ollama returned an empty answer")
    return text


def unload(cfg: OllamaConfig) -> None:
    """Free the model's memory. Failing to do so is harmless, so errors are ignored."""
    with contextlib.suppress(OllamaError):
        _post(cfg, "/api/generate", {"model": cfg.model, "keep_alive": 0, "stream": False})


@dataclass(frozen=True)
class OllamaInfo:
    reachable: bool
    models: tuple[tuple[str, int], ...] = ()
    error: str = ""


def list_models(url: str, timeout: float = 3.0) -> OllamaInfo:
    """Ask a server which models it has installed. Never raises: the status page uses it."""
    request = urllib.request.Request(url.rstrip("/") + "/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return OllamaInfo(False, error=str(exc))
    items = body.get("models") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return OllamaInfo(False, error="unexpected reply")
    found = [
        (str(m["name"]), int(m.get("size") or 0))
        for m in items
        if isinstance(m, dict) and "name" in m
    ]
    return OllamaInfo(True, models=tuple(sorted(found)))


class OllamaEditor:
    """An ``Editor`` for ``process_document`` that asks a local Ollama model."""

    def __init__(self, cfg: OllamaConfig, voice: str = "") -> None:
        self.cfg = cfg
        self._system = build_system(voice)

    def __call__(self, block: str) -> str:
        return generate(self.cfg, self._system, block)
