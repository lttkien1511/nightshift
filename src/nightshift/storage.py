"""Draft storage on a plain directory, shared by the web service and the nightly worker."""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

MAX_SLUG = 80
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class StorageError(ValueError):
    """The request is invalid. The message is safe to show to the user."""


class DraftTooLarge(StorageError):
    pass


class QueueFull(StorageError):
    pass


def slugify(name: str) -> str:
    """Turn a title such as ``Gỡ Rancher.md`` into a safe file name such as ``go-rancher``."""
    base = name.strip()
    if base.lower().endswith(".md"):
        base = base[:-3]
    base = base.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFKD", base)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", plain).strip("-")[:MAX_SLUG].strip("-")
    if not slug:
        raise StorageError("the draft name has no usable letters or digits")
    return slug


def is_slug(value: str) -> bool:
    return len(value) <= MAX_SLUG and _SLUG.fullmatch(value) is not None


def model_file(model: str) -> str:
    """A model tag such as ``qwen2.5:7b`` as a safe file stem such as ``qwen2.5_7b``."""
    stem = re.sub(r"[^a-z0-9.-]+", "_", model.lower()).strip("._-")
    if not stem:
        raise StorageError("the model name has no usable characters")
    return stem[:80]


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@dataclass(frozen=True)
class Draft:
    slug: str
    size: int
    has_results: bool


@dataclass(frozen=True)
class Store:
    root: Path
    max_bytes: int = 1_000_000
    max_queue: int = 20

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def outbox(self) -> Path:
        return self.root / "outbox"

    @property
    def voice_path(self) -> Path:
        return self.root / "voice.md"

    def _result_path(self, slug: str, model: str) -> Path:
        return self.outbox / slug / f"{model_file(model)}.json"

    def write_result(self, slug: str, model: str, payload: dict[str, object]) -> None:
        path = self._result_path(slug, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def result(self, slug: str, model: str) -> dict[str, object] | None:
        if not is_slug(slug):
            return None
        return _load_json(self._result_path(slug, model))

    def results(self, slug: str) -> list[dict[str, object]]:
        """Every model's result for a draft, ordered by model name."""
        if not is_slug(slug):
            return []
        found = (_load_json(p) for p in sorted((self.outbox / slug).glob("*.json")))
        return [r for r in found if r is not None]

    def ensure(self) -> None:
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, text: str) -> str:
        """Queue a draft and return its slug. Saving an existing slug replaces it."""
        slug = slugify(name)
        if not text.strip():
            raise StorageError("the draft is empty")
        data = text.encode("utf-8")
        if len(data) > self.max_bytes:
            raise DraftTooLarge(f"the draft is larger than {self.max_bytes} bytes")
        path = self.inbox / f"{slug}.md"
        if not path.exists() and len(list(self.inbox.glob("*.md"))) >= self.max_queue:
            raise QueueFull(f"the queue already holds {self.max_queue} drafts; delete one first")
        tmp = self.inbox / f".{slug}.tmp"
        tmp.write_bytes(data)
        tmp.replace(path)
        return slug

    def read(self, slug: str) -> str | None:
        if not is_slug(slug):
            return None
        path = self.inbox / f"{slug}.md"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def delete(self, slug: str) -> bool:
        if not is_slug(slug):
            return False
        path = self.inbox / f"{slug}.md"
        if not path.is_file():
            return False
        path.unlink()
        shutil.rmtree(self.outbox / slug, ignore_errors=True)
        return True

    def drafts(self) -> list[Draft]:
        found = []
        for path in sorted(self.inbox.glob("*.md")):
            results = self.outbox / path.stem
            has_results = results.is_dir() and any(results.iterdir())
            found.append(Draft(slug=path.stem, size=path.stat().st_size, has_results=has_results))
        return found
