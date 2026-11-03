"""Draft storage on a plain directory, shared by the web service and the nightly worker."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MAX_SLUG = 80
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class StorageError(ValueError):
    """The request is invalid. The message is safe to show to the user."""


class DraftTooLarge(StorageError):
    pass


class QueueFull(StorageError):
    pass


class NameTaken(StorageError):
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


def digest(text: str) -> str:
    """Fingerprint of a draft, so a result can tell whether the draft changed since."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


MAX_POSTS = 20
_POST_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class PostItem:
    slug: str
    title: str
    stage: str


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
    def posts_dir(self) -> Path:
        return self.root / "posts"

    def post(self, slug: str) -> dict[str, object] | None:
        if not is_slug(slug):
            return None
        return _load_json(self.posts_dir / f"{slug}.json")

    def posts(self) -> list[PostItem]:
        found = []
        for path in sorted(self.posts_dir.glob("*.json"), key=lambda p: p.stem):
            record = _load_json(path)
            if record is not None:
                found.append(
                    PostItem(
                        path.stem, str(record.get("title", path.stem)), str(record.get("stage", ""))
                    )
                )
        return found

    def _write_post(self, slug: str, record: dict[str, object]) -> None:
        path = self.posts_dir / f"{slug}.json"
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def create_post(self, title: str, brief: dict[str, str]) -> str:
        """Start a blog post from a brief. The address comes from the title and stays unique."""
        base = slugify(title)[:70].strip("-") or "post"
        with _POST_LOCK:
            if len(list(self.posts_dir.glob("*.json"))) >= MAX_POSTS:
                raise QueueFull(f"there are already {MAX_POSTS} blog posts; delete one first")
            slug, number = base, 1
            while (self.posts_dir / f"{slug}.json").exists():
                number += 1
                slug = f"{base}-{number}"
            now = _now()
            self._write_post(
                slug,
                {
                    "slug": slug,
                    "title": title,
                    "brief": brief,
                    "stage": "brief",
                    "outline": "",
                    "post": "",
                    "model": "",
                    "checks": [],
                    "error": "",
                    "created": now,
                    "updated": now,
                },
            )
        return slug

    def update_post(self, slug: str, **fields: object) -> dict[str, object]:
        with _POST_LOCK:
            record = self.post(slug)
            if record is None:
                raise StorageError("no such blog post")
            record.update(fields)
            record["updated"] = _now()
            self._write_post(slug, record)
            return record

    def delete_post(self, slug: str) -> bool:
        if not is_slug(slug):
            return False
        path = self.posts_dir / f"{slug}.json"
        if not path.is_file():
            return False
        path.unlink()
        return True

    @property
    def voice_path(self) -> Path:
        return self.root / "voice.md"

    @property
    def worker_status_path(self) -> Path:
        return self.root / "worker-status.json"

    def write_worker_status(self, payload: dict[str, object]) -> None:
        tmp = self.worker_status_path.with_name(".worker-status.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.worker_status_path)

    def worker_status(self) -> dict[str, object] | None:
        """What the last worker run reported, or ``None`` if it has never run."""
        return _load_json(self.worker_status_path)

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
        self.posts_dir.mkdir(parents=True, exist_ok=True)

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

    def validate(self, text: str) -> None:
        """Raise the error ``save`` would raise for this text, without saving anything."""
        if not text.strip():
            raise StorageError("the draft is empty")
        if len(text.encode("utf-8")) > self.max_bytes:
            raise DraftTooLarge(f"the draft is larger than {self.max_bytes} bytes")

    def rename(self, old: str, new: str) -> None:
        """Give a draft another name. Its results move with it and stay valid."""
        if not is_slug(old) or not is_slug(new):
            raise StorageError("the draft name has no usable letters or digits")
        source = self.inbox / f"{old}.md"
        target = self.inbox / f"{new}.md"
        if not source.is_file():
            raise StorageError("no such draft")
        if target.exists():
            raise NameTaken("a draft with that name already exists")
        source.replace(target)
        shutil.rmtree(self.outbox / new, ignore_errors=True)
        results = self.outbox / old
        if results.is_dir():
            results.replace(self.outbox / new)

    def drafts(self) -> list[Draft]:
        found = []
        for path in sorted(self.inbox.glob("*.md")):
            results = self.outbox / path.stem
            has_results = results.is_dir() and any(results.iterdir())
            found.append(Draft(slug=path.stem, size=path.stat().st_size, has_results=has_results))
        return found
