import pytest

from nightshift import MaskingError, PipelineConfig, process_document
from nightshift.masking import PLACEHOLDER_RE

DOC = """---
title: Gỡ Rancher
---

# Gỡ Rancher khỏi k3s

Để gỡ finalizer, chạy `kubectl patch ns local --type=merge` rồi xem https://kubernetes.io/docs/ nhé.

- bước một: chạy `helm uninstall rancher`
- bước hai: đợi vài phút

```bash
kubectl get ns
```

> Lưu ý: namespace có thể kẹt 10 phút.
"""

FENCE = "```bash\nkubectl get ns\n```"


def identity(block: str) -> str:
    return block


def restyle(block: str) -> str:
    return block.replace("chạy", "hãy chạy").replace("đợi", "chờ")


def test_identity_editor_changes_nothing():
    result = process_document(DOC, identity)
    assert result.edited == DOC
    assert not result.changed
    assert result.stats == {"skipped": 3, "unchanged": 3}


def test_prose_is_edited_but_code_and_links_are_not():
    result = process_document(DOC, restyle)
    assert "hãy chạy `kubectl patch ns local --type=merge` rồi xem https://kubernetes.io/docs/" in (
        result.edited
    )
    assert "hãy chạy `helm uninstall rancher`" in result.edited
    assert "chờ vài phút" in result.edited
    assert FENCE in result.edited
    assert result.edited.startswith("---\ntitle: Gỡ Rancher\n---\n")
    assert result.stats == {"skipped": 3, "edited": 2, "unchanged": 1}


def test_an_editor_that_shouts_cannot_touch_code():
    result = process_document(DOC, str.upper)
    assert "ĐỂ GỠ FINALIZER" in result.edited
    assert "`kubectl patch ns local --type=merge`" in result.edited
    assert "https://kubernetes.io/docs/" in result.edited
    assert FENCE in result.edited
    assert result.stats["edited"] == 3


def test_dropping_a_placeholder_rejects_the_block_and_keeps_the_original():
    def drop_code(block: str) -> str:
        return PLACEHOLDER_RE.sub("", block)

    result = process_document(DOC, drop_code)
    assert result.edited == DOC
    rejected = [b for b in result.blocks if b.status == "rejected"]
    assert {b.kind for b in rejected} == {"paragraph", "list"}
    assert all(b.reasons[0].startswith("placeholders lost") for b in rejected)
    assert all(b.proposal for b in rejected)


def test_injecting_a_url_is_rejected_everywhere():
    result = process_document(DOC, lambda block: block + " http://evil.example")
    assert result.edited == DOC
    assert result.stats == {"skipped": 3, "rejected": 3}


def test_a_crashing_editor_loses_nothing():
    def boom(block: str) -> str:
        raise RuntimeError("model crashed")

    result = process_document(DOC, boom)
    assert result.edited == DOC
    assert result.stats == {"skipped": 3, "error": 3}
    assert "model crashed" in result.blocks[2].reasons[0]


def test_headings_are_skipped_unless_enabled():
    def rename(block: str) -> str:
        return block.replace("Gỡ", "Dỡ")

    default = process_document(DOC, rename)
    assert "# Gỡ Rancher khỏi k3s" in default.edited

    enabled = process_document(DOC, rename, PipelineConfig(edit_headings=True))
    assert "# Dỡ Rancher khỏi k3s" in enabled.edited
    assert "title: Gỡ Rancher" in enabled.edited


def test_windows_line_endings_survive_a_round_trip():
    crlf = DOC.replace("\n", "\r\n")
    assert process_document(crlf, identity).edited == crlf

    edited = process_document(crlf, restyle).edited
    assert "hãy chạy" in edited
    assert "\n" not in edited.replace("\r\n", "")


def test_spacing_between_blocks_is_preserved():
    text = "  \n\nHello world.   \n\n\n\nBye."
    result = process_document(text, lambda block: block.replace("Hello", "Hi"))
    assert result.edited == "  \n\nHi world.   \n\n\n\nBye."


def test_unclosed_fence_is_reported():
    result = process_document("a\n\n```py\nx", identity)
    assert len(result.warnings) == 1


def test_empty_document():
    result = process_document("", identity)
    assert result.edited == ""


def test_placeholder_lookalikes_in_the_source_are_refused():
    with pytest.raises(MaskingError):
        process_document("see [[CODE_1]]", identity)
