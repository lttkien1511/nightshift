import pytest

from nightshift.merge import MergeError, merge
from nightshift.pipeline import process_document
from nightshift.worker import result_payload

TEXT = (
    "---\ntitle: Gỡ Rancher\n---\n\n"
    "# Tiêu đề\n\n"
    "Cái này tốt, xem [tài liệu](https://example.com/a) nhé.\n\n"
    "Cái này tốt, xem [tài liệu](https://example.com/a) nhé.\n\n"
    "- mục tốt một\n- mục hai\n\n"
    "```bash\nhelm uninstall rancher\n```\n"
)


def editor(block: str) -> str:
    return block.replace("tốt", "hay")


def blocks_of(text: str):
    result = process_document(text, editor)
    return result, result_payload("m", text, result)["blocks"]


def edited_indexes(blocks):
    return [b["index"] for b in blocks if b["status"] == "edited"]


def test_keeping_every_edit_equals_the_pipeline_output():
    result, blocks = blocks_of(TEXT)
    assert merge(TEXT, blocks, set(edited_indexes(blocks))) == result.edited


def test_keeping_nothing_returns_the_draft_unchanged():
    _, blocks = blocks_of(TEXT)
    assert merge(TEXT, blocks, set()) == TEXT


def test_identical_paragraphs_are_replaced_one_at_a_time():
    _, blocks = blocks_of(TEXT)
    first, second, _third = edited_indexes(blocks)
    merged = merge(TEXT, blocks, {second})
    assert merged.count("Cái này hay") == 1
    assert merged.index("Cái này hay") > merged.index("Cái này tốt")
    assert "- mục tốt một" in merged
    assert first != second


def test_code_links_and_front_matter_are_never_touched():
    _, blocks = blocks_of(TEXT)
    merged = merge(TEXT, blocks, set(edited_indexes(blocks)))
    assert merged.startswith("---\ntitle: Gỡ Rancher\n---\n")
    assert "```bash\nhelm uninstall rancher\n```" in merged
    assert merged.count("(https://example.com/a)") == 2


def test_crlf_drafts_keep_their_line_endings():
    crlf = TEXT.replace("\n", "\r\n")
    result, blocks = blocks_of(crlf)
    merged = merge(crlf, blocks, set(edited_indexes(blocks)))
    assert merged == result.edited
    assert "\r\n" in merged and "\n\n" not in merged.replace("\r\n", "")


def test_rejected_blocks_cannot_be_merged():
    blocks = [{"index": 1, "status": "rejected", "original": "a", "result": "a", "proposal": "b"}]
    assert merge("a\n", blocks, {1}) == "a\n"


def test_a_block_that_no_longer_matches_is_an_error():
    blocks = [{"index": 1, "status": "edited", "original": "gone", "result": "new"}]
    with pytest.raises(MergeError, match="block 1"):
        merge("something else\n", blocks, {1})
