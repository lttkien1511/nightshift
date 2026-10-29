from nightshift import CheckConfig, check_block


def test_identical_text_passes():
    assert check_block("paragraph", "Run [[CODE_1]] now.", "Run [[CODE_1]] now.") == []


def test_empty_result_is_rejected():
    assert check_block("paragraph", "Some text.", "  ") == ["model returned an empty result"]


def test_lost_placeholder_is_reported():
    problems = check_block("paragraph", "Run [[CODE_1]] now", "Run now")
    assert problems == ["placeholders lost: [[CODE_1]]"]


def test_duplicated_placeholder_is_reported():
    problems = check_block("paragraph", "Run [[CODE_1]] now", "Run [[CODE_1]] and [[CODE_1]] now")
    assert problems == ["placeholders added or duplicated: [[CODE_1]]"]


def test_reordered_placeholders_fail_in_strict_mode_only():
    before = "A [[CODE_1]] B [[CODE_2]]"
    after = "A [[CODE_2]] B [[CODE_1]]"
    assert check_block("paragraph", before, after) == ["placeholders were reordered"]
    assert check_block("paragraph", before, after, CheckConfig(strict_order=False)) == []


def test_model_cannot_introduce_backticks_or_urls():
    assert "model introduced backticks" in check_block(
        "paragraph", "Run it now please.", "Run `it` now please."
    )
    assert "model introduced a URL" in check_block(
        "paragraph", "Read the docs please.", "Read the docs at http://x.io please."
    )


def test_changed_numbers_are_rejected_but_placeholder_digits_are_ignored():
    assert check_block("paragraph", "It takes 30 seconds", "It takes 40 seconds") == [
        "numbers changed"
    ]
    assert check_block("paragraph", "Run [[CODE_12]] 3 times", "Run [[CODE_12]] 3 times") == []


def test_big_length_change_is_rejected():
    problems = check_block("paragraph", "This is a reasonably long sentence about nothing.", "ok.")
    assert len(problems) == 1
    assert problems[0].startswith("length changed too much")


def test_list_must_keep_its_lines_and_markers():
    assert check_block("list", "- a\n- b", "- a")[0].startswith("line count changed")
    assert check_block("list", "- a\n- b", "- a\n* b") == ["list or quote markers changed"]
    assert check_block("list", "- a\n  - b", "- a\n- b") == ["list or quote markers changed"]
    assert check_block("list", "- a\n- b", "- aa\n- bb") == []


def test_quote_must_keep_its_markers():
    assert check_block("quote", "> a\n> b", "> a\nb") == ["list or quote markers changed"]


def test_paragraph_must_stay_one_paragraph():
    assert check_block("paragraph", "Some text here.", "Some text.\n\nMore text.") == [
        "paragraph was split by a blank line"
    ]
    original = "One line only here for test purposes."
    assert check_block("paragraph", original, f"Sure! Here it is:\n{original}") == [
        "single-line paragraph gained line breaks"
    ]
