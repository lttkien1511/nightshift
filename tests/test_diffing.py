from nightshift import unified_diff, word_diff


def test_unified_diff_is_empty_when_nothing_changed():
    assert unified_diff("a\n", "a\n") == ""


def test_unified_diff_shows_the_change():
    diff = unified_diff("hello world\n", "hello there\n")
    assert "--- a/draft.md" in diff
    assert "+++ b/draft.md" in diff
    assert "-hello world" in diff
    assert "+hello there" in diff


def test_word_diff_marks_replaced_words():
    pairs = word_diff("the quick brown fox", "the slow brown fox")
    assert ("delete", "quick") in pairs
    assert ("insert", "slow") in pairs


def test_word_diff_can_rebuild_both_sides():
    before, after = "the quick brown fox", "the slow brown fox jumps"
    pairs = word_diff(before, after)
    assert "".join(t for op, t in pairs if op != "insert") == before
    assert "".join(t for op, t in pairs if op != "delete") == after


def test_word_diff_handles_vietnamese():
    pairs = word_diff("Tôi đã gỡ Rancher", "Mình đã gỡ Rancher")
    assert ("delete", "Tôi") in pairs
    assert ("insert", "Mình") in pairs
