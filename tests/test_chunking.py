import pytest

from nightshift import split
from nightshift.chunking import classify


@pytest.mark.parametrize(
    "text",
    [
        "",
        "one",
        "a\n\nb",
        "\n\na\n\n\n\nb\n\n",
        "  lead\n\ntrail  \n",
        "a\nb\n\nc\n",
        "  \n\nHello world.   \n\n\n\nBye.",
    ],
)
def test_split_is_lossless(text):
    assert "".join(chunk.render() for chunk in split(text)) == text


def test_split_keeps_outer_whitespace_out_of_the_core():
    chunk = split("Hello world.   \n")[0]
    assert (chunk.lead, chunk.core, chunk.trail) == ("", "Hello world.", "   \n")


def test_render_can_replace_the_core():
    chunk = split("Hello.  \n\nNext")[0]
    assert chunk.render("Hi.") == "Hi.  \n\n"


@pytest.mark.parametrize(
    ("core", "kind"),
    [
        ("Plain words here.", "paragraph"),
        ("# Title", "heading"),
        ("Title\n=====", "heading"),
        ("- a\n- b", "list"),
        ("1. a\n2. b", "list"),
        ("> quoted", "quote"),
        ("---", "protected"),
        ("***", "protected"),
        ("| a | b |\n|---|---|\n| 1 | 2 |", "protected"),
        ("[[BLOCK_1]]", "protected"),
        ("[[BLOCK_1]]\n[[CODE_2]]", "protected"),
        ("[^1]: note", "protected"),
        ("", "blank"),
    ],
)
def test_classify(core, kind):
    assert classify(core) == kind


def test_only_prose_blocks_are_editable():
    kinds = {c.kind: c.editable for c in split("# H\n\ntext\n\n- a\n\n> q\n\n---\n")}
    assert kinds == {
        "heading": False,
        "paragraph": True,
        "list": True,
        "quote": True,
        "protected": False,
    }
