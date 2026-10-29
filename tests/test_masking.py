import pytest

from nightshift import MaskingError, mask, unmask


def test_fenced_block_is_masked_and_restored():
    text = "Run this:\n\n```bash\nkubectl get pods\n```\n\nDone."
    m = mask(text)
    assert m.text == "Run this:\n\n[[BLOCK_1]]\n\nDone."
    assert m.originals["[[BLOCK_1]]"] == "```bash\nkubectl get pods\n```"
    assert unmask(m.text, m.originals) == text


def test_longer_fence_may_contain_shorter_fence():
    m = mask("````md\n```py\nx = 1\n```\n````\n")
    assert m.text == "[[BLOCK_1]]\n"
    assert not m.warnings


def test_tilde_fence_is_masked():
    m = mask("~~~\nplain\n~~~\n")
    assert m.text == "[[BLOCK_1]]\n"


def test_unclosed_fence_is_masked_to_the_end_with_a_warning():
    m = mask("a\n\n```py\nx = 1\nmore")
    assert m.text == "a\n\n[[BLOCK_1]]"
    assert len(m.warnings) == 1
    assert "unclosed" in m.warnings[0]


def test_inline_code_single_and_double_backticks():
    m = mask("Use `kubectl` or ``a`b`` here.")
    assert m.text == "Use [[CODE_1]] or [[CODE_2]] here."
    assert m.originals == {"[[CODE_1]]": "`kubectl`", "[[CODE_2]]": "``a`b``"}


def test_stray_backtick_is_left_alone():
    m = mask("It costs 5` only")
    assert m.text == "It costs 5` only"
    assert not m.originals


def test_inline_code_cannot_span_paragraphs():
    text = "a ` b\n\nc ` d"
    assert mask(text).text == text


def test_bare_url_excludes_closing_paren_and_period():
    m = mask("(see https://example.com/a).")
    assert m.text == "(see [[URL_1]])."
    assert m.originals["[[URL_1]]"] == "https://example.com/a"


def test_url_with_balanced_parentheses_stays_whole():
    m = mask("https://en.wikipedia.org/wiki/Foo_(bar) is")
    assert m.text == "[[URL_1]] is"


def test_markdown_link_keeps_its_text_and_hides_the_url():
    m = mask("[docs](https://x.dev/y)")
    assert m.text == "[docs]([[URL_1]])"


def test_relative_link_destination_with_title_is_masked():
    m = mask('See [img](./a/b.png "A title") now')
    assert m.text == "See [img]([[LINK_1]]) now"
    assert m.originals["[[LINK_1]]"] == './a/b.png "A title"'


def test_reference_definition_is_masked_with_nested_url_expanded():
    m = mask('[ref]: https://example.com/x "T"\n')
    assert m.text == "[[LINK_1]]\n"
    assert m.originals["[[LINK_1]]"] == '[ref]: https://example.com/x "T"'


def test_frontmatter_is_masked():
    m = mask("---\ntitle: Hi\ntags: [a, b]\n---\n\nBody https://x.io\n")
    assert m.text == "[[BLOCK_1]]\n\nBody [[URL_1]]\n"
    assert m.originals["[[BLOCK_1]]"] == "---\ntitle: Hi\ntags: [a, b]\n---"


def test_html_comment_hides_the_code_inside_it():
    m = mask("Hello <!-- note: `x` --> world <br/> and <b>bold</b>.")
    assert m.text == "Hello [[HTML_1]] world [[HTML_2]] and [[HTML_3]]bold[[HTML_4]]."
    assert m.originals["[[HTML_1]]"] == "<!-- note: `x` -->"
    assert "[[CODE_1]]" not in m.text


def test_text_that_looks_like_a_placeholder_is_refused():
    with pytest.raises(MaskingError):
        mask("literal [[CODE_1]] in the source")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no markup at all",
        "# Title\n\nText with `code` and https://a.b/c?d=1&e=2 and **bold**.\n",
        "```py\nprint(1)\n```\n\n~~~\nplain\n~~~\n",
        '<!-- c `x` -->\n<div class="a">hi</div>\n',
        "---\na: 1\n---\nbody\n",
        '[a][1] and ![alt](pic.png)\n\n[1]: https://example.com "Title"\n',
        "unclosed ``` fence\n",
        "```\nopen forever\n",
        "a\r\n\r\n```\r\ncode\r\n```\r\n",
        "tilde ~~~ not a fence\n",
        "Ví dụ: `kubectl` và https://example.com/đường-dẫn.\n",
    ],
)
def test_mask_then_unmask_is_lossless(text):
    m = mask(text)
    assert unmask(m.text, m.originals) == text
