import pytest

from nightshift.compose import (
    LIMITS,
    BriefError,
    build_inputs,
    check_post,
    clean_output,
    hide_terms,
    normalize_brief,
    parse_terms,
    restore,
)


def brief(**changes):
    form = {"topic": "Gỡ Rancher khỏi k3s", "language": "vi", **changes}
    return normalize_brief(form)


def record(**changes):
    return {"brief": brief(**changes), "outline": ""}


# --- the brief ----------------------------------------------------------------------------


def test_defaults_and_trimming():
    b = normalize_brief({"topic": "  Chủ đề  ", "material": "a\r\nb"})
    assert b["topic"] == "Chủ đề" and b["material"] == "a\nb"
    assert (b["tone"], b["length"], b["language"]) == ("conversational", "short", "vi")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"topic": "  "}, "about"),
        ({"tone": "angry"}, "tone"),
        ({"length": "epic"}, "length"),
        ({"language": "fr"}, "language"),
        ({"material": "x" * (LIMITS["material"] + 1)}, "material is longer"),
        ({"avoid": "x" * 81}, "term to hide"),
    ],
)
def test_bad_briefs_are_refused_with_a_clear_message(changes, message):
    with pytest.raises(BriefError, match=message):
        brief(**changes)


# --- terms to hide ------------------------------------------------------------------------


def test_terms_are_parsed_deduplicated_and_longest_first():
    terms = parse_terms("EVN\nevn => một đơn vị điện lực\n\nCông ty EVN Hà Nội => một công ty\n")
    assert [t for t, _ in terms] == ["Công ty EVN Hà Nội", "EVN"]
    assert dict(terms)["EVN"] == ""
    with pytest.raises(BriefError):
        parse_terms("\n".join(f"t{i}" for i in range(51)))


def test_hiding_matches_whole_words_in_any_case():
    terms = [("EVN", "")]
    text, tokens = hide_terms("Tôi đến evn và EVNX và (EVN).", terms)
    assert text == "Tôi đến [[HIDE_1]] và EVNX và ([[HIDE_1]])."
    assert tokens == {"[[HIDE_1]]": 0}


# --- what the model sees ------------------------------------------------------------------


MATERIAL = (
    "Hôm nay tôi gặp đối tác ở EVN, số 11 Cửa Bắc.\n\n"
    "Chạy `helm uninstall rancher` và xem [tài liệu](https://example.com/docs).\n\n"
    "```bash\nkubectl get pods -n cattle-system\n```\n"
)


def test_the_model_never_sees_hidden_terms_code_or_links():
    inputs = build_inputs(record(material=MATERIAL, avoid="EVN\nCửa Bắc"), "write")
    for secret in ("EVN", "Cửa Bắc", "helm uninstall", "example.com", "kubectl"):
        assert secret not in inputs.prompt
    assert "[[HIDE_1]]" in inputs.prompt and "[[CODE_1]]" in inputs.prompt
    assert "Gỡ Rancher khỏi k3s" in inputs.prompt


def test_prompts_follow_the_brief():
    write = build_inputs(record(language="en", tone="technical", length="medium"), "write")
    assert "English" in write.system and "technical and concise" in write.system
    assert "[TODO: ...]" in write.system and "'# ' title" in write.system
    assert write.num_predict == 1800 and write.target_words == 600
    outline = build_inputs(record(), "outline")
    assert "4 to 6 bullet points" in outline.system and "Vietnamese" in outline.system
    assert outline.num_predict == 400
    assert "Length:" not in outline.prompt


def test_the_outline_is_part_of_the_write_prompt_only_when_there_is_one():
    plain = build_inputs(record(), "write")
    assert "Outline to follow" not in plain.prompt
    rec = record()
    rec["outline"] = "- ý một\n- ý hai"
    assert "Outline to follow:\n- ý một" in build_inputs(rec, "write").prompt


def test_text_that_looks_like_a_placeholder_is_refused():
    with pytest.raises(BriefError):
        build_inputs(record(material="gõ [[CODE_1]] vào"), "write")


# --- what comes back ----------------------------------------------------------------------


def test_restore_puts_code_back_and_swaps_hidden_terms_for_their_stand_ins():
    inputs = build_inputs(record(material=MATERIAL, avoid="EVN => một đơn vị\nCửa Bắc"), "write")
    text, redacted, invented = restore(
        "# Tiêu đề\n\nTôi gặp [[HIDE_2]] ở [[HIDE_1]].\n\nChạy [[CODE_1]] và [[URL_1]].", inputs
    )
    assert "Tôi gặp một đơn vị ở [đã ẩn]." in text
    assert "`helm uninstall rancher`" in text and "https://example.com/docs" in text
    assert (redacted, invented) == (0, 0)


def test_a_term_the_model_wrote_by_itself_is_redacted_and_counted():
    inputs = build_inputs(record(avoid="EVN"), "write")
    text, redacted, _ = restore("Tôi đến EVN, rồi evn nữa.", inputs)
    assert "EVN" not in text and "evn" not in text.lower().replace("[đã ẩn]", "")
    assert redacted == 2


def test_made_up_tokens_are_removed_and_counted():
    inputs = build_inputs(record(), "write")
    text, _, invented = restore("Dùng [[CODE_9]] và [[HIDE_3]] nhé.", inputs)
    assert "[[" not in text and invented == 2


def test_a_wrapping_code_fence_is_dropped_but_inner_fences_stay():
    assert clean_output("```markdown\n# Tiêu đề\n\nNội dung\n```") == "# Tiêu đề\n\nNội dung"
    inner = "# T\n\n```bash\nls\n```\n\ntext"
    assert clean_output(inner) == inner


# --- checks -------------------------------------------------------------------------------


def levels(items):
    return [i["level"] for i in items]


def test_new_numbers_are_flagged_but_small_ones_and_known_ones_are_not():
    inputs = build_inputs(record(material="Giảm 1.2 GB RAM, 11 Cửa Bắc."), "write")
    post = "# T\n\nGiảm 1.2 GB và 11 Cửa Bắc, khoảng 3 lần, tăng 450 MB."
    items = check_post(post, inputs, 0, 0)
    numbers = next(i["text"] for i in items if i["text"].startswith("Numbers in the post"))
    assert "450" in numbers and "1.2" not in numbers and "11" not in numbers


def test_private_looking_strings_are_flagged():
    inputs = build_inputs(record(), "write")
    post = (
        "# T\n\nTruy cập 192.168.50.10 hoặc nightshift.kienhomeserver.lab, mail me@example.com, "
        "khoá 0123456789abcdef0123456789abcdef01."
    )
    text = " ".join(i["text"] for i in check_post(post, inputs, 0, 0))
    for item in (
        "192.168.50.10",
        "nightshift.kienhomeserver.lab",
        "me@example.com",
        "0123456789abcdef",
    ):
        assert item in text


def test_missing_material_markers_and_odd_length_are_reported_and_warnings_come_first():
    inputs = build_inputs(record(), "write")
    items = check_post("Không có tiêu đề [cần bổ sung: ngày] [TODO: x]", inputs, 1, 2)
    assert levels(items) == sorted(levels(items), key=lambda x: x != "warn")
    text = " ".join(i["text"] for i in items)
    assert "2 place(s) marked as missing" in text and "no title line" in text
    assert (
        "hidden term(s) itself" in text and "made up" in text and "the target was about 300" in text
    )


def test_a_clean_post_has_no_warnings():
    material = "Tôi gỡ Rancher khỏi cụm k3s. " * 10
    inputs = build_inputs(record(material=material), "write")
    post = "# Gỡ Rancher\n\n" + " ".join(["Tôi gỡ Rancher khỏi cụm k3s."] * 55)
    assert "warn" not in levels(check_post(post, inputs, 0, 0))
