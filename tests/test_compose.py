import pytest

from nightshift.compose import (
    LIMITS,
    OUTLINE_TOKENS,
    VOICES_TOKENS,
    BriefError,
    build_inputs,
    check_post,
    clean_output,
    ensure_frame,
    ensure_points,
    flag_pairs,
    hide_terms,
    normalize_brief,
    parse_terms,
    parse_voices,
    restore,
    tidy_post,
    trim_cut_off,
    uncovered_points,
    unsupported_clauses,
    unsupported_spans,
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
    assert write.num_predict == 2600 and write.target_words == 600
    outline = build_inputs(record(), "outline")
    assert "Mở bài" in outline.system and "Kết bài" in outline.system
    assert "Vietnamese" in outline.system and "Choose the sections yourself" in outline.system
    assert outline.num_predict == OUTLINE_TOKENS
    with_points = build_inputs(record(points="ý một\ný hai"), "outline")
    assert "exactly one section for each" in with_points.system
    assert "lists 2 key points" in with_points.system
    assert "Choose the sections yourself" not in with_points.system
    english = build_inputs(record(language="en"), "outline")
    assert "Opening" in english.system and "Closing" in english.system
    assert "follow its order" in write.system
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


# --- the outline frame --------------------------------------------------------------------


def test_a_missing_opening_and_closing_are_added_neutrally():
    framed = ensure_frame("- Một: ý\n- Hai: ý", "vi")
    lines = framed.splitlines()
    assert lines[0].startswith("- Mở bài:") and lines[-1].startswith("- Kết bài:")
    assert lines[1:3] == ["- Một: ý", "- Hai: ý"]
    english = ensure_frame("- One: idea", "en").splitlines()
    assert english[0].startswith("- Opening:") and english[-1].startswith("- Closing:")


def test_an_outline_that_already_has_them_is_left_alone():
    outline = "- Mở bài: a\n- Giữa: b\n- Kết bài: c"
    assert ensure_frame(outline, "vi") == outline
    assert (
        ensure_frame("- **Mở bài**: a\n- b: x\n* kết bài: c", "vi")
        == "- **Mở bài**: a\n- b: x\n* kết bài: c"
    )


def test_only_the_missing_end_is_added_and_an_empty_outline_gets_both():
    only_open = ensure_frame("- Mở bài: a\n- Giữa: b", "vi").splitlines()
    assert (
        only_open[0] == "- Mở bài: a"
        and only_open[-1].startswith("- Kết bài:")
        and len(only_open) == 3
    )
    only_close = ensure_frame("- Giữa: b\n- Kết bài: c", "vi").splitlines()
    assert only_close[0].startswith("- Mở bài:") and only_close[-1] == "- Kết bài: c"
    assert len(ensure_frame("", "vi").splitlines()) == 2


def test_the_frame_says_nothing_about_the_material():
    for language in ("vi", "en"):
        framed = ensure_frame("- Một: ý", language)
        assert not any(ch.isdigit() for ch in framed)


# --- the stricter post prompt ---------------------------------------------------------------


def test_the_post_prompt_forbids_copying_repeating_and_unsupported_claims():
    system = build_inputs(record(mode="close"), "write").system
    for phrase in (
        "never copy a sentence",
        "never repeat a point",
        "state no benefit, result, opinion or general claim",
        "Cover each part of the outline once, in order",
        "Write for the audience named in the brief",
    ):
        assert phrase in system


# --- sentences that say more than the material -----------------------------------------------

SOURCE = (
    "Mất 2 tháng để làm quen với Claude Code. Tôi triển khai dự án lên môi trường production "
    "và tự phát triển công cụ viết blog cho portfolio. Tâm đắc nhất là portfolio trở nên đặc sắc."
)


def test_a_claim_the_material_never_makes_is_flagged():
    post = "Quá trình này giúp tôi tiết kiệm thời gian và tăng năng suất làm việc."
    assert unsupported_clauses(post, SOURCE) == [
        "Quá trình này giúp tôi tiết kiệm thời gian và tăng năng suất làm việc"
    ]


def test_sentences_built_from_the_material_are_not_flagged():
    post = (
        "# Tiêu đề\n\nTôi triển khai dự án lên môi trường production. "
        "Tôi tự phát triển công cụ viết blog cho portfolio, và portfolio trở nên đặc sắc."
    )
    assert unsupported_clauses(post, SOURCE) == []


def test_code_links_headings_and_short_clauses_are_ignored():
    post = (
        "## Tiêu đề hoàn toàn mới lạ chưa từng có\n"
        "Chạy `kubectl apply manifest riêng biệt` và xem https://example.com/trang-moi-hoan-toan.\n"
        "Rất tốt. Quá hay."
    )
    assert unsupported_clauses(post, SOURCE) == []


def test_the_worst_clauses_come_first_and_the_list_is_capped():
    post = "\n".join(
        [
            "Ứng dụng này mang lại hiệu quả vượt trội cho doanh nghiệp lớn.",
            "Tôi triển khai dự án với nhiều bất ngờ thú vị đáng nhớ.",
        ]
        + [f"Mục số {i} nói về chuyện hoàn toàn khác biệt không liên quan gì." for i in range(10)]
    )
    found = unsupported_clauses(post, SOURCE, limit=3)
    assert len(found) == 3 and all("Tôi triển khai" not in clause for clause in found)


def test_the_check_is_part_of_check_post_and_words_from_the_outline_do_not_count_as_known():
    rec = record(material=SOURCE)
    rec["outline"] = "- Lợi ích: giúp tôi tiết kiệm thời gian và tăng năng suất làm việc"
    inputs = build_inputs(rec, "write")
    post = "# T\n\nQuá trình này giúp tôi tiết kiệm thời gian và tăng năng suất làm việc."
    texts = [i["text"] for i in check_post(post, inputs, 0, 0)]
    assert any(t.startswith("These parts may say more than your material does") for t in texts)
    clean = "# T\n\nTôi triển khai dự án lên môi trường production."
    texts = [i["text"] for i in check_post(clean, inputs, 0, 0)]
    assert not any("may say more" in t for t in texts)


# --- where the flagged parts are -------------------------------------------------------------

INVENTED = "Quá trình này giúp tôi tiết kiệm thời gian"


def test_spans_point_at_exactly_the_flagged_text_even_around_code_and_links():
    post = (
        "# Tiêu đề\n\nTôi triển khai dự án lên môi trường production.\n"
        f"Chạy `kubectl apply riêng` rồi xem https://example.com/x: {INVENTED}, tăng năng suất.\n"
    )
    spans = unsupported_spans(post, SOURCE)
    flagged = [post[s:e] for s, e, _ in spans]
    assert INVENTED in flagged
    assert all(text and text == text.strip() for text in flagged)
    assert not any("kubectl" in text or "example.com" in text for text in flagged)
    assert [s for s, _, _ in spans] == sorted(s for s, _, _ in spans)


def test_spans_survive_windows_line_endings_and_blank_lines():
    post = (
        f"Tôi triển khai dự án lên môi trường production.\r\n\r\n{INVENTED} và tăng năng suất.\r\n"
    )
    spans = unsupported_spans(post, SOURCE)
    assert len(spans) == 1
    start, end, _ = spans[0]
    assert post[start:end].startswith("Quá trình này") and "\r" not in post[start:end]


def test_flag_pairs_is_what_gets_stored_next_to_the_post():
    post = f"# T\n\n{INVENTED} và tăng năng suất."
    pairs = flag_pairs(post, SOURCE)
    assert pairs and all(isinstance(p, list) and len(p) == 2 for p in pairs)
    assert post[pairs[0][0] : pairs[0][1]].startswith("Quá trình này")


def test_check_post_counts_the_flagged_parts_and_mentions_the_highlight():
    inputs = build_inputs(record(material=SOURCE), "write")
    lines = [f"Câu số {i} nói về chuyện hoàn toàn khác biệt không liên quan gì" for i in range(6)]
    post = "# T\n\n" + ". ".join(lines) + "."
    items = check_post(post, inputs, 0, 0)
    text = next(i["text"] for i in items if "may say more" in i["text"])
    assert "(highlighted in the text)" in text and "and 3 more" in text


# --- key points ---------------------------------------------------------------------------------

POINTS = (
    "- Ban đầu chưa biết đến sự lợi hại của Claude\n"
    "- Sau khi biết rồi thì đã liều mạng bỏ ra 600k mua tài khoản Claude Pro\n"
    "* Thử ứng dụng Claude Code vào công việc, trong đó có triển khai dự án\n"
    "1. Tự phát triển công cụ viết blog và deploy lên portfolio\n"
)
OUTLINE_TEXT = (
    "- Mở bài: Tôi từng chỉ biết ChatGPT, nay làm việc cùng Claude Pro.\n"
    "- Ứng dụng: Tôi thử ứng dụng Claude Code vào công việc, từ triển khai dự án.\n"
    "- Công cụ cá nhân: Tôi tự phát triển công cụ viết blog và deploy lên portfolio.\n"
)


def test_points_the_outline_leaves_out_are_listed_with_their_original_wording():
    assert uncovered_points(POINTS, OUTLINE_TEXT) == [
        "Ban đầu chưa biết đến sự lợi hại của Claude",
        "Sau khi biết rồi thì đã liều mạng bỏ ra 600k mua tài khoản Claude Pro",
    ]


def test_a_number_in_a_point_must_appear_in_the_text():
    point = "- Bỏ ra 600k mua tài khoản"
    assert uncovered_points(point, "- Mua tài khoản Pro, bỏ ra tiền") == [
        "Bỏ ra 600k mua tài khoản"
    ]
    assert uncovered_points(point, "- Bỏ ra 600k mua tài khoản Pro") == []


def test_points_are_covered_by_words_in_any_order_and_list_markers_are_ignored():
    assert (
        uncovered_points("- Triển khai dự án lên production", "dự án: triển khai production") == []
    )
    assert uncovered_points("1) Một\n2) Hai ba", "hai ba") == []


def test_no_points_or_blank_lines_mean_nothing_is_missing():
    assert uncovered_points("", OUTLINE_TEXT) == []
    assert uncovered_points("\n  \n- \n", OUTLINE_TEXT) == []


def test_the_post_check_names_the_key_points_the_post_dropped():
    inputs = build_inputs(record(points=POINTS, material=SOURCE), "write")
    post = "# T\n\nTôi thử ứng dụng Claude Code vào công việc và tự phát triển công cụ viết blog."
    texts = [i["text"] for i in check_post(post, inputs, 0, 0)]
    cover = next(t for t in texts if t.startswith("Key points the post does not seem to cover"))
    assert "600k" in cover and "Ban đầu chưa biết" in cover and "Thử ứng dụng" not in cover


# --- mode, voice, story and outline points -----------------------------------------------------

VOICES_TEXT = (
    "Tôi từng nghĩ mình chỉ cần một công cụ hỏi đáp, cho đến khi gặp Claude trong công việc.\n"
    "---\n"
    "1. Buổi tối hôm đó tôi ngồi trước màn hình, gõ thử câu hỏi đầu tiên cho Claude Code.\n"
    "---\n"
    "**C)** Hóa ra tôi mất sáu trăm nghìn để biết mình đã bỏ lỡ cả một thế giới, haha vui thật.\n"
)


def test_mode_and_voice_have_defaults_and_are_validated():
    b = normalize_brief({"topic": "x"})
    assert (b["mode"], b["voice_sentences"], b["voice_humor"]) == ("expand", "mixed", "light")
    assert b["story"] == "" and b["voice_sample"] == "" and b["voice_address"] == ""
    for field in ("mode", "voice_sentences", "voice_humor"):
        with pytest.raises(BriefError):
            normalize_brief({"topic": "x", field: "nope"})
    with pytest.raises(BriefError, match="voice sample"):
        normalize_brief({"topic": "x", "voice_sample": "a" * (LIMITS["voice_sample"] + 1)})


def test_expand_lets_the_model_reflect_and_close_keeps_the_prohibitions():
    expand = build_inputs(record(mode="expand"), "write").system
    close = build_inputs(record(mode="close"), "write").system
    assert "Go beyond the notes" in expand and "state no benefit" not in expand
    assert "Do not add new events, results, numbers, names or products" in expand
    assert "state no benefit" in close and "Go beyond the notes" not in close


def test_the_voice_rules_reach_the_write_prompt_and_only_there():
    voice = {
        "voice_address": "mình - các bạn",
        "voice_sentences": "short",
        "voice_humor": "playful",
        "voice_avoid": "hành trình, không thể phủ nhận",
    }
    system = build_inputs(record(**voice), "write").system
    assert "'mình - các bạn'" in system and "Prefer short sentences" in system
    assert "Playful humor" in system and "hành trình, không thể phủ nhận" in system
    assert "Voice:" not in build_inputs(record(**voice), "outline").system


def test_the_story_counts_as_fact_but_the_voice_sample_does_not():
    rec = record(story="Tối đó tôi đã hồi hộp.", voice_sample="Mẫu giọng riêng biệt.")
    write = build_inputs(rec, "write")
    assert "Tối đó tôi đã hồi hộp." in write.prompt and "Tối đó tôi đã hồi hộp." in write.source
    assert "Mẫu giọng riêng biệt." in write.prompt and "Mẫu giọng riêng" not in write.source
    assert "Never repeat its sentences" in write.prompt
    assert "Mẫu giọng riêng biệt." not in build_inputs(rec, "outline").prompt


def test_the_outline_asks_for_one_section_per_key_point_in_order():
    inputs = build_inputs(record(points="- ý một\n2) ý hai\n\n* ý ba"), "outline")
    assert "lists 3 key points" in inputs.system and "between them come 3 sections" in inputs.system
    assert "1. ý một\n2. ý hai\n3. ý ba" in inputs.prompt


def test_points_the_outline_still_leaves_out_are_added_before_the_closing():
    points = "- ý một\n- bỏ ra 600k mua tài khoản"
    outline = "- Mở bài: chào\n- Ý một: nói về ý một\n- Kết bài: tạm biệt"
    fixed = ensure_points(outline, points, "vi")
    assert fixed.splitlines() == [
        "- Mở bài: chào",
        "- Ý một: nói về ý một",
        "- bỏ ra 600k mua tài khoản",
        "- Kết bài: tạm biệt",
    ]
    assert ensure_points(fixed, points, "vi") == fixed
    assert ensure_points("- Mở bài: chào", points, "vi").splitlines()[-1].startswith("- bỏ ra 600k")
    assert ensure_points(outline, "", "vi") == outline


def test_voices_are_split_on_their_dividers_and_stripped_of_numbering():
    voices = parse_voices(VOICES_TEXT)
    assert len(voices) == 3
    assert voices[0].startswith("Tôi từng nghĩ") and voices[1].startswith("Buổi tối")
    assert voices[2].startswith("Hóa ra")
    assert parse_voices("quá ngắn\n---\nvẫn ngắn") == []
    inline = "Đoạn thứ nhất có đủ tám từ để dùng. --- Đoạn thứ hai cũng có đủ tám từ nhé. --- "
    inline += "Đoạn thứ ba thì viết thế này cho đủ."
    assert len(parse_voices(inline)) == 3
    assert len(parse_voices("\n---\n".join([VOICES_TEXT] * 3))) == 3


def test_the_voices_prompt_asks_for_three_versions_and_trims_the_material():
    inputs = build_inputs(record(material="x " * 3000, story="chuyện của tôi"), "voices")
    assert inputs.num_predict == VOICES_TOKENS
    assert "three times" in inputs.system and "line containing only ---" in inputs.system
    assert "chuyện của tôi" in inputs.prompt and len(inputs.prompt) < 2500


def test_expand_mode_flags_only_clauses_that_are_almost_all_new():
    source = "alpha beta gamma delta epsilon"
    half = "alpha beta unknownone unknowntwo."
    assert flag_pairs(half, source) != [] and flag_pairs(half, source, True) == []
    brand_new = "pretty unknownone unknowntwo unknownthree unknownfour."
    assert flag_pairs(brand_new, source, True) != []
    expand = check_post(half, build_inputs(record(mode="expand", material=source), "write"), 0, 0)
    close = check_post(half, build_inputs(record(mode="close", material=source), "write"), 0, 0)
    assert not any("may say more" in i["text"] for i in expand)
    assert any("may say more" in i["text"] for i in close)


def test_a_missing_point_goes_where_the_author_put_it_not_at_the_end():
    points = "- chuyện đầu tiên\n- ý giữa\n- chuyện cuối cùng"
    outline = "- Mở bài: chào\n- Ý giữa: nói về ý giữa\n- Kết bài: tạm biệt"
    fixed = ensure_points(outline, points, "vi").splitlines()
    assert fixed == [
        "- Mở bài: chào",
        "- chuyện đầu tiên",
        "- Ý giữa: nói về ý giữa",
        "- chuyện cuối cùng",
        "- Kết bài: tạm biệt",
    ]


def test_headings_keep_only_the_label_and_the_frame_headings_go():
    post = (
        "# Tiêu đề\n\n## Mở bài: Tôi đã gặp Claude như thế nào\n\nĐoạn mở đầu có đủ sáu từ.\n\n"
        "## Ứng dụng vào công việc: Tôi thử áp dụng nó vào công việc\n\nĐoạn giữa có đủ sáu từ.\n\n"
        "## Kết bài: Tóm lại điều rút ra\n\nĐoạn cuối có đủ sáu từ.\n"
    )
    fixed, removed = tidy_post(post, "vi")
    assert removed == 0
    assert fixed.splitlines()[0] == "# Tiêu đề"
    assert "## Ứng dụng vào công việc\n" in fixed
    assert "Mở bài" not in fixed and "Kết bài" not in fixed
    assert fixed.count("##") == 1 and "\n\n\n" not in fixed
    assert tidy_post("## Closing: x\n\nMột đoạn văn có đủ sáu từ.", "en")[0].startswith("Một")


def test_a_sentence_repeated_word_for_word_is_removed_but_code_and_short_ones_stay():
    post = (
        "Mình nghĩ chắc sẽ gỡ nhầm file. Nhưng nó hỏi lại mình. Ừ.\n\n"
        "Mình nghĩ chắc sẽ gỡ nhầm file. Nhưng nó hỏi lại mình. Ừ.\n\n"
        "```\nmình nghĩ chắc sẽ gỡ nhầm file.\n```\n"
    )
    fixed, removed = tidy_post(post, "vi")
    assert removed == 2
    assert fixed.count("Mình nghĩ chắc sẽ gỡ nhầm file.") == 1
    assert fixed.count("Nhưng nó hỏi lại mình.") == 1 and fixed.count("Ừ.") == 2
    assert "```\nmình nghĩ chắc sẽ gỡ nhầm file.\n```" in fixed


def test_a_sentence_cut_off_at_the_end_is_dropped_with_its_empty_heading():
    whole = "# T\n\nMột đoạn xong rồi. Và câu này dở"
    text, cut = trim_cut_off(whole)
    assert cut and text.endswith("Một đoạn xong rồi.\n")
    text, cut = trim_cut_off(
        "# T\n\nĐoạn một xong.\n\n## Cuối cùng\n\nTôi đã tự phát triển công cụ"
    )
    assert cut and text.rstrip().endswith("Đoạn một xong.") and "Cuối cùng" not in text
    text, cut = trim_cut_off("# T\n\nĐoạn một xong.\n")
    assert not cut and text == "# T\n\nĐoạn một xong.\n"
    assert not trim_cut_off("```\ncode\n```")[1]
