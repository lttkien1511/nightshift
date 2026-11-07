import io
import json
import threading
import time
import urllib.error

import pytest
from fastapi.testclient import TestClient

from nightshift.compose import OUTLINE_TOKENS, VOICES_TOKENS, ensure_frame, pieces
from nightshift.composer import Composer
from nightshift.gate import Gate
from nightshift.runner import Runner
from nightshift.storage import Store
from nightshift.web import composer_from_env, create_app

FORM = {
    "topic": "Gỡ Rancher khỏi k3s",
    "audience": "người mới học Kubernetes",
    "tone": "technical",
    "length": "short",
    "language": "vi",
    "points": "vì sao gỡ\ncách gỡ",
    "material": "Tôi gặp đối tác ở EVN.\n\nChạy `helm uninstall rancher` rồi xem.",
    "avoid": "EVN",
}
OUTLINE = "- Vì sao gỡ\n- Cách gỡ\n- Kết quả"
VOICES = (
    "Một đoạn mở bài theo giọng tâm sự với bạn thân.\n---\n"
    "Buổi tối hôm đó tôi ngồi một mình trước màn hình máy tính.\n---\n"
    "Hóa ra mọi chuyện buồn cười hơn tôi tưởng rất nhiều."
)
POST = "# Gỡ Rancher\n\nTôi gặp [[HIDE_1]]. Chạy [[CODE_1]] rồi xem, giảm 450 MB."


class Script:
    """A model that answers outline and post requests, and can wait at a gate."""

    def __init__(self):
        self.hold = threading.Event()
        self.hold.set()
        self.prompts = []

    def __call__(self, system, prompt, num_predict, on_text):
        self.prompts.append(prompt)
        text = {OUTLINE_TOKENS: OUTLINE, VOICES_TOKENS: VOICES}.get(num_predict, POST)
        out = []
        for index, piece in enumerate([text[: len(text) // 2], text[len(text) // 2 :]]):
            if index == 1:
                self.hold.wait(5)
            out.append(piece)
            if on_text(piece) is False:
                break
        return "".join(out)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path, max_queue=3)
    s.ensure()
    return s


@pytest.fixture
def script():
    return Script()


@pytest.fixture
def composer(store, script):
    return Composer(store, lambda model: script, ["m1", "m2"], Gate())


@pytest.fixture
def client(store, composer):
    runner = Runner(store, lambda m: lambda b: b, ["m1"], gate=composer.gate)
    return TestClient(create_app(store, runner, composer))


def create(client, action="save", **changes):
    return client.post("/posts", data={**FORM, "action": action, **changes}, follow_redirects=False)


# --- the form -----------------------------------------------------------------------------


def test_sidebar_links_to_new_post_and_lists_posts(client, store):
    page = client.get("/").text
    assert 'href="/posts/new"' in page and "No blog posts yet." in page
    create(client)
    page = client.get("/").text
    assert 'href="/posts/go-rancher-khoi-k3s"' in page


def test_the_brief_form_has_every_field_and_both_ways_to_start(client):
    page = client.get("/posts/new").text
    for name in (
        "topic",
        "audience",
        "tone",
        "tone_note",
        "length",
        "language",
        "points",
        "material",
        "avoid",
    ):
        assert f'name="{name}"' in page
    assert (
        'value="outline">Suggest an outline' in page and 'value="write">Write the post now' in page
    )
    assert '<option value="vi" selected>Vietnamese</option>' in page
    assert "term =&gt; stand-in" in page


def test_without_a_model_the_form_only_saves(store):
    quiet = TestClient(create_app(store))
    page = quiet.get("/posts/new").text
    assert "Suggest an outline" not in page and "Save the brief" in page


def test_creating_a_post_saves_it_and_opens_it(client, store):
    response = create(client)
    assert response.status_code == 303
    slug = response.headers["location"].rsplit("/", 1)[1]
    record = store.post(slug)
    assert record["stage"] == "brief" and record["brief"]["topic"] == FORM["topic"]
    page = client.get(f"/posts/{slug}").text
    assert "Suggest an outline" in page and "Skip it and write the post now" in page


def test_a_bad_brief_comes_back_with_the_message_and_what_was_typed(client, store):
    response = client.post("/posts", data={**FORM, "topic": "", "points": "giữ <lại>"})
    assert response.status_code == 400
    assert "say what the post is about" in response.text and "giữ &lt;lại&gt;" in response.text
    assert store.posts() == []
    assert client.post("/posts", data={**FORM, "tone": "angry"}).status_code == 400


def test_titles_are_escaped_everywhere(client):
    slug = create(client, topic="<script>alert(1)</script>").headers["location"].rsplit("/", 1)[1]
    for path in ("/", f"/posts/{slug}", f"/posts/{slug}/edit"):
        assert "<script>alert" not in client.get(path).text


def test_there_is_a_limit_on_posts(client):
    for index in range(20):
        assert create(client, topic=f"bài {index}").status_code == 303
    assert create(client, topic="thêm").status_code == 429


# --- the steps ----------------------------------------------------------------------------


def slug_of(response):
    return response.headers["location"].rsplit("/", 1)[1]


def test_suggest_an_outline_then_edit_it_then_write(client, store, composer, script):
    slug = slug_of(create(client, action="outline"))
    composer.wait()
    assert store.post(slug)["outline"] == ensure_frame(OUTLINE, "vi")
    assert store.post(slug)["stage"] == "outline"
    page = client.get(f"/posts/{slug}").text
    assert '<textarea name="outline" class="outline"' in page and "- Vì sao gỡ" in page

    edited = "- Chỉ một ý\n- Hai ý"
    response = client.post(
        f"/posts/{slug}/write", data={"outline": edited, "model": "m2"}, follow_redirects=False
    )
    assert response.status_code == 303
    composer.wait()
    record = store.post(slug)
    assert record["outline"] == edited and record["stage"] == "done" and record["model"] == "m2"
    assert "Outline to follow:\n- Chỉ một ý" in script.prompts[-1]
    assert "EVN" not in script.prompts[-1] and "helm uninstall" not in script.prompts[-1]


def test_the_written_post_shows_text_checks_and_ways_out(client, store, composer):
    slug = slug_of(create(client, action="write"))
    composer.wait()
    page = client.get(f"/posts/{slug}").text
    assert "Tôi gặp [đã ẩn]." in page and "`helm uninstall rancher`" in page
    assert "Numbers in the post that the material does not contain: 450" in page
    assert f'href="/posts/{slug}/post.md">Download .md' in page
    assert "Save as a draft to polish" in page and "Write it again" in page

    download = client.get(f"/posts/{slug}/post.md")
    assert download.headers["content-type"].startswith("text/markdown")
    assert f'filename="{slug}.md"' in download.headers["content-disposition"]
    assert download.text.startswith("# Gỡ Rancher") and "[đã ẩn]" in download.text


def test_downloading_or_saving_a_post_that_is_not_written_yet_is_refused(client):
    slug = slug_of(create(client))
    assert client.get(f"/posts/{slug}/post.md").status_code == 404
    assert client.post(f"/posts/{slug}/to-draft").status_code == 400


def test_save_as_draft_makes_a_unique_draft_for_the_edit_pass(client, store, composer):
    slug = slug_of(create(client, action="write"))
    composer.wait()
    first = client.post(f"/posts/{slug}/to-draft", follow_redirects=False)
    second = client.post(f"/posts/{slug}/to-draft", follow_redirects=False)
    assert first.headers["location"] != second.headers["location"]
    name = first.headers["location"].rsplit("/", 1)[1]
    assert "[đã ẩn]" in store.read(name) and store.read(name).endswith("\n")


def test_progress_and_live_text_while_writing(client, composer, script):
    script.hold.clear()
    slug = slug_of(create(client, action="write"))
    for _ in range(100):
        if composer.state and composer.state.chars:
            break
        threading.Event().wait(0.02)
    page = client.get(f"/posts/{slug}").text
    assert "Writing the post with m1" in page and '<pre class="pstream">' in page
    assert '<meta http-equiv="refresh" content="3">' in page and f"/posts/{slug}/cancel" in page
    assert '<i class="dot writing"' in page and "writing go-rancher" in page
    assert "Suggest an outline" not in page
    state = client.get(f"/posts/{slug}/run.json").json()
    assert state["state"] == "running" and state["kind"] == "write" and state["chars"] > 0
    script.hold.set()
    composer.wait()
    done = client.get(f"/posts/{slug}").text
    assert "http-equiv" not in done and "pstream" not in done


def test_cancel_stops_the_writing(client, store, composer, script):
    script.hold.clear()
    slug = slug_of(create(client, action="write"))
    client.post(f"/posts/{slug}/cancel")
    script.hold.set()
    composer.wait()
    assert composer.state.state == "cancelled" and store.post(slug)["stage"] == "brief"
    assert "The last attempt was cancelled" in client.get(f"/posts/{slug}").text


def test_another_job_is_refused_while_the_model_is_busy(client, store, composer, script):
    script.hold.clear()
    first = slug_of(create(client, action="write"))
    second = slug_of(create(client, topic="Bài khác"))
    assert client.post(f"/posts/{second}/outline").status_code == 409
    assert "Another blog post is being written" in client.get(f"/posts/{second}").text
    store.save("draft", "Đoạn ngắn.\n")
    busy = client.post("/drafts/draft/run")
    assert busy.status_code == 409 and "writing" in busy.text
    client.post(f"/posts/{first}/cancel")
    script.hold.set()
    composer.wait()


def test_a_failed_attempt_is_explained(store):
    def down(system, prompt, num_predict, on_text):
        raise RuntimeError("ollama is unreachable")

    composer = Composer(store, lambda m: down, ["m1"], Gate())
    client = TestClient(create_app(store, None, composer))
    slug = slug_of(create(client, action="write"))
    composer.wait()
    page = client.get(f"/posts/{slug}").text
    assert "The last attempt failed" in page and "unreachable" in page


# --- editing the brief, deleting ----------------------------------------------------------


def test_edit_the_brief(client, store):
    slug = slug_of(create(client))
    page = client.get(f"/posts/{slug}/edit").text
    assert 'value="Gỡ Rancher khỏi k3s"' in page and "Save the brief" in page
    assert '<option value="technical" selected>' in page
    response = client.post(
        f"/posts/{slug}/edit", data={**FORM, "topic": "Tiêu đề mới"}, follow_redirects=False
    )
    assert response.status_code == 303 and response.headers["location"] == f"/posts/{slug}"
    assert (
        store.post(slug)["title"] == "Tiêu đề mới"
        and store.post(slug)["brief"]["topic"] == "Tiêu đề mới"
    )
    bad = client.post(f"/posts/{slug}/edit", data={**FORM, "topic": " "})
    assert bad.status_code == 400 and "say what the post is about" in bad.text
    assert store.post(slug)["title"] == "Tiêu đề mới"


def test_a_post_cannot_be_edited_or_deleted_while_it_is_being_written(
    client, store, composer, script
):
    script.hold.clear()
    slug = slug_of(create(client, action="write"))
    assert client.post(f"/posts/{slug}/edit", data=FORM).status_code == 409
    assert client.post(f"/posts/{slug}/delete").status_code == 409
    client.post(f"/posts/{slug}/cancel")
    script.hold.set()
    composer.wait()
    response = client.post(f"/posts/{slug}/delete", follow_redirects=False)
    assert response.status_code == 303 and store.post(slug) is None


def test_unknown_posts_and_odd_input(client):
    for path in ("/posts/nope", "/posts/nope/edit", "/posts/nope/post.md", "/posts/nope/run.json"):
        assert client.get(path).status_code == 404
    assert client.post("/posts/nope/write").status_code == 404
    slug = slug_of(create(client))
    too_long = client.post(f"/posts/{slug}/write", data={"outline": "x" * 3001})
    assert too_long.status_code == 400


def test_cross_site_posts_are_refused(client):
    assert (
        client.post("/posts", data=FORM, headers={"origin": "https://evil.example"}).status_code
        == 403
    )
    slug = slug_of(create(client))
    evil = {"origin": "https://evil.example"}
    assert client.post(f"/posts/{slug}/write", headers=evil).status_code == 403
    assert client.post(f"/posts/{slug}/delete", headers=evil).status_code == 403


def test_the_outline_can_be_saved_without_writing(client, store, composer):
    slug = slug_of(create(client))
    page = client.get(f"/posts/{slug}").text
    assert store.post(slug)["stage"] == "brief" and "Save outline" not in page
    saved = client.post(
        f"/posts/{slug}/outline/save",
        data={"outline": "- Mở bài: một\r\n- Kết bài: hai "},
        follow_redirects=False,
    )
    assert saved.status_code == 303 and saved.headers["location"] == f"/posts/{slug}"
    record = store.post(slug)
    assert record["outline"] == "- Mở bài: một\n- Kết bài: hai" and record["stage"] == "outline"
    assert composer.state is None
    page = client.get(f"/posts/{slug}").text
    assert 'formaction="/posts/' in page and "Save outline" in page
    assert "- Mở bài: một" in page and "changes you have not saved are lost" in page


def test_saving_an_empty_outline_clears_it_and_bad_saves_are_refused(
    client, store, script, composer
):
    slug = slug_of(create(client, action="outline"))
    composer.wait()
    client.post(f"/posts/{slug}/outline/save", data={"outline": ""})
    assert store.post(slug)["outline"] == "" and store.post(slug)["stage"] == "outline"
    assert (
        client.post(f"/posts/{slug}/outline/save", data={"outline": "x" * 3001}).status_code == 400
    )
    assert client.post("/posts/nope/outline/save", data={"outline": "a"}).status_code == 404
    script.hold.clear()
    client.post(f"/posts/{slug}/write")
    assert client.post(f"/posts/{slug}/outline/save", data={"outline": "a"}).status_code == 409
    client.post(f"/posts/{slug}/cancel")
    script.hold.set()
    composer.wait()


MARK = '<mark class="odd" title="May say more than your material does">'


def test_flagged_parts_are_highlighted_in_the_post_and_escaped(client, store):
    slug = slug_of(create(client))
    text = "# T\n\nCâu bình thường. Điều này <giúp> tôi tiết kiệm thời gian & tăng năng suất."
    start = text.index("Điều này")
    flags = [[start, len(text) - 1]]
    store.update_post(slug, post=text, stage="done", model="m1", checks=[], flags=flags)
    page = client.get(f"/posts/{slug}").text
    assert (
        f"{MARK}Điều này &lt;giúp&gt; tôi tiết kiệm thời gian &amp; tăng năng suất</mark>." in page
    )
    assert "1 highlighted part(s)" in page and "<giúp>" not in page


def test_bad_overlapping_or_missing_flags_never_break_the_page(client, store):
    slug = slug_of(create(client))
    bad = [[5, 2], [-1, 3], [0, 99], ["a", 1], [1], [True, 3], [0, 3], [2, 5]]
    store.update_post(slug, post="Một hai ba.", stage="done", model="m1", checks=[], flags=bad)
    page = client.get(f"/posts/{slug}").text
    assert f"{MARK}Một</mark> hai ba." in page and "1 highlighted part(s)" in page
    store.update_post(slug, flags=None)
    page = client.get(f"/posts/{slug}").text
    assert 'class="odd"' not in page and "highlighted part" not in page


def test_the_outline_warns_about_key_points_it_does_not_contain(client, store):
    slug = slug_of(create(client, points="tính năng mới\nkết quả đo được"))
    client.post(f"/posts/{slug}/outline/save", data={"outline": "- Mở bài: chào"})
    page = client.get(f"/posts/{slug}").text
    assert "These key points do not seem to be in the outline:" in page
    assert "<li>tính năng mới</li>" in page and "<li>kết quả đo được</li>" in page
    covered = "- Tính năng mới: a\n- Kết quả đo được: b"
    client.post(f"/posts/{slug}/outline/save", data={"outline": covered})
    assert "do not seem to be in the outline" not in client.get(f"/posts/{slug}").text


def test_the_post_lists_the_key_points_it_dropped_among_its_checks(client, composer):
    slug = slug_of(create(client, action="write", points="tính năng mới\nkết quả đo được"))
    composer.wait()
    page = client.get(f"/posts/{slug}").text
    assert "Key points the post does not seem to cover" in page
    assert "tính năng mới" in page and "kết quả đo được" in page


# --- voice, mode and story -----------------------------------------------------------------------


def test_the_form_has_the_voice_card_and_the_story_and_defaults_to_expand(client):
    page = client.get("/posts/new").text
    for name in (
        "mode",
        "voice_address",
        "voice_sentences",
        "voice_humor",
        "voice_avoid",
        "voice_sample",
        "story",
    ):
        assert f'name="{name}"' in page
    assert '<option value="expand" selected>' in page and "Your voice" in page


def test_the_voice_is_remembered_for_the_next_blog_post(client):
    create(client, voice_address="mình - các bạn", voice_humor="playful", mode="close")
    page = client.get("/posts/new").text
    assert 'value="mình - các bạn"' in page
    assert '<option value="playful" selected>' in page and '<option value="close" selected>' in page


def test_the_story_reaches_the_model(client, composer, script):
    create(client, action="write", story="Tối đó tôi hồi hộp lắm.")
    composer.wait()
    assert "Tối đó tôi hồi hộp lắm." in script.prompts[-1]


def test_three_voices_can_be_tried_and_one_becomes_the_voice_sample(client, store, composer):
    slug = slug_of(create(client))
    assert "Try three voices" in client.get(f"/posts/{slug}").text
    client.post(f"/posts/{slug}/voices", data={}, follow_redirects=False)
    composer.wait()
    page = client.get(f"/posts/{slug}").text
    assert page.count('name="choice"') == 3 and "Use this voice" in page
    assert "Try three new voices" in page
    picked = client.post(f"/posts/{slug}/voice", data={"choice": "1"}, follow_redirects=False)
    assert picked.status_code == 303
    sample = store.post(slug)["brief"]["voice_sample"]
    assert sample.startswith("Buổi tối") and store.blog_voice()["voice_sample"] == sample
    assert "The voice sample in use" in client.get(f"/posts/{slug}").text


def test_picking_a_voice_that_does_not_exist_is_refused(client, store):
    slug = slug_of(create(client))
    assert client.post(f"/posts/{slug}/voice", data={"choice": "0"}).status_code == 400
    store.update_post(slug, voices=["một đoạn", 5])
    assert client.post(f"/posts/{slug}/voice", data={"choice": "1"}).status_code == 400
    assert client.post(f"/posts/{slug}/voice", data={"choice": "9"}).status_code == 400
    assert client.post("/posts/nope/voices").status_code == 404


def test_editing_the_brief_keeps_the_voice_fields(client, store):
    slug = slug_of(create(client, voice_address="tớ - cậu"))
    page = client.get(f"/posts/{slug}/edit").text
    assert 'value="tớ - cậu"' in page
    client.post(f"/posts/{slug}/edit", data={**FORM, "voice_address": "mình - bạn"})
    assert store.post(slug)["brief"]["voice_address"] == "mình - bạn"
    assert store.blog_voice()["voice_address"] == "mình - bạn"


class PerModel:
    """Each model has its own outline and post."""

    def __init__(self, texts):
        self.texts = texts

    def make(self, model):
        texts = self.texts[model]

        def stream(system, prompt, num_predict, on_text):
            text = texts["outline"] if num_predict == OUTLINE_TOKENS else texts["write"]
            on_text(text)
            return text

        return stream


@pytest.fixture
def duo(store):
    texts = {
        "m1": {
            "outline": "- Mở bài: một\n- Phần: hai\n- Kết bài: ba",
            "write": "# Gỡ Rancher\n\nĐoạn đầu của m1.\n\nChạy xong rồi.",
        },
        "m2": {
            "outline": "- Mở bài: khác\n- Phần: nữa\n- Kết bài: hết",
            "write": "# Gỡ Rancher\n\nĐoạn đầu của m2. Giảm 450 MB hẳn.\n\nXong.",
        },
    }
    composer = Composer(store, PerModel(texts).make, ["m1", "m2"], Gate())
    runner = Runner(store, lambda m: lambda b: b, ["m1"], gate=composer.gate)
    return TestClient(create_app(store, runner, composer)), composer


def test_the_new_post_form_offers_models_for_the_outline_and_for_the_post(client):
    page = client.get("/posts/new").text
    assert 'name="models_outline"' in page and 'name="models_write"' in page
    assert "For the outline" in page and "For the post" in page


def test_models_ticked_in_the_form_are_kept_and_ticked_again_next_time(client, store):
    response = create(client, models_outline=["m1", "m2"], models_write=["m2"])
    slug = slug_of(response)
    assert store.post(slug)["models"] == {"outline": ["m1", "m2"], "write": ["m2"]}
    assert store.blog_voice()["pick_outline"] == "m1,m2" and store.blog_voice()["voice_humor"]
    page = client.get("/posts/new").text
    assert 'name="models_write" value="m2" checked' in page
    assert 'name="models_write" value="m1" checked' not in page
    assert 'name="models_write" value="zzz"' not in page
    other = slug_of(create(client, models_write=["zzz"]))
    assert "models" not in store.post(other)


def test_two_models_give_two_answers_to_pick_parts_from(duo, store):
    client, composer = duo
    slug = slug_of(create(client))
    client.post(f"/posts/{slug}/outline", data={"models": ["m1", "m2"]}, follow_redirects=False)
    composer.wait()
    record = store.post(slug)
    first, second = record["candidates"]["outline"]
    assert record["models"]["outline"] == ["m1", "m2"] and "một" in record["outline"]
    page = client.get(f"/posts/{slug}").text
    shown = sum(len(pieces(c["text"], "outline")) for c in (first, second))
    assert "Ideas from the models" in page and page.count('name="pick"') == shown
    parts = pieces(second["text"], "outline")
    wanted = next(n for n, part in enumerate(parts) if "nữa" in part[2])
    added = client.post(
        f"/posts/{slug}/gather",
        data={"kind": "outline", "grain": "paragraph", "pick": [f"{second['id']}.{wanted}"]},
        follow_redirects=False,
    )
    assert added.status_code == 303 and "#build-outline" in added.headers["location"]
    assert store.post(slug)["outline"].endswith(parts[wanted][2])
    assert "một" in store.post(slug)["outline"]
    assert first["text"] != second["text"]
    assert "in your text" in client.get(f"/posts/{slug}").text


def test_picks_that_make_no_sense_are_refused(duo, store):
    client, composer = duo
    slug = slug_of(create(client))
    client.post(f"/posts/{slug}/outline", data={"models": ["m1"]}, follow_redirects=False)
    composer.wait()
    cid = store.post(slug)["candidates"]["outline"][0]["id"]
    base = f"/posts/{slug}/gather"
    assert client.post(base, data={"kind": "outline", "pick": ["nope.0"]}).status_code == 400
    assert client.post(base, data={"kind": "outline"}).status_code == 400
    assert client.post(base, data={"kind": "voices", "pick": [f"{cid}.0"]}).status_code == 400
    assert (
        client.post(base, data={"kind": "outline", "grain": "x", "pick": [f"{cid}.0"]}).status_code
        == 400
    )
    assert client.post("/posts/nope/gather", data={"kind": "outline"}).status_code == 404


def test_gathering_is_refused_while_the_model_is_writing(client, composer, script):
    slug = slug_of(create(client))
    script.hold.clear()
    client.post(f"/posts/{slug}/outline", data={}, follow_redirects=False)
    try:
        refused = client.post(f"/posts/{slug}/gather", data={"kind": "outline", "pick": ["x.0"]})
        assert refused.status_code == 409
        assert client.post(f"/posts/{slug}/post/save", data={"post": "x"}).status_code == 409
    finally:
        script.hold.set()
        composer.wait()


def test_the_run_card_says_which_model_of_several_is_working(client, composer, script):
    slug = slug_of(create(client))
    script.hold.clear()
    client.post(f"/posts/{slug}/outline", data={"models": ["m1", "m2"]}, follow_redirects=False)
    try:
        assert "(model 1 of 2)" in client.get(f"/posts/{slug}").text
    finally:
        script.hold.set()
        composer.wait()


def test_parts_picked_for_the_post_are_checked_again_and_sentences_can_be_shown(duo, store):
    client, composer = duo
    slug = slug_of(create(client))
    client.post(f"/posts/{slug}/outline/save", data={"outline": "- Mở bài: một"})
    client.post(f"/posts/{slug}/write", data={"models": ["m1", "m2"]}, follow_redirects=False)
    composer.wait()
    record = store.post(slug)
    assert "m1" in record["post"] and len(record["candidates"]["write"]) == 2
    second = record["candidates"]["write"][1]
    assert "450" not in record["post"]
    page = client.get(f"/posts/{slug}?grain=sentence").text
    assert "Show paragraphs instead" in page and "Show sentences instead" not in page
    assert "Show sentences instead" in client.get(f"/posts/{slug}").text
    index = [p[2] for p in pieces(second["text"], "write")].index(
        "Đoạn đầu của m2. Giảm 450 MB hẳn."
    )
    client.post(
        f"/posts/{slug}/gather",
        data={"kind": "write", "grain": "paragraph", "pick": [f"{second['id']}.{index}"]},
        follow_redirects=False,
    )
    record = store.post(slug)
    assert "Giảm 450 MB" in record["post"] and record["model"] == "picks"
    assert any("450" in c["text"] for c in record["checks"])
    assert "Put together from the parts you picked" in client.get(f"/posts/{slug}").text


def test_the_post_text_can_be_edited_by_hand_and_cleared(client, store, composer):
    slug = slug_of(create(client))
    client.post(f"/posts/{slug}/write", data={}, follow_redirects=False)
    composer.wait()
    saved = client.post(
        f"/posts/{slug}/post/save",
        data={"post": "# Mới\r\n\r\nSửa tay 999 xong."},
        follow_redirects=False,
    )
    assert saved.status_code == 303
    record = store.post(slug)
    assert record["post"] == "# Mới\n\nSửa tay 999 xong." and record["stage"] == "done"
    assert any("999" in c["text"] for c in record["checks"]) and record["model"] == "picks"
    assert client.post(f"/posts/{slug}/post/save", data={"post": "x" * 30001}).status_code == 400
    client.post(f"/posts/{slug}/post/save", data={"post": "  "})
    cleared = store.post(slug)
    assert cleared["post"] == "" and cleared["stage"] == "brief" and cleared["checks"] == []
    # the answers of the models are still there to build a new post from
    assert "Nothing is here yet" in client.get(f"/posts/{slug}").text


def test_answers_can_be_cleared_and_the_text_built_stays(duo, store):
    client, composer = duo
    slug = slug_of(create(client))
    client.post(f"/posts/{slug}/outline", data={"models": ["m1", "m2"]}, follow_redirects=False)
    composer.wait()
    before = store.post(slug)["outline"]
    cleared = client.post(f"/posts/{slug}/candidates/clear", data={"kind": "outline"})
    assert cleared.status_code == 200 or cleared.history
    assert store.post(slug)["candidates"] == {} and store.post(slug)["outline"] == before
    assert client.post(f"/posts/{slug}/candidates/clear", data={"kind": "x"}).status_code == 400


def test_blog_models_come_from_the_environment_or_from_ollama(store, monkeypatch):
    monkeypatch.delenv("NIGHTSHIFT_POST_MODELS", raising=False)
    monkeypatch.delenv("NIGHTSHIFT_RUN_MODELS", raising=False)
    assert composer_from_env(store, Gate()).models == []
    monkeypatch.setenv("NIGHTSHIFT_RUN_MODELS", "a")
    monkeypatch.setattr("nightshift.web.list_models", lambda url: ["a", "b"])
    assert composer_from_env(store, Gate()).models == ["a", "b"]
    monkeypatch.setattr("nightshift.web.list_models", lambda url: [])
    assert composer_from_env(store, Gate()).models == ["a"]
    monkeypatch.setenv("NIGHTSHIFT_POST_MODELS", "x, y")
    assert composer_from_env(store, Gate()).models == ["x", "y"]


# --- sending a finished post to the portfolio's blog admin ---------------------------------------


class FakeReply:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def finished(client, store):
    create(client)
    slug = "go-rancher-khoi-k3s"
    store.update_post(slug, post="# Gỡ Rancher\n\nNội dung bài.", stage="done")
    return slug


def test_sending_to_the_portfolio_is_closed_until_it_is_configured(client, store, monkeypatch):
    monkeypatch.delenv("NIGHTSHIFT_PORTFOLIO_DEV_URL", raising=False)
    monkeypatch.delenv("NIGHTSHIFT_PORTFOLIO_DEV_TOKEN", raising=False)
    slug = finished(client, store)
    assert client.post(f"/posts/{slug}/send-to-dev").status_code == 503


def test_a_post_that_is_not_written_cannot_be_sent(client, store, monkeypatch):
    monkeypatch.setenv("NIGHTSHIFT_PORTFOLIO_DEV_URL", "http://portfolio/admin/api/blog/incoming")
    monkeypatch.setenv("NIGHTSHIFT_PORTFOLIO_DEV_TOKEN", "t")
    create(client)
    assert client.post("/posts/go-rancher-khoi-k3s/send-to-dev").status_code == 400
    assert client.post("/posts/nope/send-to-dev").status_code == 404


def test_the_post_is_sent_as_a_draft_with_the_shared_token(client, store, monkeypatch):
    monkeypatch.setenv("NIGHTSHIFT_PORTFOLIO_DEV_URL", "http://portfolio/admin/api/blog/incoming")
    monkeypatch.setenv("NIGHTSHIFT_PORTFOLIO_DEV_TOKEN", "shared-secret")
    seen = {}

    def fake(request, timeout):
        seen["url"] = request.full_url
        seen["token"] = request.get_header("X-ingest-token")
        seen["body"] = json.loads(request.data)
        return FakeReply(b'{"post": {"slug": "go-rancher"}}')

    monkeypatch.setattr("nightshift.web.urllib.request.urlopen", fake)
    slug = finished(client, store)
    assert client.get(f"/posts/{slug}").text.count("Send to Portfolio Dev") == 1
    reply = client.post(f"/posts/{slug}/send-to-dev", follow_redirects=False)
    assert reply.status_code == 303 and reply.headers["location"] == f"/posts/{slug}"
    assert seen["url"] == "http://portfolio/admin/api/blog/incoming"
    assert seen["token"] == "shared-secret"
    assert seen["body"]["markdown"].startswith("# Gỡ Rancher") and seen["body"]["title"]
    record = store.post(slug)
    assert record["sent_to_dev_slug"] == "go-rancher" and record["sent_to_dev"] > 0
    assert "Sent to Portfolio Dev as a draft" in client.get(f"/posts/{slug}").text


def test_a_refusal_or_an_unreachable_portfolio_is_reported_not_recorded(client, store, monkeypatch):
    monkeypatch.setenv("NIGHTSHIFT_PORTFOLIO_DEV_URL", "http://portfolio/admin/api/blog/incoming")
    monkeypatch.setenv("NIGHTSHIFT_PORTFOLIO_DEV_TOKEN", "shared-secret")
    slug = finished(client, store)

    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, io.BytesIO(b"bad token"))

    monkeypatch.setattr("nightshift.web.urllib.request.urlopen", refuse)
    reply = client.post(f"/posts/{slug}/send-to-dev")
    assert reply.status_code == 502 and "403" in reply.text and "bad token" in reply.text

    def down(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("nightshift.web.urllib.request.urlopen", down)
    reply = client.post(f"/posts/{slug}/send-to-dev")
    assert reply.status_code == 502 and "connection refused" in reply.text
    assert "sent_to_dev" not in store.post(slug)


def test_the_page_says_when_the_machine_is_resting_between_models(client, store, composer):
    create(client)
    slug = "go-rancher-khoi-k3s"
    composer.rest_seconds = 30
    client.post(f"/posts/{slug}/outline", data={"models": ["m1", "m2"]}, follow_redirects=False)
    for _ in range(300):
        if composer.state and composer.state.resting_until > 0:
            break
        time.sleep(0.01)
    page = client.get(f"/posts/{slug}").text
    assert "Resting before m2 (model 2 of 2)" in page and "left" in page
    assert client.get(f"/posts/{slug}/run.json").json()["resting_until"] > 0
    client.post(f"/posts/{slug}/cancel", follow_redirects=False)
    composer.wait(3)
    assert "Resting before" not in client.get(f"/posts/{slug}").text


def test_the_post_has_its_own_column_and_the_helpers_sit_beside_it(client, store):
    slug = finished(client, store)
    page = client.get(f"/posts/{slug}").text
    left, right = page.split(chr(34) + "helpcol" + chr(34))
    assert "ppost" in left and "ppost" not in right
    assert "build-checks" in right and "The brief" in right
    assert "Send to Portfolio Dev" in left


def test_without_text_the_outline_takes_the_wide_column(client):
    create(client)
    page = client.get("/posts/go-rancher-khoi-k3s").text
    left, right = page.split(chr(34) + "helpcol" + chr(34))
    assert "build-outline" in left and "The brief" in right
