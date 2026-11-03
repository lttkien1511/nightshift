import threading

import pytest
from fastapi.testclient import TestClient

from nightshift.composer import Composer
from nightshift.gate import Gate
from nightshift.runner import Runner
from nightshift.storage import Store
from nightshift.web import create_app

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
POST = "# Gỡ Rancher\n\nTôi gặp [[HIDE_1]]. Chạy [[CODE_1]] rồi xem, giảm 450 MB."


class Script:
    """A model that answers outline and post requests, and can wait at a gate."""

    def __init__(self):
        self.hold = threading.Event()
        self.hold.set()
        self.prompts = []

    def __call__(self, system, prompt, num_predict, on_text):
        self.prompts.append(prompt)
        text = OUTLINE if num_predict == 400 else POST
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
    assert store.post(slug)["outline"] == OUTLINE and store.post(slug)["stage"] == "outline"
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
    assert '<span class="badge writing">writing</span>' in page and "writing go-rancher" in page
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
