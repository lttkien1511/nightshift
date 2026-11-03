import threading

import pytest
from fastapi.testclient import TestClient

from nightshift.runner import Runner
from nightshift.storage import DraftTooLarge, NameTaken, StorageError, Store
from nightshift.web import create_app
from nightshift.worker import run

DRAFT = "Đoạn một ngắn.\n\nĐoạn hai cũng ngắn.\n"


def fix(block: str) -> str:
    return block.replace("ngắn", "gọn")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path, max_bytes=300, max_queue=5)
    s.ensure()
    return s


@pytest.fixture
def client(store):
    return TestClient(create_app(store))


def post_edit(client, slug, name=None, text=DRAFT, headers=None, **extra):
    data = {"name": slug if name is None else name, "text": text, **extra}
    return client.post(f"/drafts/{slug}/edit", data=data, headers=headers, follow_redirects=False)


# --- storage ------------------------------------------------------------------------------


def test_rename_moves_the_draft_and_its_results(store):
    store.save("a", DRAFT)
    run(store, ["m"], lambda model: fix)
    store.rename("a", "b")
    assert store.read("a") is None and store.read("b") == DRAFT
    assert store.result("b", "m") is not None and store.result("a", "m") is None
    assert [d.slug for d in store.drafts()] == ["b"] and store.drafts()[0].has_results


def test_rename_refuses_a_taken_or_missing_name(store):
    store.save("a", DRAFT)
    store.save("b", DRAFT)
    with pytest.raises(NameTaken):
        store.rename("a", "b")
    with pytest.raises(StorageError, match="no such draft"):
        store.rename("zzz", "c")
    with pytest.raises(StorageError):
        store.rename("a", "Not A Slug")
    assert store.read("a") == DRAFT and store.read("b") == DRAFT


def test_validate_matches_what_save_would_refuse(store):
    store.validate("ok")
    with pytest.raises(StorageError, match="empty"):
        store.validate("   ")
    with pytest.raises(DraftTooLarge):
        store.validate("a" * 301)


# --- the edit page ------------------------------------------------------------------------


def test_draft_page_has_an_edit_button(client, store):
    store.save("a", DRAFT)
    assert 'href="/drafts/a/edit">Edit</a>' in client.get("/drafts/a").text


def test_edit_page_is_prefilled_and_escapes_the_text(client, store):
    store.save("a", "\n<script>alert(1)</script> & thêm\n")
    page = client.get("/drafts/a/edit").text
    assert 'value="a"' in page and 'name="text"' in page
    assert '<textarea name="text" id="text" rows="22" required>\n\n&lt;script&gt;' in page
    assert "<script>alert" not in page
    assert 'data-max="300"' in page
    assert "Save and run now" not in page
    assert client.get("/drafts/missing/edit").status_code == 404


def test_edit_page_offers_save_and_run_now_when_a_model_exists(store):
    runner = Runner(store, lambda model: fix, ["m1"])
    page = TestClient(create_app(store, runner))
    store.save("a", DRAFT)
    assert "Save and run now" in page.get("/drafts/a/edit").text


# --- saving -------------------------------------------------------------------------------


def test_saving_new_text_replaces_the_draft_and_marks_results_out_of_date(client, store):
    store.save("a", DRAFT)
    run(store, ["m"], lambda model: fix)
    response = post_edit(client, "a", text="Một đoạn hoàn toàn mới.\n")
    assert response.status_code == 303 and response.headers["location"] == "/drafts/a"
    assert store.read("a") == "Một đoạn hoàn toàn mới.\n"
    assert "You changed this draft after the run" in client.get("/drafts/a").text


def test_saving_unchanged_text_keeps_the_results_valid(client, store):
    store.save("a", DRAFT)
    run(store, ["m"], lambda model: fix)
    post_edit(client, "a")
    assert "You changed this draft" not in client.get("/drafts/a").text


def test_a_draft_stored_with_crlf_is_unchanged_when_posted_back_by_a_browser(client, store):
    store.save("a", DRAFT.replace("\n", "\r\n"))
    run(store, ["m"], lambda model: fix)
    post_edit(client, "a", text=DRAFT.replace("\n", "\r\n"))
    assert "You changed this draft" not in client.get("/drafts/a").text


def test_new_drafts_from_a_form_are_stored_with_lf(client, store):
    client.post("/drafts", data={"name": "x", "text": "a\r\n\r\nb\r\n"})
    assert store.read("x") == "a\n\nb\n"


def test_renaming_keeps_the_results_and_opens_the_new_address(client, store):
    store.save("a", DRAFT)
    run(store, ["m"], lambda model: fix)
    response = post_edit(client, "a", name="Tên Mới")
    assert response.headers["location"] == "/drafts/ten-moi"
    assert store.read("a") is None and store.read("ten-moi") == DRAFT
    assert "You changed this draft" not in client.get("/drafts/ten-moi").text
    assert client.get("/drafts/a").status_code == 404


def test_rename_and_new_text_in_one_save(client, store):
    store.save("a", DRAFT)
    post_edit(client, "a", name="b", text="Bản mới hoàn toàn.\n")
    assert store.read("b") == "Bản mới hoàn toàn.\n" and store.read("a") is None


def test_bad_saves_change_nothing(client, store):
    store.save("a", DRAFT)
    store.save("b", "Bài khác.\n")
    assert post_edit(client, "a", name="b").status_code == 409
    assert post_edit(client, "a", name="???").status_code == 400
    assert post_edit(client, "a", name="c", text="  ").status_code == 400
    assert post_edit(client, "a", name="c", text="x" * 301).status_code == 413
    assert post_edit(client, "nope").status_code == 404
    assert store.read("a") == DRAFT and store.read("b") == "Bài khác.\n"
    assert store.read("c") is None


def test_cross_site_edits_are_refused(client, store):
    store.save("a", DRAFT)
    assert post_edit(client, "a", headers={"origin": "https://evil.example"}).status_code == 403


# --- with the model -----------------------------------------------------------------------


class Gate:
    def __init__(self):
        self.open = threading.Event()

    def __call__(self, block):
        self.open.wait(5)
        return fix(block)


def test_save_and_run_now_starts_a_run_on_the_new_address(store):
    gate = Gate()
    runner = Runner(store, lambda model: gate, ["m1"])
    app = TestClient(create_app(store, runner))
    store.save("a", DRAFT)
    response = post_edit(app, "a", name="b", run="1")
    assert response.headers["location"] == "/drafts/b"
    assert runner.state.slug == "b" and runner.state.running
    gate.open.set()
    runner.wait()
    assert store.result("b", "m1") is not None


def test_a_draft_cannot_be_edited_while_the_model_is_working_on_it(store):
    gate = Gate()
    runner = Runner(store, lambda model: gate, ["m1"])
    app = TestClient(create_app(store, runner))
    store.save("a", DRAFT)
    store.save("b", DRAFT)
    app.post("/drafts/a/run")
    assert post_edit(app, "a", text="Đổi giữa chừng.\n").status_code == 409
    assert post_edit(app, "b", text="Bài khác đổi được.\n").status_code == 303
    assert store.read("a") == DRAFT
    gate.open.set()
    runner.wait()
