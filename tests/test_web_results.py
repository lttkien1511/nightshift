from fastapi.testclient import TestClient

from nightshift.storage import Store
from nightshift.web import create_app
from nightshift.worker import run


def test_draft_page_shows_a_diff_per_model_and_escapes_html(tmp_path):
    store = Store(tmp_path)
    app = TestClient(create_app(store))
    store.save("a", "Một đoạn <b>văn</b> ngắn.\n")
    run(store, ["m:1b"], lambda model: lambda block: block.replace("ngắn", "gọn"))
    page = app.get("/drafts/a").text
    assert "<h2>m:1b</h2>" in page
    assert "<del>ngắn</del>" in page and "<ins>gọn</ins>" in page
    assert "<b>" not in page.replace("<b>văn</b>", "") and "&lt;b&gt;" in page
    assert app.get("/drafts/a/results").json()[0]["model"] == "m:1b"
    assert app.get("/drafts/nope/results").status_code == 404


def test_page_warns_when_the_draft_changed_after_the_run(tmp_path):
    store = Store(tmp_path)
    app = TestClient(create_app(store))
    store.save("a", "Đoạn một.\n")
    run(store, ["m"], lambda model: lambda block: block)
    store.save("a", "Đoạn hai.\n")
    assert "changed after this run" in app.get("/drafts/a").text
