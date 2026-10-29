import re

import pytest
from fastapi.testclient import TestClient

from nightshift.storage import Store
from nightshift.ui import split_diff, strip_html
from nightshift.web import create_app
from nightshift.worker import run


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(Store(tmp_path, max_bytes=500, max_queue=2)))


def test_split_diff_puts_deletions_left_and_insertions_right():
    left, right = split_diff("Chạy lệnh nhé", "Chạy lệnh nha")
    assert left == "Chạy lệnh <del>nhé</del>"
    assert right == "Chạy lệnh <ins>nha</ins>"
    assert split_diff("<b>", "<i>") == ("&lt;<del>b</del>&gt;", "&lt;<ins>i</ins>&gt;")


def test_layout_has_sidebar_main_theme_switch_and_no_inline_script(client):
    page = client.get("/").text
    assert '<aside class="side">' in page and "<main>" in page
    for mode in ("auto", "light", "dark"):
        assert f'data-mode="{mode}"' in page
    assert '<link rel="stylesheet" href="/static/ui.css">' in page
    assert re.findall(r"<script[^>]*>", page) == ['<script src="/static/ui.js">']
    assert " style=" not in page


def test_csp_allows_only_own_scripts_and_styles(client):
    csp = client.get("/").headers["content-security-policy"]
    assert "script-src 'self'" in csp and "style-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp


def test_static_files_are_served(client):
    css = client.get("/static/ui.css")
    js = client.get("/static/ui.js")
    assert css.status_code == 200 and "text/css" in css.headers["content-type"]
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert 'data-phase="night"' in css.text and "prefers-color-scheme" in css.text
    assert "eval(" not in js.text and "innerHTML" not in js.text
    assert client.get("/static/other").status_code == 404


def test_new_draft_form_shows_the_size_limit_and_a_file_picker(client):
    page = client.get("/").text
    assert 'data-max="500"' in page and "of 500 bytes" in page
    assert 'type="file"' in page and 'name="text"' in page


def test_sidebar_lists_drafts_with_state_and_marks_the_open_one(client):
    client.post("/drafts", data={"name": "one", "text": "Đoạn ngắn.\n"})
    client.post("/drafts", data={"name": "two", "text": "Đoạn hai.\n"})
    page = client.get("/drafts/one").text
    assert page.count('<ul class="queue">') == 1
    assert 'href="/drafts/one" aria-current="page"' in page
    assert 'href="/drafts/two"><' in page
    assert '<span class="badge waiting">waiting</span>' in page
    assert "The queue is full" in page


def test_creating_a_draft_opens_it(client):
    form = {"name": "Bài mới", "text": "x\n"}
    response = client.post("/drafts", data=form, follow_redirects=False)
    assert response.headers["location"] == "/drafts/bai-moi"


def test_status_strip_counts_and_worker(tmp_path):
    store = Store(tmp_path)
    app = TestClient(create_app(store))
    store.save("a", "Đoạn ngắn.\n")
    assert "1 waiting" in app.get("/").text and "worker has not run yet" in app.get("/").text
    run(store, ["m"], lambda m: lambda b: b)
    done = "2026-10-29T18:00:00+00:00"
    store.write_worker_status({"state": "finished", "finished": done, "failed": 0})
    page = app.get("/").text
    assert "1 with results" in page and 'time data-rel datetime="2026-10-29T18:00:00+00:00"' in page


def test_strip_shows_running_failed_and_only_safe_monitor_links():
    snap = {"queue": {}, "worker": {"state": "running", "failed": 2}}
    html = strip_html(snap, "https://monitor.example.lab")
    assert "worker running" in html and "2 failed" in html
    assert 'href="https://monitor.example.lab"' in html
    assert "Monitor" not in strip_html(snap, "javascript:alert(1)")
    assert "Monitor" not in strip_html(snap, "")


def test_monitor_link_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHTSHIFT_MONITOR_URL", "https://monitor.example.lab")
    page = TestClient(create_app(Store(tmp_path))).get("/").text
    assert 'class="chip link" href="https://monitor.example.lab"' in page


def test_delete_asks_for_confirmation_in_the_page(client):
    client.post("/drafts", data={"name": "x", "text": "Đoạn.\n"})
    assert "data-confirm=" in client.get("/drafts/x").text
