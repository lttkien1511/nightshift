import os
import re
from pathlib import Path

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
    assert page.count('<ul class="queue">') == 2  # drafts and blog posts
    assert 'href="/drafts/one" aria-current="page"' in page
    assert 'href="/drafts/two" title="two' in page
    assert '<i class="dot waiting"' in page and 'aria-label="waiting"' in page
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


def make_app(tmp_path, drafts=0, posts=0):
    store = Store(tmp_path, max_queue=60)
    app = TestClient(create_app(store))
    for number in range(drafts):
        store.save(f"draft-{number:02d}", "x")
        os.utime(store.inbox / f"draft-{number:02d}.md", ns=(10**9 * (number + 1),) * 2)
    for number in range(posts):
        store.create_post(f"Bài {number}", {"topic": f"Bài {number}"})
    return app, store


def test_the_sidebar_shows_the_newest_eight_and_folds_the_rest_away(tmp_path):
    app, _ = make_app(tmp_path, drafts=12)
    page = app.get("/").text
    side = page[page.index('<aside class="side">') : page.index("</aside>")]
    assert 'Drafts <span class="count">12</span>' in side
    assert "Show all (12)" in side and "Show fewer" in side
    top, rest = side.split('<details class="rest"')
    assert top.index("draft-11") < top.index("draft-04") and "draft-03" not in top
    assert "draft-03" in rest and "draft-00" in rest and " open" not in rest.split(">")[0]


def test_the_open_draft_keeps_its_folded_list_open(tmp_path):
    app, _ = make_app(tmp_path, drafts=12)
    side = app.get("/drafts/draft-00").text
    assert '<details class="rest" open>' in side
    assert '<details class="rest" open>' not in app.get("/drafts/draft-11").text


def test_rows_are_one_line_with_the_details_in_a_tooltip(tmp_path):
    app, _ = make_app(tmp_path, drafts=1)
    page = app.get("/").text
    assert 'title="draft-00 \u00b7 1 bytes \u00b7 waiting"' in page
    assert '<span class="t">draft-00</span>' in page and 'class="m"' not in page
    assert 'data-q="draft-00"' in page


def test_the_search_box_is_there_only_when_there_is_something_to_search(tmp_path):
    few, _ = make_app(tmp_path / "few", drafts=2, posts=2)
    assert 'id="find"' not in few.get("/").text
    many, _ = make_app(tmp_path / "many", drafts=3, posts=2)
    page = many.get("/").text
    assert 'class="finder" hidden' in page and 'id="find"' in page and 'id="nomatch"' in page
    assert "Search drafts and posts" in page
    assert 'data-q="bài 0 bai-0"' in page


def test_the_two_new_buttons_sit_side_by_side_above_the_groups(tmp_path):
    app, _ = make_app(tmp_path)
    page = app.get("/").text
    row = page[page.index('<div class="newrow">') : page.index("</div>", page.index("newrow"))]
    assert 'href="/"' in row and 'href="/posts/new"' in row
    assert (
        page.index("newrow") < page.index('data-group="drafts"') < page.index('data-group="posts"')
    )
    assert "No drafts yet." in page and "No blog posts yet." in page


def test_the_script_folds_groups_and_filters_rows():
    script = (Path(__file__).parent.parent / "src/nightshift/static/ui.js").read_text("utf-8")
    assert "nightshift-groups" in script and 'getElementById("find")' in script
    assert "data-q" in script and "Escape" in script
