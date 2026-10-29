import re

import pytest
from fastapi.testclient import TestClient

from nightshift.storage import Store
from nightshift.web import create_app
from nightshift.worker import run

DRAFT = "Một đoạn <b>văn</b> ngắn.\n\nĐoạn thứ hai cũng ngắn.\n"


def fix(block: str) -> str:
    return block.replace("ngắn", "gọn")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


@pytest.fixture
def client(store):
    return TestClient(create_app(store))


def processed(store, name="a", text=DRAFT, model="m:1b"):
    store.save(name, text)
    run(store, [model], lambda m: fix)


def test_draft_page_shows_side_by_side_diff_and_escapes_html(store, client):
    processed(store)
    page = client.get("/drafts/a").text
    assert 'class="tab"' in page and ">m:1b</a>" in page
    assert '<pre class="old">' in page and '<pre class="new">' in page
    assert "<del>ngắn</del>" in page and "<ins>gọn</ins>" in page
    assert "<b>văn</b>" not in page and "&lt;b&gt;" in page
    assert client.get("/drafts/a/results").json()[0]["model"] == "m:1b"
    assert client.get("/drafts/nope/results").status_code == 404


def test_page_without_results_says_so_and_still_shows_the_original(store, client):
    store.save("a", DRAFT)
    page = client.get("/drafts/a").text
    assert "No results yet" in page and "Original draft" in page


def test_filters_changed_rejected_and_all(store, client):
    processed(store, text="Đoạn ngắn.\n\n## Tiêu đề\n\nĐoạn khác ngắn.\n")
    changed = client.get("/drafts/a?show=changed").text
    everything = client.get("/drafts/a?show=all").text
    rejected = client.get("/drafts/a?show=rejected").text
    assert changed.count('class="block edited"') == 2
    assert everything.count('class="block ') > changed.count('class="block ')
    assert "Nothing to show" in rejected
    assert client.get("/drafts/a?show=nonsense").status_code == 200


def test_a_rejected_edit_shows_the_reason_and_cannot_be_kept(store, client):
    store.save("a", "Đoạn có mã `kubectl get pods` ở đây.\n")
    run(store, ["m"], lambda m: lambda block: block.replace("[[CODE_1]]", ""))
    page = client.get("/drafts/a?show=rejected").text
    assert 'class="block rejected"' in page and "Original kept:" in page
    assert 'name="keep"' not in page
    assert "Most common reasons" in page


def test_two_models_get_two_tabs_and_the_requested_one_is_shown(store, client):
    store.save("a", DRAFT)
    run(store, ["m1", "m2"], lambda m: fix if m == "m1" else (lambda b: b.replace("ngắn", "cụt")))
    page = client.get("/drafts/a").text
    assert page.count('class="tab"') + page.count('class="tab" aria-current') >= 2
    assert "<ins>gọn</ins>" in page
    other = client.get("/drafts/a?model=m2").text
    assert "<ins>cụt</ins>" in other and "<ins>gọn</ins>" not in other


def test_stale_results_switch_merging_off(store, client):
    processed(store, text="Đoạn ngắn.\n")
    store.save("a", "Đoạn ngắn, đã sửa.\n")
    page = client.get("/drafts/a").text
    assert "You changed this draft after the run" in page
    assert "Download merged" not in page and "disabled" in page


def test_merged_download_applies_only_the_kept_edits(store, client):
    processed(store)
    page = client.get("/drafts/a").text
    first = re.findall(r'name="keep" value="(\d+)"', page)[0]
    response = client.get("/drafts/a/merged", params={"model": "m_1b", "keep": first})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert 'filename="a-edited.md"' in response.headers["content-disposition"]
    assert response.text.count("gọn") == 1 and response.text.count("ngắn") == 1
    nothing = client.get("/drafts/a/merged", params={"model": "m_1b"})
    assert nothing.text == DRAFT


def test_merged_download_rejects_bad_requests(store, client):
    processed(store)
    assert client.get("/drafts/a/merged", params={"model": "unknown"}).status_code == 404
    assert client.get("/drafts/nope/merged", params={"model": "m_1b"}).status_code == 404
    assert client.get("/drafts/a/merged", params={"model": "m_1b", "keep": "x"}).status_code == 400
    store.save("a", "đổi rồi\n")
    assert client.get("/drafts/a/merged", params={"model": "m_1b"}).status_code == 409
