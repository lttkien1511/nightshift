import pytest
from fastapi.testclient import TestClient

from nightshift.status import queue_counts, snapshot
from nightshift.storage import Store
from nightshift.web import create_app
from nightshift.worker import run


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.ensure()
    return s


def test_counts_waiting_ready_and_stale(store):
    store.save("one", "Đoạn một.\n")
    store.save("two", "Đoạn hai.\n")
    store.save("three", "Đoạn ba.\n")
    run(store, ["m"], lambda model: lambda block: block)
    store.save("three", "Đoạn ba đã đổi.\n")
    store.save("four", "Bài mới.\n")
    assert queue_counts(store) == (1, 3, 1)


def test_worker_status_roundtrip(store):
    assert store.worker_status() is None
    store.write_worker_status({"state": "finished", "done": 3})
    assert store.worker_status() == {"state": "finished", "done": 3}


def test_snapshot_includes_the_worker_status(store):
    assert snapshot(store, "1.2.3") == {
        "version": "1.2.3",
        "queue": {"waiting": 0, "ready": 0, "stale": 0},
        "worker": {},
    }
    store.write_worker_status({"state": "finished", "done": 1})
    assert snapshot(store, "1")["worker"] == {"state": "finished", "done": 1}


def test_status_json_has_counts_and_no_draft_text(tmp_path):
    store = Store(tmp_path)
    client = TestClient(create_app(store))
    store.save("secret-title", "Nội dung riêng tư.\n")
    response = client.get("/status.json")
    assert response.json()["queue"] == {"waiting": 1, "ready": 0, "stale": 0}
    assert "riêng tư" not in response.text and "secret-title" not in response.text


def test_the_html_status_page_is_gone(tmp_path):
    client = TestClient(create_app(Store(tmp_path)))
    assert client.get("/status").status_code == 404
    assert "/status" not in client.get("/").text
