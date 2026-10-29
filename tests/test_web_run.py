import threading

import pytest
from fastapi.testclient import TestClient

from nightshift.runner import Runner
from nightshift.storage import Store
from nightshift.web import create_app

DRAFT = "Đoạn một ngắn.\n\nĐoạn hai cũng ngắn.\n"


class Gate:
    """An editor that waits at a gate, so a test can look at a run while it is going on."""

    def __init__(self):
        self.open = threading.Event()

    def __call__(self, block):
        self.open.wait(5)
        return block.replace("ngắn", "gọn")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.ensure()
    s.save("a", DRAFT)
    s.save("b", DRAFT)
    return s


@pytest.fixture
def gate():
    return Gate()


@pytest.fixture
def runner(store, gate):
    return Runner(store, lambda model: gate, ["m1", "m2"])


@pytest.fixture
def client(store, runner):
    return TestClient(create_app(store, runner))


def test_without_models_the_run_controls_stay_hidden(store):
    plain = TestClient(create_app(store))
    assert "Run now" not in plain.get("/drafts/a").text
    assert "Queue and run now" not in plain.get("/").text


def test_the_draft_page_offers_run_now_with_a_model_choice(client):
    page = client.get("/drafts/a").text
    assert '<form class="runform" method="post" action="/drafts/a/run">' in page
    assert '<option value="m1">m1</option>' in page and '<option value="m2">m2</option>' in page
    assert "Run now" in page and "http-equiv" not in page


def test_the_new_draft_form_can_queue_and_run(client):
    assert "Queue and run now" in client.get("/").text


def test_run_shows_progress_reloads_itself_and_then_the_result(client, runner, gate, store):
    response = client.post("/drafts/a/run", data={"model": "m2"}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/drafts/a"
    page = client.get("/drafts/a").text
    assert "Editing with m2" in page and "<progress" in page
    assert '<meta http-equiv="refresh" content="3">' in page
    assert "Run now" not in page and "/drafts/a/cancel" in page
    assert '<span class="badge editing">editing</span>' in page
    assert "editing a 0/2" in page
    gate.open.set()
    runner.wait()
    done = client.get("/drafts/a").text
    assert "http-equiv" not in done and "<progress" not in done
    assert "<ins>gọn</ins>" in done and ">m2</a>" in done
    assert store.result("a", "m2") is not None


def test_another_draft_cannot_start_while_one_is_running(client, runner, gate):
    client.post("/drafts/a/run")
    assert client.post("/drafts/b/run").status_code == 409
    page = client.get("/drafts/b").text
    assert "Another draft is being edited" in page and 'href="/drafts/a"' in page
    assert "http-equiv" not in page
    gate.open.set()
    runner.wait()


def test_cancel_stops_the_run(client, runner, gate, store):
    client.post("/drafts/a/run")
    assert client.post("/drafts/b/cancel", follow_redirects=False).status_code == 303
    assert runner.state.running
    client.post("/drafts/a/cancel")
    gate.open.set()
    runner.wait()
    assert runner.state.state == "cancelled" and store.result("a", "m1") is None
    assert "The last run was cancelled" in client.get("/drafts/a").text


def test_run_state_as_json(client, runner, gate):
    assert client.get("/drafts/a/run.json").json() == {"state": "idle"}
    client.post("/drafts/a/run")
    body = client.get("/drafts/a/run.json").json()
    assert body["state"] == "running" and body["total"] == 2 and body["model"] == "m1"
    assert client.get("/drafts/b/run.json").json() == {"state": "idle"}
    gate.open.set()
    runner.wait()
    assert client.get("/drafts/a/run.json").json()["state"] == "done"


def test_queue_and_run_now_starts_the_run(client, runner, gate, store):
    response = client.post(
        "/drafts", data={"name": "Mới", "text": DRAFT, "run": "1"}, follow_redirects=False
    )
    assert response.headers["location"] == "/drafts/moi"
    assert runner.state.slug == "moi" and runner.state.running
    gate.open.set()
    runner.wait()
    assert store.result("moi", "m1") is not None


def test_queue_and_run_now_still_queues_when_the_model_is_busy(client, runner, gate, store):
    client.post("/drafts/a/run")
    response = client.post(
        "/drafts", data={"name": "Mới", "text": DRAFT, "run": "1"}, follow_redirects=False
    )
    assert response.status_code == 303 and store.read("moi") == DRAFT
    gate.open.set()
    runner.wait()


def test_bad_run_requests(client):
    assert client.post("/drafts/nope/run").status_code == 404
    assert client.post("/drafts/a/run", data={"model": "other"}).status_code == 400


def test_a_failed_run_explains_itself(store, gate):
    def down(block):
        from nightshift.ollama import OllamaError

        raise OllamaError("ollama is unreachable")

    runner = Runner(store, lambda model: down, ["m1"])
    client = TestClient(create_app(store, runner))
    client.post("/drafts/a/run")
    runner.wait()
    page = client.get("/drafts/a").text
    assert "The last run failed" in page and "unreachable" in page and "Run now" in page


def test_cross_site_run_requests_are_refused(client):
    assert (
        client.post("/drafts/a/run", headers={"origin": "https://evil.example"}).status_code == 403
    )
