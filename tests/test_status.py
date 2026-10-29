from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from nightshift.ollama import list_models
from nightshift.status import LOCAL, collect, next_run, render
from nightshift.storage import Store
from nightshift.web import create_app
from nightshift.worker import run

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)  # 19:00 local


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.ensure()
    return s


def nodes(store, url, now=NOW):
    deploy, nightly = collect(store, url, now=now, version="9.9")
    return {n.title: n for n in deploy + nightly}


def test_next_run_is_the_coming_one_oclock_local():
    assert next_run(NOW).astimezone(LOCAL) == datetime(2026, 10, 30, 1, 0, tzinfo=LOCAL)
    early = datetime(2026, 10, 28, 17, 30, tzinfo=UTC)  # 00:30 local on the 29th
    assert next_run(early).astimezone(LOCAL) == datetime(2026, 10, 29, 1, 0, tzinfo=LOCAL)


def test_list_models_reports_installed_models(fake_ollama):
    fake_ollama.installed = [{"name": "b:1", "size": 5}, {"name": "a:1", "size": 7}]
    info = list_models(fake_ollama.url)
    assert info.reachable
    assert info.models == (("a:1", 7), ("b:1", 5))


def test_list_models_never_raises_when_the_server_is_gone():
    info = list_models("http://127.0.0.1:1", timeout=1)
    assert not info.reachable and info.error


def test_unreachable_ollama_shows_as_down(store):
    assert nodes(store, "http://127.0.0.1:1")["Ollama"].state == "bad"


def test_ollama_without_the_expected_model_is_a_warning(store, fake_ollama):
    store.write_worker_status(
        {"state": "finished", "started": NOW.isoformat(), "models": ["want:1"], "done": 0}
    )
    fake_ollama.installed = [{"name": "other:1", "size": 1}]
    node = nodes(store, fake_ollama.url)["Ollama"]
    assert node.state == "warn"
    assert any("want:1" in line for line in node.lines)


def test_worker_that_never_ran_is_idle(store, fake_ollama):
    node = nodes(store, fake_ollama.url)["Worker"]
    assert node.state == "idle"
    assert any("next run" in line for line in node.lines)


@pytest.mark.parametrize(
    ("extra", "state"),
    [
        ({"done": 2, "failed": 0}, "ok"),
        ({"done": 1, "failed": 1}, "warn"),
        ({"done": 0, "failed": 1}, "bad"),
        ({"done": 1, "failed": 0, "timed_out": True}, "warn"),
    ],
)
def test_finished_worker_state(store, fake_ollama, extra, state):
    done = (NOW - timedelta(hours=2)).isoformat()
    store.write_worker_status({"state": "finished", "started": done, "finished": done, **extra})
    assert nodes(store, fake_ollama.url)["Worker"].state == state


def test_a_run_that_never_reported_back_is_flagged(store, fake_ollama):
    long_ago = (NOW - timedelta(hours=9)).isoformat()
    store.write_worker_status({"state": "running", "started": long_ago})
    assert nodes(store, fake_ollama.url)["Worker"].state == "bad"
    recent = (NOW - timedelta(minutes=5)).isoformat()
    store.write_worker_status({"state": "running", "started": recent})
    assert nodes(store, fake_ollama.url)["Worker"].state == "warn"


def test_queue_and_results_count_waiting_ready_and_stale(store, fake_ollama):
    store.save("one", "Đoạn một.\n")
    store.save("two", "Đoạn hai.\n")
    store.save("three", "Đoạn ba.\n")
    run(store, ["m"], lambda model: lambda block: block, deadline=None)
    store.save("three", "Đoạn ba đã đổi.\n")
    store.save("four", "Bài mới.\n")
    found = nodes(store, fake_ollama.url)
    assert found["Queue"].lines[0].startswith("1 waiting")
    assert found["Results"].state == "warn"
    assert "1 out of date" in found["Results"].lines


def test_render_escapes_everything():
    from nightshift.status import Node

    page = render([Node("<script>", "ok", ("<b>x</b>",), "https://x.example/?a=1&b=2")], [])
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "&lt;b&gt;x&lt;/b&gt;" in page
    assert "a=1&amp;b=2" in page


def test_status_page_is_served_and_refreshes(tmp_path):
    client = TestClient(create_app(Store(tmp_path)))
    response = client.get("/status")
    assert response.status_code == 200
    assert 'http-equiv="refresh"' in response.text
    assert "Pipeline status" in response.text
    assert "script-src" not in response.headers["content-security-policy"]
    assert "/status" in client.get("/").text


def test_worker_main_style_status_roundtrip(store):
    store.write_worker_status({"state": "finished", "done": 3})
    assert store.worker_status() == {"state": "finished", "done": 3}
    assert Store(store.root / "elsewhere").worker_status() is None


def test_status_json_has_counts_and_no_draft_text(tmp_path):
    store = Store(tmp_path)
    client = TestClient(create_app(store))
    store.save("secret-title", "Nội dung riêng tư.\n")
    body = client.get("/status.json").json()
    assert body["queue"] == {"waiting": 1, "ready": 0, "stale": 0}
    assert body["worker"] == {}
    assert body["version"]
    assert "riêng tư" not in client.get("/status.json").text
