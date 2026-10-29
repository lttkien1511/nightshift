import threading

import pytest

from nightshift.ollama import OllamaError
from nightshift.runner import Busy, Runner, RunnerError
from nightshift.storage import Store

DRAFT = "Đoạn một ngắn.\n\nĐoạn hai cũng ngắn.\n\n```bash\nls\n```\n\nĐoạn ba ngắn nữa.\n"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.ensure()
    s.save("a", DRAFT)
    s.save("b", DRAFT)
    return s


def fix(block: str) -> str:
    return block.replace("ngắn", "gọn")


def runner_with(store, editor=fix, models=("m1", "m2")):
    return Runner(store, lambda model: editor, models)


def test_a_run_writes_the_same_result_the_nightly_worker_would(store):
    runner = runner_with(store)
    started = runner.start("a")
    assert started.model == "m1" and started.total == 3 and started.running
    runner.wait()
    state = runner.state
    assert state.state == "done" and state.done == 3 and state.error == ""
    result = store.result("a", "m1")
    assert result["stats"]["edited"] == 3
    assert "gọn" in result["edited"] and result["model"] == "m1"
    assert "```bash\nls\n```" in result["edited"]


def test_the_chosen_model_is_used_and_unknown_ones_are_refused(store):
    runner = runner_with(store)
    runner.start("a", "m2")
    runner.wait()
    assert store.result("a", "m2") is not None and store.result("a", "m1") is None
    with pytest.raises(RunnerError, match="not available"):
        runner.start("a", "other")


def test_a_missing_draft_is_refused(store):
    with pytest.raises(RunnerError, match="no such draft"):
        runner_with(store).start("nope")


def test_only_one_run_at_a_time_and_progress_is_visible(store):
    gate = threading.Event()
    seen = []

    def slow(block):
        seen.append(block)
        gate.wait(5)
        return fix(block)

    runner = runner_with(store, slow)
    runner.start("a")
    with pytest.raises(Busy, match="a is being edited"):
        runner.start("b")
    gate.set()
    runner.wait()
    assert runner.state.state == "done"
    runner.start("b")
    runner.wait()
    assert store.result("b", "m1") is not None


def test_cancel_stops_the_run_and_saves_nothing(store):
    entered = threading.Event()
    release = threading.Event()

    def blocking(block):
        entered.set()
        release.wait(5)
        return fix(block)

    runner = runner_with(store, blocking)
    runner.start("a")
    assert entered.wait(5)
    runner.cancel()
    release.set()
    runner.wait()
    assert runner.state.state == "cancelled"
    assert store.result("a", "m1") is None


def test_a_model_that_never_answers_fails_without_saving(store):
    def down(block):
        raise OllamaError("ollama is unreachable")

    runner = runner_with(store, down)
    runner.start("a")
    runner.wait()
    assert runner.state.state == "failed"
    assert "unreachable" in runner.state.error
    assert store.result("a", "m1") is None


def test_one_bad_block_does_not_fail_the_whole_run(store):
    def flaky(block):
        if "hai" in block:
            raise OllamaError("timeout")
        return fix(block)

    runner = runner_with(store, flaky)
    runner.start("a")
    runner.wait()
    assert runner.state.state == "done"
    stats = store.result("a", "m1")["stats"]
    assert stats["error"] == 1 and stats["edited"] == 2


def test_a_finished_run_can_be_followed_by_another_on_the_same_draft(store):
    runner = runner_with(store)
    runner.start("a")
    runner.wait()
    runner.start("a")
    runner.wait()
    assert runner.state.state == "done"
