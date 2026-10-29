import pytest

from nightshift.ollama import OllamaConfig, OllamaEditor, OllamaError
from nightshift.storage import Store, model_file
from nightshift.worker import digest, run

DRAFT = "# Tiêu đề\n\nĐây là đoạn văn đầu tiên của bài.\n\nGọi `kubectl get pods` để xem.\n"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.ensure()
    return s


def test_run_writes_one_result_per_draft_and_model(store):
    store.save("Bài một", DRAFT)
    summary = run(store, ["a:1b", "b:2b"], lambda model: lambda block: block + " ok")
    assert summary.done == [("bai-mot", "a:1b"), ("bai-mot", "b:2b")]
    result = store.result("bai-mot", "a:1b")
    assert result["model"] == "a:1b"
    assert result["draft_sha256"] == digest(DRAFT)
    assert result["edited"].count(" ok") == result["stats"]["edited"]
    assert [r["model"] for r in store.results("bai-mot")] == ["a:1b", "b:2b"]


def test_models_run_one_after_another_and_are_released(store):
    store.save("a", DRAFT)
    store.save("b", DRAFT)
    events = []

    def factory(model):
        def editor(block):
            events.append(("edit", model))
            return block

        return editor

    run(store, ["m1", "m2"], factory, release=lambda model: events.append(("release", model)))
    assert events.index(("release", "m1")) < events.index(("edit", "m2"))
    assert events[-1] == ("release", "m2")


def test_second_run_skips_finished_work(store):
    store.save("a", DRAFT)
    calls = []

    def editor(block):
        calls.append(block)
        return block

    run(store, ["m"], lambda model: editor)
    first = len(calls)
    summary = run(store, ["m"], lambda model: editor)
    assert len(calls) == first
    assert summary.skipped == [("a", "m")]


def test_changed_draft_is_processed_again(store):
    store.save("a", DRAFT)
    run(store, ["m"], lambda model: lambda block: block)
    store.save("a", DRAFT + "\nThêm một đoạn mới.\n")
    summary = run(store, ["m"], lambda model: lambda block: block)
    assert summary.done == [("a", "m")]


def test_model_down_leaves_no_result_so_tomorrow_retries(store):
    store.save("a", DRAFT)

    def broken(block):
        raise OllamaError("down")

    summary = run(store, ["m"], lambda model: broken)
    assert summary.failed == [("a", "m")]
    assert store.result("a", "m") is None
    assert run(store, ["m"], lambda model: lambda b: b).done == [("a", "m")]


def test_a_bad_answer_is_rejected_and_the_original_kept(store):
    store.save("a", "Đây là một đoạn văn khá dài để kiểm tra phần đo độ dài.\n")
    run(store, ["m"], lambda model: lambda block: block + " " + "thừa " * 60)
    result = store.result("a", "m")
    assert result["stats"] == {"rejected": 1}
    assert result["edited"] == "Đây là một đoạn văn khá dài để kiểm tra phần đo độ dài.\n"


def test_deadline_stops_the_run(store):
    store.save("a", DRAFT)
    store.save("b", DRAFT)
    ticks = iter([0, 100])
    summary = run(
        store,
        ["m"],
        lambda model: lambda block: block,
        deadline=50,
        clock=lambda: next(ticks),
    )
    assert summary.timed_out
    assert len(summary.done) == 1


def test_end_to_end_against_a_fake_ollama(store, fake_ollama):
    store.save("a", DRAFT)
    fake_ollama.reply = lambda payload: payload["prompt"].replace("đầu tiên", "mở đầu")
    cfg = OllamaConfig(url=fake_ollama.url, model="m:1b", timeout=5)
    summary = run(store, ["m:1b"], lambda model: OllamaEditor(cfg))
    assert summary.done == [("a", "m:1b")]
    assert "mở đầu" in store.result("a", "m:1b")["edited"]
    assert not any("kubectl" in r["prompt"] for r in fake_ollama.generated())


def test_model_names_become_safe_file_names():
    assert model_file("Qwen2.5:7B-Instruct") == "qwen2.5_7b-instruct"
    assert "/" not in model_file("../../etc/passwd")
