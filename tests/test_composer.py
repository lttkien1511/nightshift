import io
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nightshift.compose import OUTLINE_TOKENS, ensure_frame, normalize_brief
from nightshift.composer import Composer
from nightshift.gate import Gate
from nightshift.ollama import OllamaConfig, OllamaError, generate_stream, list_models
from nightshift.runner import Busy, Runner, RunnerError
from nightshift.storage import MAX_POSTS, QueueFull, Store

MATERIAL = (
    "Gặp đối tác ở EVN.\n\nChạy `helm uninstall rancher` rồi xem [tài liệu](https://x.dev/d)."
)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.ensure()
    return s


@pytest.fixture
def slug(store):
    brief = normalize_brief({"topic": "Gỡ Rancher", "material": MATERIAL, "avoid": "EVN"})
    return store.create_post("Gỡ Rancher", brief)


def fake_stream(chunks, seen=None, hold=None):
    """A model that answers in pieces, optionally waiting at a gate in the middle."""

    def stream(system, prompt, num_predict, on_text):
        if seen is not None:
            seen.append({"system": system, "prompt": prompt, "num_predict": num_predict})
        out = []
        for index, piece in enumerate(chunks):
            if hold is not None and index == 1:
                hold.wait(5)
            out.append(piece)
            if on_text(piece) is False:
                break
        return "".join(out)

    return stream


def composer_with(store, stream, models=("m1", "m2"), gate=None):
    return Composer(store, lambda model: stream, models, gate or Gate())


# --- storage ------------------------------------------------------------------------------


def test_posts_get_unique_addresses_and_can_be_updated_and_deleted(store):
    brief = normalize_brief({"topic": "x"})
    first = store.create_post("Bài viết", brief)
    second = store.create_post("Bài viết", brief)
    assert (first, second) == ("bai-viet", "bai-viet-2")
    store.update_post(first, stage="outline", outline="- ý")
    assert store.post(first)["stage"] == "outline"
    assert [(p.slug, p.stage) for p in store.posts()] == [
        ("bai-viet", "outline"),
        ("bai-viet-2", "brief"),
    ]
    assert store.delete_post(first) and store.post(first) is None and not store.delete_post(first)
    assert store.post("Not A Slug") is None


def test_there_is_a_limit_on_blog_posts(store):
    brief = normalize_brief({"topic": "x"})
    for index in range(MAX_POSTS):
        store.create_post(f"bài {index}", brief)
    with pytest.raises(QueueFull):
        store.create_post("thêm", brief)


# --- the composer -------------------------------------------------------------------------


def test_an_outline_run_saves_the_outline_and_moves_the_post_on(store, slug):
    seen = []
    composer = composer_with(store, fake_stream(["- ý một\n", "- ý hai"], seen))
    composer.start(slug, "outline")
    composer.wait()
    assert composer.state.state == "done" and composer.state.kind == "outline"
    record = store.post(slug)
    assert record["stage"] == "outline"
    assert record["outline"] == ensure_frame("- ý một\n- ý hai", "vi")
    assert (
        record["outline"].startswith("- Mở bài:")
        and "- ý một\n- ý hai\n- Kết bài:" in record["outline"]
    )
    assert seen[0]["num_predict"] == OUTLINE_TOKENS


def test_a_write_run_saves_the_post_with_checks_and_restores_what_was_protected(store, slug):
    seen = []
    text = [
        "# Gỡ Rancher\n\n",
        "Tôi gặp [[HIDE_1]]. ",
        "Chạy [[CODE_1]] và xem [tài liệu]([[URL_1]]).",
    ]
    composer = composer_with(store, fake_stream(text, seen))
    composer.start(slug, "write", "m2")
    composer.wait()
    record = store.post(slug)
    assert record["stage"] == "done" and record["model"] == "m2"
    assert "Tôi gặp [đã ẩn]." in record["post"]
    assert "`helm uninstall rancher`" in record["post"] and "https://x.dev/d" in record["post"]
    assert isinstance(record["checks"], list)
    assert "EVN" not in seen[0]["prompt"] and "helm uninstall" not in seen[0]["prompt"]


def test_the_page_can_see_the_text_while_it_is_being_written(store, slug):
    hold = threading.Event()
    composer = composer_with(store, fake_stream(["# Tiêu đề\n\n", "Một ", "hai"], hold=hold))
    composer.start(slug, "write")
    for _ in range(100):
        state = composer.state
        if state.chars:
            break
        threading.Event().wait(0.02)
    assert composer.state.running and composer.state.partial.startswith("# Tiêu đề")
    hold.set()
    composer.wait()
    assert composer.state.chars == len("# Tiêu đề\n\nMột hai")


def test_cancel_stops_the_run_and_leaves_the_post_alone(store, slug):
    hold = threading.Event()
    composer = composer_with(store, fake_stream(["a ", "b ", "c "], hold=hold))
    composer.start(slug, "write")
    composer.cancel()
    hold.set()
    composer.wait()
    assert composer.state.state == "cancelled"
    assert store.post(slug)["stage"] == "brief" and store.post(slug)["post"] == ""


def test_a_failing_model_marks_the_run_failed_and_frees_the_gate(store, slug):
    def down(system, prompt, num_predict, on_text):
        raise OllamaError("ollama is unreachable")

    gate = Gate()
    composer = composer_with(store, down, gate=gate)
    composer.start(slug, "write")
    composer.wait()
    assert composer.state.state == "failed" and "unreachable" in composer.state.error
    assert gate.holder is None and store.post(slug)["stage"] == "brief"
    composer.start(slug, "write")
    composer.wait()


def test_an_empty_answer_is_a_failure_not_a_blank_post(store, slug):
    composer = composer_with(store, fake_stream(["   "]))
    composer.start(slug, "write")
    composer.wait()
    assert composer.state.state == "failed" and store.post(slug)["post"] == ""


def test_bad_requests_are_refused_before_anything_starts(store, slug):
    composer = composer_with(store, fake_stream(["x"]))
    with pytest.raises(RunnerError, match="not available"):
        composer.start(slug, "write", "other")
    with pytest.raises(RunnerError, match="no such blog post"):
        composer.start("nope", "write")
    with pytest.raises(RunnerError, match="unknown step"):
        composer.start(slug, "poem")
    assert composer.state is None


def test_one_job_at_a_time_across_editing_and_writing(store, slug):
    gate = Gate()
    hold = threading.Event()
    composer = composer_with(store, fake_stream(["# T\n", "x"], hold=hold), gate=gate)
    composer.start(slug, "write")
    with pytest.raises(Busy, match="being written"):
        composer.start(slug, "outline")
    store.save("draft", "Đoạn ngắn.\n")
    editing = Runner(store, lambda model: lambda block: block, ["m1"], gate=gate)
    with pytest.raises(Busy, match="writing"):
        editing.start("draft")
    hold.set()
    composer.wait()

    block = threading.Event()
    slow = Runner(store, lambda model: lambda b: (block.wait(5), b)[1], ["m1"], gate=gate)
    slow.start("draft")
    with pytest.raises(Busy, match="editing"):
        composer.start(slug, "write")
    block.set()
    slow.wait()
    composer.start(slug, "write")
    composer.wait()
    assert store.post(slug)["stage"] == "done"


# --- streaming from Ollama ----------------------------------------------------------------


class StreamServer:
    def __init__(self, lines, status=200):
        outer = self
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                outer.requests.append(json.loads(self.rfile.read(length)))
                self.send_response(status)
                self.end_headers()
                for line in lines:
                    self.wfile.write((json.dumps(line) + "\n").encode())
                    self.wfile.flush()

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.cfg = OllamaConfig(
            url=f"http://127.0.0.1:{self.httpd.server_address[1]}", model="m", timeout=5
        )
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    started = []

    def make(*args, **kwargs):
        server = StreamServer(*args, **kwargs)
        started.append(server)
        return server

    yield make
    for server in started:
        server.close()


def test_stream_collects_pieces_and_asks_for_a_streamed_answer(serve):
    server = serve([{"response": "Xin "}, {"response": "chào"}, {"response": "", "done": True}])
    pieces = []
    text = generate_stream(server.cfg, "sys", "hi", pieces.append, num_predict=50, num_ctx=8192)
    assert text == "Xin chào" and pieces == ["Xin ", "chào"]
    sent = server.requests[0]
    assert sent["stream"] is True and sent["options"]["num_predict"] == 50
    assert sent["options"]["num_ctx"] == 8192 and sent["system"] == "sys"


def test_stream_stops_when_the_callback_says_so(serve):
    server = serve([{"response": "a"}, {"response": "b"}, {"response": "c", "done": True}])
    text = generate_stream(server.cfg, "s", "p", lambda piece: piece != "b")
    assert text == "ab"


def test_stream_errors_become_ollama_errors(serve):
    with pytest.raises(OllamaError, match="out of memory"):
        generate_stream(serve([{"error": "out of memory"}]).cfg, "s", "p", lambda _: None)
    with pytest.raises(OllamaError, match="empty"):
        generate_stream(serve([{"response": "", "done": True}]).cfg, "s", "p", lambda _: None)
    with pytest.raises(OllamaError, match="HTTP 500"):
        generate_stream(serve([], status=500).cfg, "s", "p", lambda _: None)
    with pytest.raises(OllamaError, match="unreachable"):
        generate_stream(OllamaConfig(url="http://127.0.0.1:1", timeout=1), "s", "p", lambda _: None)


def test_the_live_text_never_shows_raw_tokens_or_hidden_terms(store, slug):
    hold = threading.Event()
    seen = []

    def watching(system, prompt, num_predict, on_text):
        for piece in ["Tôi gặp [[HIDE_1]] và chạy ", "[[CODE_1]] hôm nay. EVN", " lại."]:
            if on_text(piece) is False:
                break
            seen.append(composer.state.partial)
        hold.set()
        return "Tôi gặp [[HIDE_1]] và chạy [[CODE_1]] hôm nay. EVN lại."

    composer = composer_with(store, watching)
    composer.start(slug, "write")
    composer.wait()
    assert seen and all("[[" not in text and "EVN" not in text for text in seen)
    assert "`helm uninstall rancher`" in seen[1] and "[đã ẩn]" in seen[0]


def test_a_written_post_keeps_the_places_that_may_say_more_than_the_material(store, slug):
    claim = "Việc này giúp tôi tiết kiệm thời gian và tăng năng suất."
    text = f"# Gỡ Rancher\n\nTôi chạy [[CODE_1]]. {claim}"
    composer = composer_with(store, fake_stream([text]))
    composer.start(slug, "write")
    composer.wait()
    record = store.post(slug)
    flagged = [record["post"][s:e] for s, e in record["flags"]]
    assert any("tiết kiệm thời gian" in part for part in flagged)
    assert not any("helm" in part for part in flagged)


def test_an_outline_gets_back_the_key_points_the_model_dropped(store):
    brief = normalize_brief({"topic": "Claude", "points": "điểm một\nbỏ ra 600k mua tài khoản"})
    slug = store.create_post("Claude", brief)
    composer = composer_with(store, fake_stream(["- Mở bài: chào\n- Điểm một: a\n- Kết bài: b"]))
    composer.start(slug, "outline")
    composer.wait()
    lines = store.post(slug)["outline"].splitlines()
    assert lines[-2] == "- bỏ ra 600k mua tài khoản" and lines[-1].startswith("- Kết bài")


def test_three_voices_are_stored_without_touching_the_stage(store, slug):
    text = (
        "Tôi từng nghĩ mình chỉ cần một công cụ hỏi đáp thôi.\n---\n"
        "Buổi tối hôm đó tôi ngồi trước màn hình và gõ thử.\n---\n"
        "Hóa ra tôi mất sáu trăm nghìn để biết mình bỏ lỡ nhiều."
    )
    composer = composer_with(store, fake_stream([text]))
    composer.start(slug, "voices")
    composer.wait()
    record = store.post(slug)
    assert len(record["voices"]) == 3 and record["stage"] == "brief" and record["post"] == ""
    assert composer.state.state == "done"


def test_a_model_that_writes_no_usable_voices_fails_the_job(store, slug):
    composer = composer_with(store, fake_stream(["ngắn quá"]))
    composer.start(slug, "voices")
    composer.wait()
    assert composer.state.state == "failed" and "usable samples" in composer.state.error
    assert "voices" not in store.post(slug)


def test_the_last_blog_voice_is_remembered_and_unreadable_files_are_ignored(store):
    assert store.blog_voice() == {}
    store.save_blog_voice({"voice_address": "mình - các bạn", "mode": "close"})
    assert store.blog_voice()["voice_address"] == "mình - các bạn"
    store.blog_voice_path.write_text("not json", encoding="utf-8")
    assert store.blog_voice() == {}


def test_the_written_post_loses_copied_headings_and_repeated_sentences(store, slug):
    text = (
        "# Tiêu đề\n\n## Mở bài: chào bạn\n\nMình nghĩ chắc sẽ gỡ nhầm file thôi. "
        "Mình nghĩ chắc sẽ gỡ nhầm file thôi.\n"
    )
    composer = composer_with(store, fake_stream([text]))
    composer.start(slug, "write")
    composer.wait()
    record = store.post(slug)
    assert "Mở bài" not in record["post"] and record["post"].count("gỡ nhầm") == 1
    assert any("1 repeated sentence" in c["text"] for c in record["checks"])


def test_a_post_cut_off_by_the_token_cap_loses_its_last_half_sentence_and_says_so(store, slug):
    composer = composer_with(store, fake_stream(["# Tiêu đề\n\nĐoạn đầu đã xong. Đoạn dở"]))
    composer.start(slug, "write")
    composer.wait()
    record = store.post(slug)
    assert record["post"].rstrip().endswith("đã xong.")
    assert any("ran out of room" in c["text"] for c in record["checks"])


OUT1 = "- Mở bài: một\n- Phần: hai\n- Kết bài: ba"
OUT2 = "- Mở bài: khác\n- Phần: nữa\n- Kết bài: hết"
THREE_VOICES = (
    "Một đoạn mở bài theo giọng tâm sự với bạn thân.\n---\n"
    "Buổi tối hôm đó tôi ngồi một mình trước màn hình máy tính.\n---\n"
    "Hóa ra mọi chuyện buồn cười hơn tôi tưởng rất nhiều."
)


def composer_for(store, answers, gate=None):
    """Each model gives the text set for it; a model set to an exception fails with it."""

    def make(model):
        answer = answers[model]
        if isinstance(answer, Exception):

            def broken(system, prompt, num_predict, on_text):
                raise answer

            return broken
        return fake_stream([answer])

    return Composer(store, make, list(answers), gate or Gate())


def test_several_models_leave_their_answers_as_candidates_and_the_first_fills_the_outline(
    store, slug
):
    composer = composer_for(store, {"m1": OUT1, "m2": OUT2})
    composer.start(slug, "outline", models=["m1", "m2"], replace=False)
    composer.wait()
    record = store.post(slug)
    assert [c["model"] for c in record["candidates"]["outline"]] == ["m1", "m2"]
    assert "một" in record["outline"] and "khác" not in record["outline"]
    assert record["stage"] == "outline"
    assert composer.state.state == "done" and composer.state.queue == ("m1", "m2")
    assert all(c["id"] and c["seconds"] >= 0 for c in record["candidates"]["outline"])


def test_a_new_answer_replaces_the_outline_only_when_asked_to(store, slug):
    composer = composer_for(store, {"m1": OUT1, "m2": OUT2})
    store.update_post(slug, outline="- Mở bài: của tôi")
    composer.start(slug, "outline", model="m2", replace=False)
    composer.wait()
    assert store.post(slug)["outline"] == "- Mở bài: của tôi"
    assert len(store.post(slug)["candidates"]["outline"]) == 1
    composer.start(slug, "outline", model="m2")
    composer.wait()
    assert "khác" in store.post(slug)["outline"]


def test_a_failing_model_does_not_stop_the_others(store, slug):
    composer = composer_for(store, {"m1": RuntimeError("boom"), "m2": OUT2})
    composer.start(slug, "outline", models=["m1", "m2"], replace=False)
    composer.wait()
    state = composer.state
    assert state.state == "done" and "m1" in state.error and "boom" in state.error
    assert [c["model"] for c in store.post(slug)["candidates"]["outline"]] == ["m2"]


def test_when_every_model_fails_the_job_fails_and_says_why(store, slug):
    composer = composer_for(store, {"m1": RuntimeError("aaa"), "m2": RuntimeError("bbb")})
    composer.start(slug, "outline", models=["m1", "m2"], replace=False)
    composer.wait()
    assert composer.state.state == "failed"
    assert "aaa" in composer.state.error and "bbb" in composer.state.error
    assert not store.post(slug).get("candidates")


def test_cancelling_stops_the_rest_of_the_queue_but_keeps_what_was_finished(store, slug):
    started, release = threading.Event(), threading.Event()

    def make(model):
        if model == "m1":
            return fake_stream([OUT1])

        def slow(system, prompt, num_predict, on_text):
            on_text("partial")
            started.set()
            release.wait(5)
            return "partial"

        return slow

    composer = Composer(store, make, ["m1", "m2", "m3"], Gate())
    composer.start(slug, "outline", models=["m1", "m2", "m3"], replace=False)
    assert started.wait(5)
    composer.cancel()
    release.set()
    composer.wait()
    assert composer.state.state == "cancelled"
    assert [c["model"] for c in store.post(slug)["candidates"]["outline"]] == ["m1"]


def test_only_the_newest_answers_are_kept(store, slug):
    composer = composer_for(store, {"m1": OUT1})
    for _ in range(14):
        composer.start(slug, "outline", model="m1", replace=False)
        composer.wait()
    assert len(store.post(slug)["candidates"]["outline"]) == 12


def test_too_many_or_unknown_models_are_refused_before_anything_starts(store, slug):
    composer = composer_for(store, {name: OUT1 for name in "abcde"})
    with pytest.raises(RunnerError):
        composer.start(slug, "outline", models=list("abcde"))
    with pytest.raises(RunnerError):
        composer.start(slug, "outline", models=["zzz"])
    assert composer.state is None


def test_the_list_of_models_may_be_a_function_that_changes(store):
    names = ["m1"]
    composer = Composer(store, lambda model: fake_stream([OUT1]), lambda: names, Gate())
    assert composer.models == ["m1"]
    names.append("m2")
    assert composer.models == ["m1", "m2"]


def test_each_written_candidate_carries_its_own_checks(store, slug):
    composer = composer_for(
        store, {"m1": "# T\n\nGiảm 450 MB hẳn rồi.", "m2": "# T\n\nGỡ xong thôi."}
    )
    composer.start(slug, "write", models=["m1", "m2"], replace=False)
    composer.wait()
    record = store.post(slug)
    first, second = record["candidates"]["write"]
    assert record["post"].startswith("# T") and "450" in record["post"]
    assert any("450" in c["text"] for c in first["checks"])
    assert not any("450" in c["text"] for c in second["checks"])
    assert any("450" in c["text"] for c in record["checks"])


def test_voices_use_one_model_even_when_more_are_ticked(store, slug):
    composer = composer_for(store, {"m1": THREE_VOICES, "m2": THREE_VOICES})
    composer.start(slug, "voices", models=["m1", "m2"])
    composer.wait()
    assert composer.state.queue == ("m1",) and len(store.post(slug)["voices"]) == 3


def test_stream_sends_the_thinking_switch_only_when_asked(serve):
    server = serve([{"response": "ok", "done": True}])
    generate_stream(server.cfg, "s", "p", lambda _piece: True)
    generate_stream(server.cfg, "s", "p", lambda _piece: True, think=False)
    assert "think" not in server.requests[0] and server.requests[1]["think"] is False


class _Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return io.BytesIO(self.body)

    def __exit__(self, *exc):
        return False


def test_the_models_ollama_has_are_listed_and_a_silent_ollama_gives_nothing(monkeypatch):
    body = json.dumps(
        {"models": [{"name": "qwen3.5:9b"}, {"name": "nomic-embed-text"}, {"name": "gemma3:12b"}]}
    ).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: _Reply(body))
    assert list_models("http://x") == ["gemma3:12b", "qwen3.5:9b"]

    def refuse(url, timeout):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert list_models("http://x") == []
