import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nightshift.compose import OUTLINE_TOKENS, ensure_frame, normalize_brief
from nightshift.composer import Composer
from nightshift.gate import Gate
from nightshift.ollama import OllamaConfig, OllamaError, generate_stream
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
