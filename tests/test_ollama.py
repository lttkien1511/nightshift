import pytest

from nightshift.ollama import (
    SYSTEM_PROMPT,
    OllamaConfig,
    OllamaEditor,
    OllamaError,
    build_system,
    generate,
    unload,
)


def cfg(server, **kwargs):
    return OllamaConfig(url=server.url, model="m:1b", timeout=kwargs.pop("timeout", 5), **kwargs)


def test_generate_sends_model_system_and_options(fake_ollama):
    fake_ollama.reply = lambda payload: "xin chào"
    assert generate(cfg(fake_ollama, num_ctx=2048), "sys", "hi") == "xin chào"
    sent = fake_ollama.requests[0]
    assert sent["model"] == "m:1b"
    assert sent["system"] == "sys"
    assert sent["prompt"] == "hi"
    assert sent["stream"] is False
    assert sent["options"]["num_ctx"] == 2048


def test_http_error_becomes_ollama_error(fake_ollama):
    fake_ollama.status = 500
    with pytest.raises(OllamaError, match="HTTP 500"):
        generate(cfg(fake_ollama), "s", "p")


def test_non_json_reply_is_rejected(fake_ollama):
    fake_ollama.raw = b"<html>nope</html>"
    with pytest.raises(OllamaError, match="not JSON"):
        generate(cfg(fake_ollama), "s", "p")


def test_empty_answer_is_rejected(fake_ollama):
    fake_ollama.reply = lambda payload: "   "
    with pytest.raises(OllamaError, match="empty"):
        generate(cfg(fake_ollama), "s", "p")


def test_slow_server_times_out(fake_ollama):
    fake_ollama.delay = 1.0
    with pytest.raises(OllamaError, match="unreachable or too slow"):
        generate(cfg(fake_ollama, timeout=0.2), "s", "p")


def test_unreachable_server_is_an_ollama_error():
    dead = OllamaConfig(url="http://127.0.0.1:1", timeout=1)
    with pytest.raises(OllamaError):
        generate(dead, "s", "p")


def test_unload_asks_for_keep_alive_zero_and_never_raises(fake_ollama):
    unload(cfg(fake_ollama))
    assert fake_ollama.requests[0]["keep_alive"] == 0
    unload(OllamaConfig(url="http://127.0.0.1:1", timeout=1))


def test_voice_samples_are_appended_to_the_system_prompt():
    assert build_system("") == SYSTEM_PROMPT
    assert build_system("  Tôi viết ngắn.  ").endswith("Tôi viết ngắn.")


def test_editor_returns_the_model_answer(fake_ollama):
    fake_ollama.reply = lambda payload: "đã sửa"
    editor = OllamaEditor(cfg(fake_ollama), voice="giọng của tôi")
    assert editor("bản gốc") == "đã sửa"
    assert "giọng của tôi" in fake_ollama.requests[0]["system"]
