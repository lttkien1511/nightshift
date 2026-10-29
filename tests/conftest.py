import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakeOllama:
    """A local HTTP server that answers /api/generate like Ollama, with settable behaviour."""

    def __init__(self):
        self.requests: list[dict] = []
        self.reply = lambda payload: payload["prompt"]
        self.status = 200
        self.raw: bytes | None = None
        self.delay = 0.0
        self.installed: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                outer.requests.append(payload)
                time.sleep(outer.delay)
                if outer.raw is not None:
                    body = outer.raw
                else:
                    body = json.dumps({"response": outer.reply(payload)}).encode()
                self.send_response(outer.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                body = json.dumps({"models": outer.installed}).encode()
                self.send_response(outer.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def generated(self) -> list[dict]:
        """Requests that asked for text, not the ones that only unload a model."""
        return [r for r in self.requests if "prompt" in r]


@pytest.fixture
def fake_ollama():
    server = FakeOllama()
    yield server
    server.server.shutdown()
    server.server.server_close()
