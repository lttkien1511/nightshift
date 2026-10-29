import pytest
from fastapi.testclient import TestClient

from nightshift.storage import Store
from nightshift.web import create_app


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path, max_bytes=200, max_queue=2)


@pytest.fixture
def client(store):
    return TestClient(create_app(store))


def upload(client, name="Gỡ Rancher", text="Xin chào", **kwargs):
    return client.post(
        "/drafts", data={"name": name, "text": text}, follow_redirects=False, **kwargs
    )


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_upload_queues_the_draft_and_redirects_home(client, store):
    response = upload(client)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert store.read("go-rancher") == "Xin chào"
    assert client.get("/drafts").json() == [
        {"slug": "go-rancher", "size": len("Xin chào".encode()), "has_results": False}
    ]


def test_bad_uploads_get_the_right_status(client):
    assert upload(client, name="???").status_code == 400
    assert upload(client, text="  ").status_code == 400
    assert upload(client, text="a" * 201).status_code == 413
    assert upload(client, name="a").status_code == 303
    assert upload(client, name="b").status_code == 303
    assert upload(client, name="c").status_code == 429


def test_oversized_request_is_refused_before_it_is_parsed(client):
    assert upload(client, text="a" * 20_000).status_code == 413


def test_index_lists_drafts_and_escapes_names(client, store):
    upload(client, name="<script>alert(1)</script>")
    page = client.get("/")
    assert page.status_code == 200
    assert "<script>" not in page.text
    assert "script-alert-1-script" in page.text
    assert "default-src 'none'" in page.headers["content-security-policy"]


def test_draft_page_escapes_the_text(client):
    upload(client, name="x", text="<img src=x onerror=alert(1)> & more")
    page = client.get("/drafts/x")
    assert page.status_code == 200
    assert "<img" not in page.text
    assert "&lt;img src=x onerror=alert(1)&gt; &amp; more" in page.text


def test_unknown_or_malicious_draft_names_are_404(client):
    assert client.get("/drafts/missing").status_code == 404
    assert client.get("/drafts/..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/drafts/Not-A-Slug").status_code == 404


def test_delete(client, store):
    upload(client, name="x")
    response = client.post("/drafts/x/delete", follow_redirects=False)
    assert response.status_code == 303
    assert store.read("x") is None
    assert client.post("/drafts/x/delete").status_code == 404


def test_cross_site_posts_are_refused(client, store):
    evil = {"origin": "https://evil.example"}
    assert upload(client, headers=evil).status_code == 403
    assert upload(client, headers={"origin": "null"}).status_code == 403
    assert upload(client, headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert store.drafts() == []
    assert upload(client, headers={"origin": "http://testserver"}).status_code == 303


def test_security_headers_are_set_everywhere(client):
    headers = client.get("/healthz").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"
