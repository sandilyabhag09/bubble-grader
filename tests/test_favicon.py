from pathlib import Path

from bubble_grader import server

STATIC = Path(server.__file__).parent / "static"


def test_favicon_assets_exist():
    for name in ["favicon.svg", "favicon-32.png", "favicon-192.png", "apple-touch-icon.png", "favicon.ico"]:
        assert (STATIC / name).exists(), name


def test_favicon_linked_and_served():
    head = (Path(server.__file__).parent / "templates" / "base.html").read_text()
    assert 'href="/static/favicon.svg"' in head and 'rel="apple-touch-icon"' in head
    paths = {getattr(r, "path", None) for r in server.app.routes}
    assert "/favicon.ico" in paths and "/static" in paths


def test_privacy_page_is_public():
    from starlette.testclient import TestClient
    with TestClient(server.app) as client:
        r = client.get("/privacy")
    assert r.status_code == 200
    assert "Limited Use" in r.text and "Google Classroom" in r.text
