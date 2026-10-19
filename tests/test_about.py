"""Public About page: renders, shows only made-up example images, never touches the DB."""

import re
from pathlib import Path

from PIL import Image
from starlette.testclient import TestClient

import bubble_grader.db as dbmod
import bubble_grader.db_backend as dbb
import bubble_grader.server as server

STATIC = Path(server.__file__).parent / "static"


def _forbid_db(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the About page must not touch the database")
    monkeypatch.setattr(dbmod, "get_conn", boom)
    monkeypatch.setattr(dbb, "get_conn", boom)


def test_about_page_renders_without_the_database(monkeypatch):
    monkeypatch.setattr(server, "IS_VERCEL", True)
    _forbid_db(monkeypatch)
    with TestClient(server.app) as client:
        r = client.get("/about")
    assert r.status_code == 200
    html = r.text
    assert "<h1>" in html and 'href="/privacy"' in html
    assert "Not affiliated with or endorsed by ACT" in html
    assert "/static/about/sheet-after.webp" in html and "/static/about/sheet-before.webp" in html


def test_example_images_exist_match_their_declared_size_and_have_alt_text(monkeypatch):
    with TestClient(server.app) as client:
        html = client.get("/about").text
    imgs = re.findall(r"<img\b[^>]*>", html)
    assert imgs
    for tag in imgs:
        src = re.search(r'src="/static/([^"]+)"', tag).group(1)
        alt = re.search(r'alt="([^"]*)"', tag).group(1)
        w, h = (int(re.search(rf'{a}="(\d+)"', tag).group(1)) for a in ("width", "height"))
        path = STATIC / src
        assert path.exists(), src
        assert path.stat().st_size < 300 * 1024, f"{src} is too heavy for a page image"
        assert Image.open(path).size == (w, h), src
        assert len(alt) > 20, f"{src} needs descriptive alt text"
        with TestClient(server.app) as client:
            assert client.get(f"/static/{src}").status_code == 200


def test_about_is_linked_from_every_page_header_and_the_home_page():
    with TestClient(server.app) as client:
        for path in ("/", "/privacy", "/about"):
            assert 'href="/about"' in client.get(path).text, path
