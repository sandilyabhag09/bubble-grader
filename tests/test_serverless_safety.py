"""Guards for the constraints serverless hosting imposes — each encodes a
production incident (libGL crash, slow cold starts, relative-path failures)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_server_imports_without_heavy_libraries():
    """Importing the web server must not pull cv2/PIL/pypdfium2/pytesseract —
    they are lazy-loaded by grading paths so cold starts stay fast."""
    code = """
import os, sys
os.environ["DATABASE_URL"] = ""
os.environ.setdefault("FERNET_KEY", "test-key")
class Block:
    HEAVY = {"cv2", "PIL", "pypdfium2", "pytesseract"}
    def find_spec(self, name, *a, **k):
        if name.split(".")[0] in self.HEAVY:
            raise ImportError(f"heavy module {name!r} imported at server import time")
sys.meta_path.insert(0, Block())
import bubble_grader.server  # noqa: F401
print("light-import-ok")
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT
    )
    assert out.returncode == 0, out.stderr
    assert "light-import-ok" in out.stdout


def test_opencv_is_headless_everywhere():
    """Full opencv-python needs libGL, which servers don't have (this took
    down grading on the first deploy). Headless only, in both dep files."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "opencv-python-headless" in pyproject
    assert '"opencv-python>' not in pyproject
    reqs = (ROOT / "requirements.txt").read_text()
    assert "opencv-python-headless" in reqs
    assert "\nopencv-python=" not in reqs


def test_vercel_entrypoint_exists():
    assert (ROOT / "api" / "index.py").exists()
    assert (ROOT / "vercel.json").exists()


def test_sheet_paths_are_absolute():
    """Templates must resolve independent of the process working directory
    (serverless functions don't run from the repo root)."""
    from bubble_grader.submissions import DEFAULT_SHEETS_DIR
    from bubble_grader.feedback import DEFAULT_TEMPLATE, DEFAULT_REFERENCE
    assert DEFAULT_SHEETS_DIR.is_absolute()
    assert DEFAULT_TEMPLATE.is_absolute() and DEFAULT_REFERENCE.is_absolute()
    assert DEFAULT_TEMPLATE.exists() and DEFAULT_REFERENCE.exists()


def test_deep_health_reports_grading_dependencies():
    """/healthz/deep must load the lazily-imported grading stack and find every
    sheet template — it is how a deploy proves its bundle is complete."""
    from starlette.testclient import TestClient
    import bubble_grader.server as server
    from bubble_grader.submissions import SHEET_LAYOUTS
    with TestClient(server.app) as client:
        r = client.get("/healthz/deep")
    body = r.json()
    assert r.status_code == 200 and body["ok"] is True, body
    assert body["checks"]["import:cv2"] is True
    for stem in SHEET_LAYOUTS:
        assert body["checks"][f"sheet:{stem}"] is True


def test_vercel_bundle_is_scoped():
    """Bundling '**' pulled the build cache's virtualenv in and blew the 500 MB
    function limit; the bundle must name what it needs and exclude the venv."""
    import json
    from bubble_grader.config import PROJECT_ROOT
    fn = json.loads((PROJECT_ROOT / "vercel.json").read_text())["functions"]["api/index.py"]
    assert fn["includeFiles"] != "**"
    for needed in ("src/**", "data/sheets/**"):
        assert needed in fn["includeFiles"]
    assert ".venv/**" in fn["excludeFiles"]
