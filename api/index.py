"""Vercel entrypoint: exposes the FastAPI server as an ASGI function.

Vercel's Python runtime discovers the module-level ``app`` object. The
package lives under src/, which isn't pip-installed on Vercel, so put it on
the path first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bubble_grader.server import app  # noqa: E402,F401
