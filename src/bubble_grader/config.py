"""Project paths, env loading, and OAuth scopes."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Serverless detection: Vercel sets VERCEL=1. There we must not start
# background threads, and the filesystem is read-only outside /tmp.
IS_VERCEL = bool(os.environ.get("VERCEL"))

CLIENT_SECRET_PATH = PROJECT_ROOT / "secrets" / "client_secret.json"
DATA_DIR = PROJECT_ROOT / "data"
try:
    DATA_DIR.mkdir(exist_ok=True)
except OSError:
    pass  # read-only deploy bundle — the dir ships with the repo anyway
DB_PATH = DATA_DIR / "bubble_grader.db"

FERNET_KEY = os.environ.get("FERNET_KEY", "").encode() or None
# A missing FERNET_KEY is non-fatal at import time so `bubble-grader setup`
# can generate one on first run. Callers that actually need to encrypt/decrypt
# (server.SessionMiddleware, db.store_credentials) check at use-time and raise
# a friendly "run setup first" message.

OAUTH_REDIRECT_URI = os.environ.get(
    "OAUTH_REDIRECT_URI", "http://localhost:8765/oauth/callback"
)
SERVER_PORT = int(os.environ.get("SERVER_PORT") or "8765")

# Local-only auto-grader: when truthy, a background poller grades work as it's
# turned in (see auto_grade.py). Off by default; intended for a local machine.
AUTO_GRADE_ON_TURNIN = os.environ.get("AUTO_GRADE_ON_TURNIN", "").strip().lower() in (
    "1", "true", "yes", "on",
)
AUTO_GRADE_POLL_SECONDS = int(os.environ.get("AUTO_GRADE_POLL_SECONDS") or "60")

# Order matters less than completeness. Strings must match exactly what's
# registered in the Google Auth Platform "Data Access" page.
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.rosters.readonly",
    "https://www.googleapis.com/auth/classroom.coursework.students",
    "https://www.googleapis.com/auth/classroom.profile.emails",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]
