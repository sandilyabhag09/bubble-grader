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

# Sign-in allowlist: comma-separated Google emails. Empty = anyone can sign in
# (fine for a laptop; set it on any public deployment). Non-listed accounts are
# turned away at sign-in and their credentials are never stored.
ALLOWED_TEACHERS = {
    e.strip().lower()
    for e in (os.environ.get("ALLOWED_TEACHERS") or "").split(",")
    if e.strip()
}

# Anthropic API key for the in-app answer-key PDF import (Tests page). This is
# a standard console.anthropic.com API key — costs pennies per imported test.
# Unset = the import feature politely says it isn't configured.
ANTHROPIC_API_KEY = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()

# Local-only auto-grader: when truthy, a background poller grades work as it's
# turned in (see auto_grade.py). Off by default; intended for a local machine.
AUTO_GRADE_ON_TURNIN = os.environ.get("AUTO_GRADE_ON_TURNIN", "").strip().lower() in (
    "1", "true", "yes", "on",
)
AUTO_GRADE_POLL_SECONDS = int(os.environ.get("AUTO_GRADE_POLL_SECONDS") or "60")

# Serverless auto-grading: when set, GET /cron/auto-grade?token=<this> runs one
# bounded grade-everything-new pass. Point an external pinger (UptimeRobot) at
# it every few minutes and turn-ins grade themselves; it doubles as the
# keep-warm ping. Unset = endpoint disabled.
AUTO_GRADE_TOKEN = (os.environ.get("AUTO_GRADE_TOKEN") or "").strip()

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
    # Return-in-Classroom: upload the marked-up sheet (drive.file = only files
    # this app creates) and post it to the student as a private announcement.
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/classroom.announcements",
    "https://www.googleapis.com/auth/gmail.send",
]
