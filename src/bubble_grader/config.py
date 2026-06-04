"""Project paths, env loading, and OAuth scopes."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

CLIENT_SECRET_PATH = PROJECT_ROOT / "secrets" / "client_secret.json"
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "bubble_grader.db"

FERNET_KEY = os.environ.get("FERNET_KEY", "").encode() or None
# A missing FERNET_KEY is non-fatal at import time so `bubble-grader setup`
# can generate one on first run. Callers that actually need to encrypt/decrypt
# (server.SessionMiddleware, db.store_credentials) check at use-time and raise
# a friendly "run setup first" message.

OAUTH_REDIRECT_URI = os.environ.get(
    "OAUTH_REDIRECT_URI", "http://localhost:8765/oauth/callback"
)
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8765"))

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
