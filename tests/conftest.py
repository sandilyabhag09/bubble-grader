"""Test environment: NEVER touch the shared Neon DB or real secrets.

These lines must run before any bubble_grader import — python-dotenv does not
override variables that already exist in the environment, so presetting them
here beats whatever a developer's .env says.
"""

import os

os.environ["DATABASE_URL"] = ""          # force the SQLite fallback
os.environ.setdefault("FERNET_KEY", "test-key-not-a-secret")
os.environ["ALLOWED_TEACHERS"] = ""
os.environ.pop("VERCEL", None)

import pytest  # noqa: E402


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """A fresh throwaway SQLite database per test."""
    import bubble_grader.db_backend as dbb
    import bubble_grader.db as dbmod

    monkeypatch.setattr(dbb, "DB_PATH", tmp_path / "test.db")
    dbmod.init_db()
    return dbmod
