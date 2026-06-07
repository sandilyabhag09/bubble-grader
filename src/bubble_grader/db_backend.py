"""Tiny backend abstraction so the same `?`-placeholder SQL works against
either local SQLite (default) or a shared Postgres (when `DATABASE_URL`
is set in `.env`).

Design choices:

* **One placeholder style in the codebase.** All existing call-sites use
  SQLite's `?`. The Postgres path rewrites to `%s` at execute time.
* **One DDL written twice.** ``init_db`` in db.py picks the right CREATE
  TABLE statements via `is_postgres()`. Differences are confined to
  ``AUTOINCREMENT`` vs ``BIGSERIAL`` and ``datetime('now')`` vs ``now()``.
* **Rows always come back as dict-like.** SQLite's `Row` and psycopg's
  `dict_row` both behave that way, so `r["col"]` and `dict(r)` work in
  both backends without per-call branching.
* **lastrowid abstraction.** ``Cursor.lastrowid`` is SQLite-only; on
  Postgres the wrapper appends ``RETURNING id`` to single-table INSERTs
  and fetches the result.
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from .config import DB_PATH


_DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()


def is_postgres() -> bool:
    return bool(_DATABASE_URL)


def database_url() -> str:
    return _DATABASE_URL


# Lazy-import psycopg only when we actually need it so SQLite-only installs
# don't pay the import cost (or fail if psycopg isn't installed yet).
def _pg():
    import psycopg
    from psycopg.rows import dict_row
    return psycopg, dict_row


class _Cursor:
    """Wraps a raw cursor so callers don't care which backend is underneath."""

    def __init__(self, cur, is_pg: bool):
        self._cur = cur
        self._pg = is_pg
        self.lastrowid: int | None = None  # set by execute() for INSERTs

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def rowcount(self) -> int:
        # SQLite and psycopg both expose this — pass it through.
        return self._cur.rowcount

    def __iter__(self) -> Iterator[Any]:
        return iter(self._cur)

    def close(self) -> None:
        self._cur.close()


def _translate(query: str) -> str:
    """Rewrite SQLite-flavored SQL into Postgres-equivalent SQL."""
    # `?` placeholders → `%s`. We do this with a state machine to avoid
    # touching `?` inside single-quoted string literals (none in our SQL
    # today, but cheap insurance against future additions).
    out: list[str] = []
    in_str = False
    for ch in query:
        if ch == "'":
            in_str = not in_str
            out.append(ch)
        elif ch == "?" and not in_str:
            out.append("%s")
        else:
            out.append(ch)
    q = "".join(out)
    # `datetime('now')` → `now()`. Easy global replace; the literal text
    # is only used as a default-value expression in our DDL/UPDATEs.
    q = q.replace("datetime('now')", "now()")
    return q


_INSERT_RE = re.compile(r"^\s*INSERT\s+INTO\s+(\w+)", re.IGNORECASE)


class Connection:
    """Backend-agnostic connection. Use as a context manager."""

    def __init__(self, raw, is_pg: bool):
        self._raw = raw
        self._pg = is_pg

    @property
    def is_postgres(self) -> bool:
        return self._pg

    def execute(self, query: str, args: tuple | list = ()) -> _Cursor:
        if self._pg:
            return self._execute_pg(query, args)
        cur = self._raw.execute(query, args)
        c = _Cursor(cur, is_pg=False)
        c.lastrowid = cur.lastrowid
        return c

    def _execute_pg(self, query: str, args) -> _Cursor:
        q = _translate(query)
        lastrowid = None
        # For single-table INSERTs that lack a RETURNING clause, tack one on
        # so we can expose ``lastrowid`` to the existing callers that rely on
        # it. Schemas where the PK isn't ``id`` aren't covered here — none
        # of our INSERTs hit that case.
        upper = q.lstrip().upper()
        needs_returning = upper.startswith("INSERT") and "RETURNING" not in upper
        if needs_returning:
            # Only append if there's an id column on the target table.
            m = _INSERT_RE.match(q)
            if m and m.group(1).lower() in {"submissions"}:
                q = q.rstrip().rstrip(";") + " RETURNING id"
                cur = self._raw.cursor()
                cur.execute(q, args)
                row = cur.fetchone()
                if row is not None:
                    # dict_row → {"id": 42}
                    lastrowid = row["id"] if isinstance(row, dict) else row[0]
                c = _Cursor(cur, is_pg=True)
                c.lastrowid = lastrowid
                return c
        cur = self._raw.cursor()
        cur.execute(q, args)
        c = _Cursor(cur, is_pg=True)
        return c

    def commit(self) -> None:
        self._raw.commit()

    def close(self) -> None:
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.commit()
        finally:
            self.close()


@contextmanager
def get_conn() -> Iterator[Connection]:
    if is_postgres():
        psycopg, dict_row = _pg()
        raw = psycopg.connect(_DATABASE_URL, row_factory=dict_row)
        try:
            yield Connection(raw, is_pg=True)
            raw.commit()
        finally:
            raw.close()
    else:
        raw = sqlite3.connect(DB_PATH)
        raw.row_factory = sqlite3.Row
        try:
            yield Connection(raw, is_pg=False)
            raw.commit()
        finally:
            raw.close()


# DDL — two versions of every CREATE TABLE so each backend gets the right
# types. Indexes are identical between the two. Kept in this module to make
# the difference easy to audit in one place.

DDL_SQLITE = [
    """
    CREATE TABLE IF NOT EXISTS teachers (
        email TEXT PRIMARY KEY,
        encrypted_credentials BLOB NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS oauth_states (
        state TEXT PRIMARY KEY,
        code_verifier TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tests (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        notes TEXT,
        answer_key_json TEXT,
        scaler_json TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_id TEXT NOT NULL,
        student_id TEXT,
        student_name TEXT,
        student_email TEXT,
        course_id TEXT,
        coursework_id TEXT,
        classroom_submission_id TEXT,
        answers_json TEXT,
        score_json TEXT,
        composite INTEGER,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_submissions_test ON submissions(test_id)",
    "CREATE INDEX IF NOT EXISTS idx_submissions_student ON submissions(student_id)",
    """
    CREATE TABLE IF NOT EXISTS app_assignments (
        course_id TEXT NOT NULL,
        coursework_id TEXT NOT NULL,
        test_id TEXT,
        title TEXT,
        created_by TEXT,
        scope_json TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (course_id, coursework_id)
    )
    """,
    # In-place migration for installs created before scope_json existed.
    # SQLite silently no-ops if the column already exists IFF we use a
    # try/except — bare ALTER fails, so we run it via a guard in init_db.
]

DDL_POSTGRES = [
    """
    CREATE TABLE IF NOT EXISTS teachers (
        email TEXT PRIMARY KEY,
        encrypted_credentials BYTEA NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS oauth_states (
        state TEXT PRIMARY KEY,
        code_verifier TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tests (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        notes TEXT,
        answer_key_json TEXT,
        scaler_json TEXT,
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS submissions (
        id BIGSERIAL PRIMARY KEY,
        test_id TEXT NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
        student_id TEXT,
        student_name TEXT,
        student_email TEXT,
        course_id TEXT,
        coursework_id TEXT,
        classroom_submission_id TEXT,
        answers_json TEXT,
        score_json TEXT,
        composite INTEGER,
        created_at TIMESTAMPTZ DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_submissions_test ON submissions(test_id)",
    "CREATE INDEX IF NOT EXISTS idx_submissions_student ON submissions(student_id)",
    """
    CREATE TABLE IF NOT EXISTS app_assignments (
        course_id TEXT NOT NULL,
        coursework_id TEXT NOT NULL,
        test_id TEXT,
        title TEXT,
        created_by TEXT,
        scope_json TEXT,
        created_at TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (course_id, coursework_id)
    )
    """,
]


def ddl_statements() -> list[str]:
    return DDL_POSTGRES if is_postgres() else DDL_SQLITE
