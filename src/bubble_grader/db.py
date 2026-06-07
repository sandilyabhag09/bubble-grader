"""Credential + grading store.

Backed by SQLite by default. If the ``DATABASE_URL`` environment variable
is set, talks to that Postgres instance instead, sharing all state across
machines (teacher + admin). See ``db_backend.py`` for the abstraction.
"""

import json

from cryptography.fernet import Fernet

from .config import FERNET_KEY
from .db_backend import get_conn, ddl_statements, is_postgres


def init_db() -> None:
    """Create tables + indexes if missing. Safe to call repeatedly."""
    with get_conn() as conn:
        for stmt in ddl_statements():
            conn.execute(stmt)
        _migrate_add_scope_column(conn)


def _migrate_add_scope_column(conn) -> None:
    """Older installs created app_assignments without scope_json.
    Add it in place if it's missing — both backends silently no-op when
    the column already exists, so this is safe to run on every startup.
    """
    if conn.is_postgres:
        sql = (
            "ALTER TABLE app_assignments "
            "ADD COLUMN IF NOT EXISTS scope_json TEXT"
        )
        conn.execute(sql)
        return
    # SQLite: no IF NOT EXISTS support for ALTER ADD COLUMN, so introspect
    # the current schema and skip if the column is already there.
    cur = conn.execute("PRAGMA table_info(app_assignments)")
    cols = {row["name"] if hasattr(row, "keys") else row[1] for row in cur.fetchall()}
    if "scope_json" not in cols:
        conn.execute("ALTER TABLE app_assignments ADD COLUMN scope_json TEXT")


def _fernet() -> Fernet:
    if not FERNET_KEY:
        raise RuntimeError(
            "FERNET_KEY missing — run `uv run bubble-grader setup` to generate one."
        )
    return Fernet(FERNET_KEY)


def store_credentials(email: str, credentials_dict: dict) -> None:
    blob = _fernet().encrypt(json.dumps(credentials_dict).encode())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO teachers (email, encrypted_credentials)
            VALUES (?, ?)
            ON CONFLICT(email) DO UPDATE SET
                encrypted_credentials = excluded.encrypted_credentials,
                updated_at = datetime('now')
            """,
            (email, blob),
        )


def load_credentials(email: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT encrypted_credentials FROM teachers WHERE email = ?",
            (email,),
        ).fetchone()
    if not row:
        return None
    return json.loads(_fernet().decrypt(row["encrypted_credentials"]).decode())


def list_teachers() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT email FROM teachers ORDER BY email"
        ).fetchall()
    return [r["email"] for r in rows]


### Tests (ACT practice tests + their keys + scalers) ----------------------

def add_test(test_id: str, name: str, notes: str | None = None) -> None:
    """Register a new test. Raises if test_id already exists."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO tests (id, name, notes) VALUES (?, ?, ?)",
            (test_id, name, notes),
        )


def upsert_test(test_id: str, name: str, notes: str | None = None) -> None:
    """Register or update a test's display name + notes."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO tests (id, name, notes) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                notes = COALESCE(excluded.notes, tests.notes),
                updated_at = datetime('now')
            """,
            (test_id, name, notes),
        )


def list_tests() -> list[dict]:
    """All tests with metadata. Also tags each test as 'new'/'legacy' format
    based on the English section's question count — new-format ACT has
    ≤ 50 English questions, legacy has 75. Test 1 (English) is the most
    reliable section to key off because its count differs most between
    the two formats and is always present.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, name, notes, answer_key_json, scaler_json,
                   answer_key_json IS NOT NULL AS has_key,
                   scaler_json IS NOT NULL AS has_scaler,
                   created_at, updated_at
            FROM tests ORDER BY id
            """
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        akj = d.pop("answer_key_json", None)
        scj = d.pop("scaler_json", None)
        d["format"] = None
        d["key_count"] = None
        if akj:
            try:
                key = json.loads(akj)
                eng = key.get("Test 1") or {}
                d["key_count"] = sum(len(v) for v in key.values() if isinstance(v, dict))
                # 50 or fewer scored English questions = new format
                d["format"] = "new" if len(eng) <= 50 else "legacy"
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        if scj:
            try:
                sc = json.loads(scj)
                d["scaler_count"] = sum(len(v) for v in sc.values() if isinstance(v, dict))
            except (json.JSONDecodeError, TypeError, AttributeError):
                d["scaler_count"] = None
        else:
            d["scaler_count"] = None
        out.append(d)
    return out


def get_test(test_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM tests WHERE id = ?", (test_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["answer_key"] = json.loads(d.pop("answer_key_json")) if d["answer_key_json"] else None
    d["scaler"] = json.loads(d.pop("scaler_json")) if d["scaler_json"] else None
    return d


def set_test_answer_key(test_id: str, answer_key: dict) -> None:
    """Replace the answer key for a test. answer_key is the inner {section: {q: opt}} dict."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE tests SET answer_key_json = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (json.dumps(answer_key), test_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"No test with id={test_id!r}. Use `test add` first.")


def set_test_scaler(test_id: str, scaler: dict) -> None:
    """Replace the scaler for a test. scaler is the {section: {raw: scaled}} dict."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE tests SET scaler_json = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (json.dumps(scaler), test_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"No test with id={test_id!r}. Use `test add` first.")


### App-owned Classroom assignments (those we can grade-write to) -----------

def record_app_assignment(
    course_id: str,
    coursework_id: str,
    *,
    test_id: str | None = None,
    title: str | None = None,
    created_by: str | None = None,
    scope: dict | None = None,
) -> None:
    """Insert or upsert an app-owned assignment.

    ``scope`` is the partial/full-scope config, serialized to JSON and stored
    in ``scope_json``. ``None`` means "full scope" — same as today's behavior.
    The COALESCE update only overwrites fields the caller actually supplied,
    so passing only ``scope=...`` on re-record won't clobber the title.
    """
    scope_json = json.dumps(scope) if scope is not None else None
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO app_assignments (
                course_id, coursework_id, test_id, title, created_by, scope_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(course_id, coursework_id) DO UPDATE SET
                test_id    = COALESCE(excluded.test_id, app_assignments.test_id),
                title      = COALESCE(excluded.title, app_assignments.title),
                scope_json = COALESCE(excluded.scope_json, app_assignments.scope_json)
            """,
            (course_id, coursework_id, test_id, title, created_by, scope_json),
        )


def get_app_assignment(course_id: str, coursework_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM app_assignments WHERE course_id=? AND coursework_id=?",
            (course_id, coursework_id),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    sj = d.pop("scope_json", None)
    try:
        d["scope"] = json.loads(sj) if sj else None
    except (json.JSONDecodeError, TypeError):
        d["scope"] = None
    return d


def delete_app_assignment(
    course_id: str, coursework_id: str, *, cascade_submissions: bool = True
) -> int:
    """Remove an assignment from our tracking + its submissions. Returns rows deleted."""
    with get_conn() as conn:
        n = 0
        if cascade_submissions:
            n += conn.execute(
                "DELETE FROM submissions WHERE course_id = ? AND coursework_id = ?",
                (course_id, coursework_id),
            ).rowcount
        n += conn.execute(
            "DELETE FROM app_assignments WHERE course_id = ? AND coursework_id = ?",
            (course_id, coursework_id),
        ).rowcount
        return n


def delete_app_assignment(
    course_id: str, coursework_id: str, *, cascade_submissions: bool = True
) -> int:
    """Remove an assignment from app_assignments + (optionally) its submission rows."""
    with get_conn() as conn:
        n = 0
        if cascade_submissions:
            n += conn.execute(
                "DELETE FROM submissions WHERE course_id = ? AND coursework_id = ?",
                (course_id, coursework_id),
            ).rowcount
        n += conn.execute(
            "DELETE FROM app_assignments WHERE course_id = ? AND coursework_id = ?",
            (course_id, coursework_id),
        ).rowcount
        return n


def list_app_assignments(course_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM app_assignments WHERE course_id=?",
            (course_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_test(test_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM tests WHERE id = ?", (test_id,))
    return cur.rowcount > 0


### Submissions (graded student results) ------------------------------------

def add_submission(
    test_id: str,
    answers: dict[int, str],
    score: dict,
    *,
    student_id: str | None = None,
    student_name: str | None = None,
    student_email: str | None = None,
    course_id: str | None = None,
    coursework_id: str | None = None,
    classroom_submission_id: str | None = None,
) -> int:
    """Store a graded submission; return its rowid."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO submissions (
                test_id, student_id, student_name, student_email,
                course_id, coursework_id, classroom_submission_id,
                answers_json, score_json, composite
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                test_id, student_id, student_name, student_email,
                course_id, coursework_id, classroom_submission_id,
                json.dumps({str(k): v for k, v in answers.items()}),
                json.dumps(score),
                score.get("composite"),
            ),
        )
        return int(cur.lastrowid)


def list_submissions(
    test_id: str | None = None,
    student_id: str | None = None,
    course_id: str | None = None,
    coursework_id: str | None = None,
    *,
    include_score: bool = False,
) -> list[dict]:
    """List submissions filtered by any combination of the provided keys.

    When ``include_score=True``, also fetches and parses the per-section
    ``score_json`` blob into a ``score`` dict on each row. This is the
    full ``full_grade`` output: ``{"sections": {"Test 1": {...}, ...},
    "composite": int}``. Use this for roster averages where you need
    raw + scaled per section.
    """
    cols = (
        "id, test_id, student_id, student_name, student_email, "
        "course_id, coursework_id, classroom_submission_id, composite, created_at"
    )
    if include_score:
        cols += ", score_json"
    sql = f"SELECT {cols} FROM submissions"
    args: list = []
    where: list[str] = []
    if test_id is not None:
        where.append("test_id = ?")
        args.append(test_id)
    if student_id is not None:
        where.append("student_id = ?")
        args.append(student_id)
    if course_id is not None:
        where.append("course_id = ?")
        args.append(course_id)
    if coursework_id is not None:
        where.append("coursework_id = ?")
        args.append(coursework_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC"
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
    if include_score:
        for r in rows:
            sj = r.pop("score_json", None)
            try:
                r["score"] = json.loads(sj) if sj else None
            except (json.JSONDecodeError, TypeError):
                r["score"] = None
    return rows


def update_submission(
    submission_id: int,
    *,
    answers: dict[int, str] | None = None,
    score: dict | None = None,
) -> None:
    """Patch specific fields on a submission row. Used by the manual-override flow."""
    sets: list[str] = []
    args: list = []
    if answers is not None:
        sets.append("answers_json = ?")
        args.append(json.dumps({str(k): v for k, v in answers.items()}))
    if score is not None:
        sets.append("score_json = ?")
        args.append(json.dumps(score))
        sets.append("composite = ?")
        args.append(score.get("composite"))
    if not sets:
        return
    args.append(submission_id)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE submissions SET {', '.join(sets)} WHERE id = ?",
            tuple(args),
        )


def get_submission(submission_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM submissions WHERE id = ?", (submission_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["answers"] = json.loads(d.pop("answers_json")) if d["answers_json"] else {}
    d["score"] = json.loads(d.pop("score_json")) if d["score_json"] else None
    return d


### OAuth-flow state (existing) --------------------------------------------

def save_state(state: str, code_verifier: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO oauth_states (state, code_verifier) VALUES (?, ?)",
            (state, code_verifier),
        )


def consume_state(state: str) -> str | None:
    """Pop the row; return the associated code_verifier, or None if state is unknown."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT code_verifier FROM oauth_states WHERE state = ?", (state,)
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))
        return row["code_verifier"]
