"""Local-only auto-grader: poll Classroom and grade newly turned-in work.

Opt-in via the ``AUTO_GRADE_ON_TURNIN`` env var (see config). True real-time
grading would need Google Pub/Sub push to a public URL, which isn't available
on a local machine — so this polls every ``AUTO_GRADE_POLL_SECONDS`` and grades
anything that's been turned in since we last graded it. Near-immediate (within
one poll interval), not instant.

What it does NOT do: release/return grades to students. It only computes and
stores the score locally; the teacher still reviews and sends feedback. Failures
are swallowed per item so one bad scan never stops the loop.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from . import db as dbmod
from .classroom import list_courses, list_submissions
from .config import AUTO_GRADE_POLL_SECONDS
from .submissions import grade_classroom_assignment, template_for_test


def _parse_update_time(ts: str | None) -> datetime | None:
    """Parse Classroom's RFC3339 ``updateTime`` (e.g. ``...Z``) to a datetime."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _as_dt(ts) -> datetime | None:
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None


# (course_id, coursework_id, student_id, update_time) we've already tried, so a
# student with no scan / a bad scan isn't re-graded every single poll. A genuine
# re-submission changes update_time, producing a new key and a fresh attempt.
_attempted: set[tuple] = set()


def _newly_turned_in(email: str, course_id: str, coursework_id: str) -> list[str]:
    """Student ids whose current turned-in work hasn't been graded yet."""
    subs = list_submissions(email, course_id, coursework_id)

    # Latest grade time per student already in our DB (rows are newest-first).
    db_rows = dbmod.list_submissions(course_id=course_id, coursework_id=coursework_id)
    graded_at: dict[str, datetime | None] = {}
    for r in db_rows:
        sid = r.get("student_id")
        if sid and sid not in graded_at:
            graded_at[sid] = _as_dt(r.get("created_at"))

    out: list[str] = []
    for s in subs:
        if s.get("state") != "TURNED_IN":
            continue  # only fresh turn-ins; RETURNED means already handled
        sid = s.get("userId")
        if not sid:
            continue
        ut = _parse_update_time(s.get("updateTime"))
        key = (course_id, coursework_id, sid, s.get("updateTime"))
        if key in _attempted:
            continue
        prev = graded_at.get(sid)
        # Grade if we've never graded this student, or they re-submitted after
        # our most recent grade.
        if prev is None or (ut is not None and ut > prev):
            out.append(sid)
            _attempted.add(key)
    return out


def _grade_once() -> None:
    """One poll cycle across every teacher's app-owned assignments."""
    for email in dbmod.list_teachers():
        try:
            courses = list_courses(email)
        except Exception:  # noqa: BLE001 — skip a teacher we can't reach
            continue
        for course in courses:
            course_id = course.get("id")
            if not course_id:
                continue
            for asg in dbmod.list_app_assignments(course_id):
                test_id = asg.get("test_id")
                cw_id = asg.get("coursework_id")
                if not test_id or not cw_id:
                    continue
                try:
                    students = _newly_turned_in(email, course_id, cw_id)
                    if not students:
                        continue
                    template_path, _ref = template_for_test(test_id)
                    grade_classroom_assignment(
                        email, course_id, cw_id, test_id, template_path,
                        only_students=students,
                    )
                except Exception:  # noqa: BLE001 — never let one assignment break the loop
                    continue


def start_auto_grade_poller() -> threading.Thread:
    """Kick off a daemon thread that grades new turn-ins on an interval."""
    def loop() -> None:
        time.sleep(10)  # let the server finish coming up first
        while True:
            try:
                _grade_once()
            except Exception:  # noqa: BLE001 — never let the poller crash the app
                pass
            time.sleep(AUTO_GRADE_POLL_SECONDS)

    t = threading.Thread(target=loop, name="auto-grade", daemon=True)
    t.start()
    return t
