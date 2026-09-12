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
from .classroom import get_coursework, list_courses, list_submissions
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
    run_auto_grade_once()


def run_auto_grade_once(
    max_students: int | None = None,
    time_budget: float | None = None,
    max_assignment_age_days: int = 45,
) -> dict:
    """Grade everything newly turned in, across all teachers. Returns a summary.

    Called two ways: by the local background thread, and by the tokened
    ``/cron/auto-grade`` endpoint that an external pinger hits on serverless
    hosting (where background threads don't exist). ``max_students`` and
    ``time_budget`` (seconds) bound one run so a serverless request never
    outstays its welcome — anything left over is picked up by the next ping.

    Only assignments created in the last ``max_assignment_age_days`` are
    checked, so the per-run Classroom API cost doesn't grow forever; old
    assignments can always be graded with the button.
    """
    started = time.monotonic()
    summary: dict = {"assignments_checked": 0, "students_graded": 0,
                     "students_failed": 0, "details": [], "truncated": False}

    def _out_of_budget() -> bool:
        if time_budget is not None and time.monotonic() - started > time_budget:
            return True
        if max_students is not None and (
            summary["students_graded"] + summary["students_failed"] >= max_students
        ):
            return True
        return False

    cutoff = None
    if max_assignment_age_days:
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_assignment_age_days)

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
                created = _as_dt(asg.get("created_at"))
                if cutoff is not None and created is not None and created < cutoff:
                    continue
                if _out_of_budget():
                    summary["truncated"] = True
                    return summary
                try:
                    summary["assignments_checked"] += 1
                    students = _newly_turned_in(email, course_id, cw_id)
                    if not students:
                        continue
                    if max_students is not None:
                        room = max_students - (summary["students_graded"] + summary["students_failed"])
                        students = students[:max(0, room)]
                    if not students:
                        summary["truncated"] = True
                        return summary
                    template_path, _ref = template_for_test(test_id)
                    result = grade_classroom_assignment(
                        email, course_id, cw_id, test_id, template_path,
                        only_students=students,
                    )
                    for r in result.get("results", []):
                        ok = r.get("status") == "graded"
                        summary["students_graded" if ok else "students_failed"] += 1
                        summary["details"].append({
                            "coursework_id": cw_id,
                            "student": r.get("name") or r.get("student_id"),
                            "status": r.get("status"),
                            "composite": r.get("composite"),
                            "error": r.get("error"),
                        })
                except Exception as e:  # noqa: BLE001 — never let one assignment break the loop
                    liveness = _assignment_liveness(email, course_id, cw_id)
                    if liveness == "deleted":
                        # Gone from Classroom for good — stop tracking it, so it
                        # never shows up or gets checked again. Grade history for
                        # students stays untouched.
                        dbmod.delete_app_assignment(
                            course_id, cw_id, cascade_submissions=False
                        )
                        summary["details"].append({
                            "coursework_id": cw_id,
                            "status": "pruned_deleted_assignment",
                        })
                    elif liveness == "unpublished":
                        pass  # draft/scheduled — will grade once it goes live
                    else:
                        summary["details"].append({
                            "coursework_id": cw_id, "status": "assignment_error",
                            "error": f"{type(e).__name__}: {e}",
                        })
                    continue
    return summary


def _assignment_liveness(email: str, course_id: str, cw_id: str) -> str:
    """Why can't this assignment's submissions be listed?

    'deleted'      -> coursework no longer exists in Classroom (permanent)
    'unpublished'  -> it exists but is DRAFT/SCHEDULED; fine once published
    'unknown'      -> anything else (transient error, permissions, ...)
    """
    from googleapiclient.errors import HttpError
    try:
        cw = get_coursework(email, course_id, cw_id)
    except HttpError as e:
        status = getattr(getattr(e, "resp", None), "status", None)
        return "deleted" if status == 404 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"
    return "unpublished" if cw.get("state") != "PUBLISHED" else "unknown"


def start_auto_grade_poller() -> threading.Thread:
    """Kick off a daemon thread that grades new turn-ins on an interval."""
    def loop() -> None:
        time.sleep(10)  # let the server finish coming up first
        while True:
            try:
                run_auto_grade_once()
            except Exception:  # noqa: BLE001 — never let the poller crash the app
                pass
            time.sleep(AUTO_GRADE_POLL_SECONDS)

    t = threading.Thread(target=loop, name="auto-grade", daemon=True)
    t.start()
    return t
