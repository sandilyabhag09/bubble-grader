"""Real-time grading: Classroom tells us the moment work is turned in.

Instead of asking Google every few minutes "anything new?", each course with
app-owned assignments gets a Classroom *registration* that publishes coursework
and submission changes to a Pub/Sub topic; a push subscription on that topic
POSTs each event to ``/hooks/classroom``. We fetch the one submission named in
the event and, if it's freshly TURNED_IN on an assignment we own, grade it
right there (which also returns the grade to the student).

Registrations expire after roughly a week, so ``ensure_registrations`` runs
from the cron ping and renews anything close to expiring. The cron's grading
pass stays as the safety net for anything a notification missed.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timedelta, timezone

from . import db as dbmod
from .classroom import create_registration, delete_registration, get_submission
from .config import PUBSUB_TOPIC

# Renew a registration when it has less than this long left.
RENEW_WITHIN = timedelta(days=2)

SUBMISSIONS_COLLECTION = "courses.courseWork.studentSubmissions"


def _parse_ts(ts) -> datetime | None:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _teacher_candidates(preferred: str | None) -> list[str]:
    teachers = dbmod.list_teachers()
    if preferred and preferred in teachers:
        return [preferred] + [t for t in teachers if t != preferred]
    return teachers


def _needs_renewal(reg: dict | None, topic: str) -> bool:
    if not reg or reg.get("topic") != topic:
        return True
    exp = _parse_ts(reg.get("expiry_time"))
    return exp is None or exp - datetime.now(timezone.utc) < RENEW_WITHIN


def ensure_registrations(
    course_ids: list[str] | None = None, time_budget: float | None = None
) -> dict:
    """Register (or renew) push notifications for every course that has an
    app-owned assignment. Returns a summary; never raises for one course."""
    summary: dict = {"checked": 0, "registered": 0, "failed": 0, "details": []}
    if not PUBSUB_TOPIC:
        summary["disabled"] = True
        return summary
    started = time.monotonic()
    courses = dbmod.list_app_assignment_courses()
    if course_ids is not None:
        wanted = set(course_ids)
        courses = [c for c in courses if c["course_id"] in wanted]

    for c in courses:
        if time_budget is not None and time.monotonic() - started > time_budget:
            summary["truncated"] = True
            break
        course_id = c["course_id"]
        summary["checked"] += 1
        existing = dbmod.get_push_registration(course_id)
        if not _needs_renewal(existing, PUBSUB_TOPIC):
            continue
        last_error = None
        for email in _teacher_candidates((existing or {}).get("teacher_email") or c.get("created_by")):
            try:
                if existing and existing.get("teacher_email") == email:
                    # Classroom rejects a duplicate (course, topic) registration,
                    # so retire the old one before asking for a fresh week.
                    try:
                        delete_registration(email, existing["registration_id"])
                    except Exception:  # noqa: BLE001 — already gone is fine
                        pass
                reg = create_registration(email, course_id, PUBSUB_TOPIC)
                dbmod.upsert_push_registration(
                    course_id,
                    registration_id=reg["registrationId"],
                    teacher_email=email,
                    topic=PUBSUB_TOPIC,
                    expiry_time=reg.get("expiryTime"),
                )
                summary["registered"] += 1
                summary["details"].append({"course_id": course_id, "teacher": email,
                                           "expires": reg.get("expiryTime")})
                last_error = None
                break
            except Exception as e:  # noqa: BLE001 — try the next teacher
                last_error = f"{type(e).__name__}: {e}"
        if last_error:
            summary["failed"] += 1
            summary["details"].append({"course_id": course_id, "error": last_error})
    return summary


def decode_pubsub_push(body: dict) -> dict | None:
    """The JSON Classroom put in a Pub/Sub push envelope, or None if it isn't one."""
    msg = (body or {}).get("message") or {}
    data = msg.get("data")
    if not data:
        return None
    try:
        return json.loads(base64.b64decode(data).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def handle_notification(body: dict) -> dict:
    """Act on one Pub/Sub push. Always returns a small status dict — callers
    answer 200 regardless so Pub/Sub doesn't redeliver events we can't use;
    the cron pass is the retry path."""
    event = decode_pubsub_push(body)
    if not event:
        return {"status": "ignored", "reason": "not a pubsub push"}
    if event.get("collection") != SUBMISSIONS_COLLECTION:
        return {"status": "ignored", "reason": f"collection {event.get('collection')}"}
    rid = event.get("resourceId") or {}
    course_id, cw_id, sub_id = rid.get("courseId"), rid.get("courseWorkId"), rid.get("id")
    if not (course_id and cw_id and sub_id):
        return {"status": "ignored", "reason": "incomplete resourceId"}

    owned = dbmod.get_app_assignment(course_id, cw_id)
    if not owned or not owned.get("test_id"):
        return {"status": "ignored", "reason": "not an app-owned assignment with a test"}

    reg = dbmod.get_push_registration(course_id) or {}
    teachers = _teacher_candidates(reg.get("teacher_email") or owned.get("created_by"))
    if not teachers:
        return {"status": "ignored", "reason": "no teacher credentials"}
    email = teachers[0]

    sub = get_submission(email, course_id, cw_id, sub_id)
    if sub.get("state") != "TURNED_IN":
        return {"status": "ignored", "reason": f"state {sub.get('state')}"}
    student_id = sub.get("userId")
    if not student_id:
        return {"status": "ignored", "reason": "no userId"}

    # Skip if we already graded this exact turn-in (Pub/Sub may deliver twice).
    turned_in_at = _parse_ts(sub.get("updateTime"))
    for r in dbmod.list_submissions(course_id=course_id, coursework_id=cw_id):
        if r.get("student_id") != student_id:
            continue
        graded_at = _parse_ts(r.get("created_at"))
        if graded_at and turned_in_at and graded_at >= turned_in_at:
            return {"status": "ignored", "reason": "already graded this turn-in",
                    "student_id": student_id}
        break

    from .submissions import grade_classroom_assignment, template_for_test
    template_path, _ref = template_for_test(owned["test_id"])
    result = grade_classroom_assignment(
        email, course_id, cw_id, owned["test_id"], template_path,
        only_students=[student_id],
    )
    r = next(iter(result.get("results", [])), {})
    return {
        "status": r.get("status") or "no_result",
        "student_id": student_id,
        "name": r.get("name"),
        "composite": r.get("composite"),
        "grade_pushed": r.get("grade_pushed"),
        "error": r.get("error") or r.get("push_error"),
    }
