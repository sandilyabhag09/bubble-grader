"""Classroom API reads: courses, coursework (assignments), roster, submissions.

Read results are cached in-process for a short while (per teacher and id):
courses, assignments and rosters change rarely, but every page used to re-ask
Google for them — ~400 ms a call, several calls a page. Writes that change a
list (create/delete coursework, grade pushes) invalidate the affected entry,
so a teacher never sees their own edit lag behind.
"""

from __future__ import annotations

import threading
import time

from .google_api import service_for

# Seconds each list stays fresh. Submissions turn over fastest.
TTL_COURSES = 60
TTL_COURSEWORK = 45
TTL_ROSTER = 120
TTL_SUBMISSIONS = 20

_cache: dict[tuple, tuple[float, list]] = {}
_lock = threading.Lock()


def _cached(key: tuple, ttl: float, compute):
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
    if hit and hit[0] > now:
        return hit[1]
    value = compute()
    with _lock:
        _cache[key] = (now + ttl, value)
    return value


def invalidate(*prefix) -> None:
    """Drop cached entries whose key starts with ``prefix`` (empty = everything)."""
    with _lock:
        for k in [k for k in _cache if k[: len(prefix)] == prefix]:
            _cache.pop(k, None)


def _paginate(request_fn, items_key: str) -> list[dict]:
    items: list[dict] = []
    page_token = None
    while True:
        resp = request_fn(page_token).execute()
        items.extend(resp.get(items_key, []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            return items


def list_courses(email: str) -> list[dict]:
    def compute():
        svc = service_for(email, "classroom", "v1")
        return _paginate(
            lambda tok: svc.courses().list(courseStates=["ACTIVE"], pageToken=tok),
            "courses",
        )
    return _cached(("courses", email), TTL_COURSES, compute)


def list_coursework(email: str, course_id: str) -> list[dict]:
    """List assignments / questions in a course (all states the teacher can see)."""
    def compute():
        svc = service_for(email, "classroom", "v1")
        return _paginate(
            lambda tok: svc.courses().courseWork().list(
                courseId=course_id, pageToken=tok
            ),
            "courseWork",
        )
    return _cached(("coursework", email, course_id), TTL_COURSEWORK, compute)


def list_roster(email: str, course_id: str) -> list[dict]:
    """List students in a course. Each item has userId + profile (name, email)."""
    def compute():
        svc = service_for(email, "classroom", "v1")
        return _paginate(
            lambda tok: svc.courses().students().list(
                courseId=course_id, pageToken=tok
            ),
            "students",
        )
    return _cached(("roster", email, course_id), TTL_ROSTER, compute)


def list_submissions(
    email: str, course_id: str, coursework_id: str
) -> list[dict]:
    """List student submissions for one assignment (all states)."""
    def compute():
        svc = service_for(email, "classroom", "v1")
        return _paginate(
            lambda tok: svc.courses().courseWork().studentSubmissions().list(
                courseId=course_id, courseWorkId=coursework_id, pageToken=tok
            ),
            "studentSubmissions",
        )
    return _cached(
        ("submissions", email, course_id, coursework_id), TTL_SUBMISSIONS, compute
    )


def get_coursework(email: str, course_id: str, coursework_id: str) -> dict:
    """Fetch the assignment metadata (incl. maxPoints needed for grade scaling)."""
    svc = service_for(email, "classroom", "v1")
    return (
        svc.courses()
        .courseWork()
        .get(courseId=course_id, id=coursework_id)
        .execute()
    )


def delete_coursework(email: str, course_id: str, coursework_id: str) -> dict:
    """Delete a Classroom assignment. Only works for coursework our OAuth
    project created — Google rejects deletes of UI-created coursework with 403.
    """
    svc = service_for(email, "classroom", "v1")
    result = (
        svc.courses()
        .courseWork()
        .delete(courseId=course_id, id=coursework_id)
        .execute()
    )
    invalidate("coursework", email, course_id)
    invalidate("submissions", email, course_id, coursework_id)
    return result


def create_coursework(
    email: str,
    course_id: str,
    title: str,
    *,
    description: str | None = None,
    max_points: float = 36,
    draft: bool = False,
    work_type: str = "ASSIGNMENT",
) -> dict:
    """Create a new assignment in a Classroom course.

    We must create coursework via our OAuth client (vs the Classroom web UI)
    if we want to PATCH its grades later — Classroom restricts grade writes
    to the same Developer Console project that created the coursework.
    """
    body: dict = {
        "title": title,
        "workType": work_type,
        "state": "DRAFT" if draft else "PUBLISHED",
        "maxPoints": max_points,
    }
    if description:
        body["description"] = description
    svc = service_for(email, "classroom", "v1")
    result = (
        svc.courses()
        .courseWork()
        .create(courseId=course_id, body=body)
        .execute()
    )
    invalidate("coursework", email, course_id)
    return result


def patch_grade(
    email: str,
    course_id: str,
    coursework_id: str,
    submission_id: str,
    grade: float,
    *,
    draft_only: bool = False,
) -> dict:
    """Set the assigned (and draft) grade on a student submission.

    Setting `draftGrade` is teacher-only visibility. Setting `assignedGrade`
    makes it the official score — the student still won't see it until the
    submission is *returned* (see `return_submission`).
    """
    svc = service_for(email, "classroom", "v1")
    body: dict = {"draftGrade": grade}
    mask = "draftGrade"
    if not draft_only:
        body["assignedGrade"] = grade
        mask = "assignedGrade,draftGrade"
    result = (
        svc.courses()
        .courseWork()
        .studentSubmissions()
        .patch(
            courseId=course_id,
            courseWorkId=coursework_id,
            id=submission_id,
            updateMask=mask,
            body=body,
        )
        .execute()
    )
    invalidate("submissions", email, course_id, coursework_id)
    return result


def return_submission(
    email: str, course_id: str, coursework_id: str, submission_id: str
) -> dict:
    """Return a graded submission so the student can see the score."""
    svc = service_for(email, "classroom", "v1")
    # `return` is a Python keyword; the API client wraps it as `return_`.
    result = (
        svc.courses()
        .courseWork()
        .studentSubmissions()
        .return_(
            courseId=course_id,
            courseWorkId=coursework_id,
            id=submission_id,
            body={},
        )
        .execute()
    )
    invalidate("submissions", email, course_id, coursework_id)
    return result
