"""Return graded work inside Google Classroom — no email.

For one student: post their results text + the marked-up sheet as a private
announcement (visible to that student and teachers), then set the final grade
and return the submission so the score shows up for them. If the post fails
(e.g. missing permission), nothing is returned — the teacher fixes and retries
with no half-done state.
"""

from __future__ import annotations

from .classroom import create_student_announcement
from .drive import upload_pdf
from .feedback import student_feedback
from .submissions import release_grades


def _safe_filename(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in text).strip()
    return keep or "results"


def return_to_student(
    teacher_email: str,
    course_id: str,
    coursework_id: str,
    student_id: str,
    *,
    test_name: str | None = None,
    teacher_name: str | None = None,
) -> dict:
    fb = student_feedback(
        teacher_email, course_id, coursework_id, student_id, test_name=test_name
    )
    if fb["status"] != "ok":
        return {"student_id": student_id, "status": "no_graded_submission"}

    name = fb["student_name"]
    first = name.split()[0] if name and name != "there" else "there"
    text = f"Hi {first},\n\n{fb['report']}"
    if fb["pdf"]:
        text += (
            "\n\nAttached is your marked-up answer sheet: green circles are correct, "
            "red Xs are missed (wrong, blank, or multi-marked)."
        )
    else:
        text += f"\n\n(Marked-up sheet unavailable: {fb['overlay_error']})"
    if teacher_name:
        text += f"\n\n— {teacher_name}"

    file_id = None
    if fb["pdf"]:
        label = fb.get("test_name") or "ACT"
        uploaded = upload_pdf(teacher_email, f"{_safe_filename(f'{name} - {label}')} - marked up.pdf", fb["pdf"])
        file_id = uploaded["id"]

    # Post first: if this fails (permissions), we haven't returned anything yet.
    ann = create_student_announcement(
        teacher_email, course_id, text, student_ids=[student_id], drive_file_id=file_id
    )

    rel = release_grades(
        teacher_email, course_id, coursework_id,
        return_to_student=True, only_students=[student_id],
    )
    r = next(iter(rel.get("results", [])), {})
    ok = r.get("status") in ("returned", "grade_assigned")
    return {
        "student_id": student_id,
        "name": name,
        "status": "returned" if ok else "grade_failed",
        "announcement_id": ann.get("id"),
        "attached_overlay": bool(file_id),
        "overlay_error": fb["overlay_error"],
        "grade_status": r.get("status"),
        "error": r.get("error"),
    }
