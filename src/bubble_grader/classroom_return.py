"""Return graded work inside Google Classroom — no email.

For one student: post their results text + the marked-up sheet as a Classwork
material only they (and teachers) can see, filed under a "Graded results"
topic, then set the final grade and return the submission so the score shows
up for them. Classwork rather than the Stream so a class of 20 doesn't get 20
posts per test. If the post fails (e.g. missing permission), nothing is
returned — the teacher fixes and retries with no half-done state.
"""

from __future__ import annotations

from .classroom import create_student_material, ensure_topic
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
    label = fb.get("test_name") or "ACT"
    title = f"{label} — your results"
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
        uploaded = upload_pdf(teacher_email, f"{_safe_filename(f'{name} - {label}')} - marked up.pdf", fb["pdf"])
        file_id = uploaded["id"]

    # Post first: if this fails (permissions), we haven't returned anything yet.
    post = create_student_material(
        teacher_email, course_id, title, text,
        student_ids=[student_id], drive_file_id=file_id,
        topic_id=ensure_topic(teacher_email, course_id),
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
        "post_id": post.get("id"),
        "attached_overlay": bool(file_id),
        "overlay_error": fb["overlay_error"],
        "grade_status": r.get("status"),
        "error": r.get("error"),
    }
