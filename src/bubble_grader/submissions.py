"""End-to-end fetch: download every Drive attachment for an assignment's submissions.

Scans are stored in the DATABASE (scan_files table), not on local disk, so a
serverless deploy and a laptop install share the same store and a grading
cycle's files are available wherever the next step runs. Blobs are transient:
deleted when feedback goes out, purged after two weeks, and refetchable from
Drive at any time.

The OMR/imaging imports are deliberately lazy so importing this module (and
therefore the web server) stays fast on serverless cold starts.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from googleapiclient.errors import HttpError

from . import db as dbmod
from .classroom import (
    get_coursework,
    list_roster,
    list_submissions,
    patch_grade,
    return_submission,
)
from .config import DATA_DIR
from .drive import EXT_BY_MIME, download_file_bytes
from .scoring import full_grade, partial_summary


def fetch_assignment(
    email: str,
    course_id: str,
    coursework_id: str,
    only_turned_in: bool = True,
    only_students: list[str] | None = None,
) -> dict:
    """Download the assignment's Drive attachments into the DB scan store.

    Returns a manifest-shaped dict — {"students": {sid: {…, "files": […]}}} —
    covering just the students processed by THIS call. Rows upsert per
    student/file, so per-student fetches merge instead of clobbering earlier
    ones (that's what makes one-student-per-request grading possible).

    ``only_students`` (a list of student ids) restricts the fetch to those
    students — used by chunked grading and auto-grade.
    """
    dbmod.purge_old_scans()  # opportunistic safety net for forgotten cycles
    roster = {s["userId"]: s for s in list_roster(email, course_id)}
    submissions = list_submissions(email, course_id, coursework_id)
    if only_students:
        wanted = set(only_students)
        submissions = [s for s in submissions if s.get("userId") in wanted]
    if only_turned_in:
        # Both TURNED_IN (newly submitted) and RETURNED (already-graded-and-returned)
        # have the student's attached work and are gradeable. Without RETURNED here,
        # any assignment whose grades have been released would silently appear empty
        # on a re-grade — blocking the override-then-republish workflow.
        submissions = [
            s for s in submissions
            if s.get("state") in ("TURNED_IN", "RETURNED")
        ]

    manifest: dict = {
        "course_id": course_id,
        "coursework_id": coursework_id,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "only_turned_in": only_turned_in,
        "students": {},
    }

    for sub in submissions:
        student_id = sub["userId"]
        profile = roster.get(student_id, {}).get("profile", {})
        student_name = (profile.get("name") or {}).get("fullName")
        student_email = profile.get("emailAddress")
        row_meta = dict(
            student_name=student_name,
            student_email=student_email,
            classroom_submission_id=sub.get("id"),
            state=sub.get("state"),
        )

        attachments = (
            sub.get("assignmentSubmission", {}).get("attachments", []) or []
        )

        files: list[dict] = []
        for att in attachments:
            df = att.get("driveFile")
            if not df or not df.get("id"):
                # Skip non-Drive attachments (link, youTubeVideo, form).
                continue
            file_id = df["id"]
            try:
                content, meta = download_file_bytes(email, file_id)
                dbmod.store_scan_file(
                    course_id, coursework_id, student_id, file_id,
                    name=meta.get("name"), mime=meta.get("mimeType"),
                    content=content, error=None, **row_meta,
                )
                files.append(
                    {
                        "file_id": file_id,
                        "name": meta.get("name"),
                        "mime": meta.get("mimeType"),
                        "size": meta.get("size"),
                    }
                )
            except HttpError as e:
                dbmod.store_scan_file(
                    course_id, coursework_id, student_id, file_id,
                    content=None, error=str(e), **row_meta,
                )
                files.append({"file_id": file_id, "error": str(e)})

        manifest["students"][student_id] = {
            "email": student_email,
            "name": student_name,
            "submission_id": sub.get("id"),
            "state": sub.get("state"),
            "files": files,
        }

    return manifest


def scan_to_tempfile(scan: dict) -> str:
    """Write a scan row's bytes to a temp file the OMR reader can open.

    The reader picks its decoder by extension, so the suffix comes from the
    stored mime type. Caller is responsible for os.unlink() when done.
    """
    ext = EXT_BY_MIME.get(scan.get("mime") or "", "bin")
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    with os.fdopen(fd, "wb") as f:
        f.write(scan["content"])
    return path


# Absolute so it works regardless of the process working directory
# (serverless functions don't run from the repo root).
DEFAULT_SHEETS_DIR = DATA_DIR / "sheets"


def template_for_test(test_id: str) -> tuple[Path, Path]:
    """Pick the right OMR template + reference image for a test.

    Inspects the test's English answer-key length to detect new-format
    (50/45/36/40) vs legacy (75/60/40/40). New-format tests use
    ``act_sheet_new.*``; everything else uses the original ``act_sheet.*``.
    Returns (template_path, reference_path) ready to hand to read_sheet_fm.
    """
    t = dbmod.get_test(test_id)
    is_new = False
    if t and t.get("answer_key"):
        eng = (t["answer_key"] or {}).get("Test 1") or {}
        is_new = len(eng) <= 50
    stem = "act_sheet_new" if is_new else "act_sheet"
    base = DEFAULT_SHEETS_DIR
    return (base / f"{stem}.template.json", base / f"{stem}.reference.png")


def _resolve_reference(template_path: Path) -> Path:
    """Find the reference image associated with a template (for feature matching)."""
    tpl = json.loads(template_path.read_text())
    ref_name = tpl.get("reference_image")
    if not ref_name:
        raise ValueError(
            f"Template {template_path} has no `reference_image` field; "
            "feature-matching reader needs one (re-run `prepare-act`)."
        )
    return template_path.parent / ref_name


def grade_classroom_assignment(
    email: str,
    course_id: str,
    coursework_id: str,
    test_id: str,
    template_path: Path | str,
    *,
    only_turned_in: bool = True,
    refetch: bool = True,
    only_students: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch → read → grade → persist for every student in one assignment.

    Returns {"results": [...]} where each item is one student's outcome:
        { student_id, name, email, file, status, composite?, error?, submission_id? }
    A failure on one student (bad scan, mis-aligned, etc.) is captured per-row
    so the run as a whole completes for the rest of the class.
    """
    template_path = Path(template_path)
    reference_path = _resolve_reference(template_path)
    template = json.loads(template_path.read_text())

    test = dbmod.get_test(test_id)
    if test is None:
        raise ValueError(f"No test '{test_id}'. Add it first with `test add`.")
    if not test["answer_key"]:
        raise ValueError(f"Test '{test_id}' has no answer key; use `test set-key`.")

    # Look up the assignment's scope (if any). Full-scope assignments
    # behave exactly like before; partial-scope ones get a `partial`
    # block attached to each student's report so release_grades can
    # write a percentage-of-max to Classroom and the UI can show
    # raw/total instead of composite.
    app_asg = dbmod.get_app_assignment(course_id, coursework_id)
    scope = (app_asg or {}).get("scope")

    from .omr import read_sheet_fm  # lazy: keep server cold starts light

    if refetch:
        manifest = fetch_assignment(
            email, course_id, coursework_id,
            only_turned_in=only_turned_in, only_students=only_students,
        )
        students = manifest["students"]
    else:
        students = dbmod.list_scan_students(course_id, coursework_id)
        if not students:
            raise ValueError(
                "No cached scans in the database for this assignment; "
                "pass refetch=True or run `fetch` first."
            )

    wanted_students = set(only_students) if only_students else None
    results: list[dict] = []
    for student_id, info in students.items():
        if wanted_students is not None and student_id not in wanted_students:
            continue
        files = info.get("files", []) or []
        if not files:
            results.append({
                "student_id": student_id,
                "name": info.get("name"),
                "email": info.get("email"),
                "file": None,
                "status": "no_files",
            })
            continue
        # Grade the first usable file. If a student uploads multiple pages we'd
        # need to merge — out of scope for now.
        chosen = next((f for f in files if "error" not in f), None)
        if chosen is None:
            results.append({
                "student_id": student_id,
                "name": info.get("name"),
                "email": info.get("email"),
                "file": None,
                "status": "download_failed",
                "error": files[0].get("error", "unknown download error"),
            })
            continue

        scan = dbmod.get_student_scan(course_id, coursework_id, student_id)
        if scan is None:
            results.append({
                "student_id": student_id,
                "name": info.get("name"),
                "email": info.get("email"),
                "file": chosen.get("name"),
                "status": "download_failed",
                "error": "scan not in database (fetch again)",
            })
            continue
        tmp_path = scan_to_tempfile(scan)
        try:
            read_result = read_sheet_fm(tmp_path, template_path, reference_path)
            answers = {int(k): v for k, v in read_result["answers"].items()}
            # Grade against the scored-only answer key — field-test answers live
            # separately and can never affect the score.
            report = full_grade(answers, template, test["answer_key"], test["scaler"])
            if scope and scope.get("type") == "partial":
                report["partial"] = partial_summary(report, scope)
            sub_id = dbmod.add_submission(
                test_id=test_id,
                answers=answers,
                score=report,
                student_id=student_id,
                student_name=info.get("name"),
                student_email=info.get("email"),
                course_id=course_id,
                coursework_id=coursework_id,
                classroom_submission_id=info.get("submission_id"),
            )
            results.append({
                "student_id": student_id,
                "name": info.get("name"),
                "email": info.get("email"),
                "file": chosen.get("name"),
                "status": "graded",
                "composite": report.get("composite"),
                "submission_id": sub_id,
                "match_info": read_result.get("match_info"),
            })
        except Exception as e:  # noqa: BLE001 — surface per-student failures, keep going
            results.append({
                "student_id": student_id,
                "name": info.get("name"),
                "email": info.get("email"),
                "file": chosen.get("name"),
                "status": "grade_failed",
                "error": f"{type(e).__name__}: {e}",
            })
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Remember failures so the assignment page can say WHY a student has no
    # score; success wipes any stale note. Never let bookkeeping break grading.
    for r in results:
        try:
            sid = r.get("student_id")
            if not sid:
                continue
            if r.get("status") == "graded":
                dbmod.clear_grade_error(course_id, coursework_id, sid)
            else:
                dbmod.record_grade_error(
                    course_id, coursework_id, sid,
                    status=r.get("status") or "failed", error=r.get("error"),
                )
        except Exception:  # noqa: BLE001
            pass

    return {
        "test_id": test_id,
        "course_id": course_id,
        "coursework_id": coursework_id,
        "results": results,
    }


def release_grades(
    email: str,
    course_id: str,
    coursework_id: str,
    *,
    scale_to: float | None = None,
    return_to_student: bool = False,
    draft_only: bool = False,
    only_students: list[str] | None = None,
) -> dict:
    """Push grades from the local DB into Classroom for one assignment.

    For each row in the submissions table for (course_id, coursework_id) with a
    Classroom submission id and a composite score:
      - PATCH the studentSubmission with the grade
      - Optionally `return` it so the student sees the score

    If `scale_to` is given (e.g. 100), composite (1-36) is rescaled to
    `composite/36 * scale_to`. Otherwise composite is sent as-is (use this when
    the Classroom assignment's maxPoints is 36).

    `draft_only=True` sets only `draftGrade` so the teacher can review before
    making it official; the student never sees a draft grade.

    `only_students` filters the release to a subset, identified by either
    Classroom userId or student email. None / empty list = release everyone.

    Returns a per-student outcome list mirroring `grade_classroom_assignment`.
    """
    # Pull the latest submission per student (DB rows are DESC by created_at).
    # include_score=True is needed for partial-scope assignments — the
    # raw/total/percent lives inside score_json, not on a top-level column.
    rows = dbmod.list_submissions(
        course_id=course_id, coursework_id=coursework_id, include_score=True,
    )
    latest_per_student: dict[str, dict] = {}
    for r in rows:
        sid = r["student_id"]
        if sid and sid not in latest_per_student:
            latest_per_student[sid] = r

    # Apply the only_students filter (accepts IDs and emails).
    if only_students:
        wanted = {s.strip() for s in only_students if s and s.strip()}
        if wanted:
            latest_per_student = {
                sid: r
                for sid, r in latest_per_student.items()
                if sid in wanted or (r.get("student_email") or "") in wanted
            }

    # Look up the assignment's maxPoints — surface a hint if Classroom-side
    # configuration doesn't match the grade we're about to send.
    try:
        cw = get_coursework(email, course_id, coursework_id)
        max_points = cw.get("maxPoints")
    except Exception as e:  # noqa: BLE001
        max_points = None

    results: list[dict] = []
    for sid, r in latest_per_student.items():
        composite = r.get("composite")
        cls_sub_id = r.get("classroom_submission_id")
        name = r.get("student_name") or r.get("student_email") or sid

        # Partial-scope assignments override the grade computation: the
        # raw count / range size becomes the percentage we send to
        # Classroom (normalized to the assignment's maxPoints). This
        # path runs whenever the submission's score blob has a
        # `partial` summary regardless of `scale_to` — the partial
        # percentage IS the canonical score for these assignments.
        score = r.get("score") or {}
        partial = score.get("partial") if isinstance(score, dict) else None
        if partial and partial.get("total"):
            mp = max_points or 100  # default to 100 if Classroom didn't tell us
            grade = (partial["raw"] / partial["total"]) * mp
            if not cls_sub_id:
                results.append({"student_id": sid, "name": name, "status": "no_classroom_submission"})
                continue
        else:
            if composite is None:
                results.append({"student_id": sid, "name": name, "status": "no_composite"})
                continue
            if not cls_sub_id:
                results.append({"student_id": sid, "name": name, "status": "no_classroom_submission"})
                continue
            grade = composite if scale_to is None else (composite / 36.0) * scale_to

        try:
            patch_grade(
                email, course_id, coursework_id, cls_sub_id, float(grade),
                draft_only=draft_only,
            )
            status = "draft_set" if draft_only else "grade_assigned"
            if return_to_student and not draft_only:
                return_submission(email, course_id, coursework_id, cls_sub_id)
                status = "returned"
            results.append({
                "student_id": sid, "name": name,
                "composite": composite, "partial": partial, "sent": grade,
                "status": status,
            })
        except Exception as e:  # noqa: BLE001
            results.append({
                "student_id": sid, "name": name,
                "composite": composite, "partial": partial, "sent": grade,
                "status": "release_failed",
                "error": f"{type(e).__name__}: {e}",
            })

    return {
        "course_id": course_id,
        "coursework_id": coursework_id,
        "max_points": max_points,
        "scale_to": scale_to,
        "results": results,
    }
