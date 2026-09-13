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
                content, mime, fname = _normalize_scan(
                    content, meta.get("mimeType"), meta.get("name")
                )
                dbmod.store_scan_file(
                    course_id, coursework_id, student_id, file_id,
                    name=fname, mime=mime,
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


_HEIC_MIMES = {"image/heic", "image/heif"}


def _normalize_scan(
    content: bytes, mime: str | None, name: str | None
) -> tuple[bytes, str | None, str | None]:
    """Convert formats the OMR reader can't open into ones it can.

    iPhones default to HEIC, which neither OpenCV nor pypdfium2 decode —
    convert to JPEG at store time so everything downstream just works.
    Unknown formats pass through untouched.
    """
    is_heic = (mime or "").lower() in _HEIC_MIMES or (name or "").lower().endswith(
        (".heic", ".heif")
    )
    if not is_heic:
        return content, mime, name
    import io
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    img = Image.open(io.BytesIO(content))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    new_name = (name.rsplit(".", 1)[0] if name else "scan") + ".jpg"
    return buf.getvalue(), "image/jpeg", new_name


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

# Every bubble sheet the reader knows, with its per-section question counts.
# A test picks the sheet whose counts equal its total questions per section
# (scored + field-test); a scan that doesn't match the expected sheet is
# retried against the others (students do grab the wrong printout).
SHEET_LAYOUTS: dict[str, dict[str, int]] = {
    "act_sheet":        {"Test 1": 75, "Test 2": 60, "Test 3": 40, "Test 4": 40},  # legacy
    "act_sheet_new":    {"Test 1": 50, "Test 2": 45, "Test 3": 36, "Test 4": 40},  # 2025 format
    "act_sheet_scored": {"Test 1": 40, "Test 2": 41, "Test 3": 27, "Test 4": 34},  # 2025, no field-test rows
}

# Feature-match quality floor. Real matches measured 49-1100+ inliers at a
# 0.31-0.60 inlier ratio; a scan of the WRONG sheet that scraped past the
# reader's minimum measured 27 @ 0.10 and 38 @ 0.25 — and graded as noise.
MIN_MATCH_INLIERS = 40
MIN_MATCH_RATIO = 0.28


def sheet_paths(stem: str) -> tuple[Path, Path]:
    return (DEFAULT_SHEETS_DIR / f"{stem}.template.json",
            DEFAULT_SHEETS_DIR / f"{stem}.reference.png")


def sheet_stem_from_path(template_path: Path | str) -> str:
    return Path(template_path).name.split(".template")[0]


def template_stem_for_test(test_id: str) -> str:
    """Which sheet a test's students should be bubbling on."""
    t = dbmod.get_test(test_id)
    key = (t or {}).get("answer_key") or {}
    if not key:
        return "act_sheet"
    ft = (t or {}).get("field_test_answers") or {}
    totals = {sec: len(key.get(sec) or {}) + len(ft.get(sec) or {}) for sec in SHEET_LAYOUTS["act_sheet"]}
    for stem, counts in SHEET_LAYOUTS.items():
        if all(totals.get(sec, 0) == n for sec, n in counts.items()):
            return stem
    # Unusual key (partial sections etc.): fall back to the format heuristic.
    return "act_sheet_new" if len(key.get("Test 1") or {}) <= 50 else "act_sheet"


def template_for_test(test_id: str) -> tuple[Path, Path]:
    """(template_path, reference_path) for the sheet this test is bubbled on."""
    return sheet_paths(template_stem_for_test(test_id))


def _match_ok(read_result: dict) -> bool:
    info = read_result.get("match_info") or {}
    return (info.get("n_inliers", 0) >= MIN_MATCH_INLIERS
            and info.get("inlier_ratio", 0.0) >= MIN_MATCH_RATIO)


def _read_with_best_sheet(scan_path: str, preferred_stem: str) -> tuple[dict, str]:
    """Read a scan against the expected sheet; if that match is weak, try every
    other known sheet and keep the best. Refuses (raises) rather than grade a
    scan that matched nothing well — a bogus score is worse than no score."""
    from .omr import read_sheet_fm

    order = [preferred_stem] + [st for st in SHEET_LAYOUTS if st != preferred_stem]
    best: tuple[dict, str, int] | None = None
    errors: list[str] = []
    for stem in order:
        tpl, ref = sheet_paths(stem)
        if not tpl.exists():
            continue
        try:
            rr = read_sheet_fm(scan_path, tpl, ref, return_warped=True)
        except Exception as e:  # noqa: BLE001 — try the next sheet
            errors.append(f"{stem}: {type(e).__name__}: {e}")
            continue
        if stem == preferred_stem and _match_ok(rr):
            return rr, stem  # fast path: expected sheet, confident match
        n = (rr.get("match_info") or {}).get("n_inliers", 0)
        if best is None or n > best[2]:
            best = (rr, stem, n)
    if best is not None and _match_ok(best[0]):
        return best[0], best[1]
    detail = (f"best was {best[2]} inliers against {best[1]}" if best
              else "; ".join(errors) or "no sheets available")
    raise ValueError(
        f"sheet didn't match any known layout ({detail}) — wrong printout or unreadable "
        "photo? Not graded, to avoid a bogus score."
    )


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


# Review queue: only the rows that cost the student a point (BLANK / MULTI).
# Borderline-but-committed answers are deliberately NOT surfaced — teachers
# found "kept B, but it was close" more confusing than helpful.
# EVERY blank/multi row is listed (a teacher asked why 34 blanks showed 20
# rows); only the row images are capped, since they're what makes the stored
# grade heavy. Rows past the cap still appear, just without a picture.
REVIEW_MAX_CROPS = 60


def _review_suspects(read_result: dict, template: dict, max_crops: int = REVIEW_MAX_CROPS) -> list[dict]:
    """BLANK and MULTI rows, each with a cropped image of the row.

    These are the calls that cost points, so the teacher confirms each from a
    picture (and types the real answer if the reader was wrong). Returned
    entries go into score["review"]; the reader's committed answers are trusted
    and never listed here.
    """
    fills = read_result.get("fills") or {}
    answers = read_result.get("answers") or {}
    warped = read_result.get("warped")
    dpi = read_result.get("warped_dpi", 200)

    meta: dict[int, tuple[str, int]] = {}
    bubbles_by_q: dict[int, list[dict]] = {}
    for b in template.get("bubbles", []):
        meta[b["q"]] = (b.get("section"), b.get("q_in_test"))
        bubbles_by_q.setdefault(b["q"], []).append(b)

    suspects: list[dict] = []
    for q in sorted(fills):
        given = answers.get(q)
        if given == "MULTI":
            reason = "read as MULTI (two marks) — enter the intended answer"
            severity = 0
        elif given == "BLANK":
            reason = "read as BLANK — confirm, or enter the answer if a mark is there"
            severity = 1
        else:
            continue
        section, q_in_test = meta.get(q, (None, None))
        suspects.append({
            "q": q, "section": section, "q_in_test": q_in_test,
            "given": given, "reason": reason, "_severity": severity,
        })

    # Multis first (an intended answer exists), then blanks.
    suspects.sort(key=lambda d: (d["_severity"], d["section"] or "", d["q_in_test"] or 0))

    if warped is not None:
        import base64
        import cv2
        px_per_mm = dpi / 25.4
        h, w = warped.shape[:2]
        for sus in suspects[:max_crops]:
            row = bubbles_by_q.get(sus["q"]) or []
            if not row:
                continue
            xs = [b["center_mm"][0] * px_per_mm for b in row]
            ys = [b["center_mm"][1] * px_per_mm for b in row]
            r = max(b["radius_mm"] * px_per_mm for b in row)
            x0 = max(0, int(min(xs) - 4 * r)); x1 = min(w, int(max(xs) + 4 * r))
            y0 = max(0, int(min(ys) - 2.2 * r)); y1 = min(h, int(max(ys) + 2.2 * r))
            if y1 <= y0 or x1 <= x0:
                continue  # row falls outside the warped frame — no crop
            crop = warped[y0:y1, x0:x1]
            ok, jpg = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                sus["crop_b64"] = base64.b64encode(jpg.tobytes()).decode()

    for sus in suspects:
        sus.pop("_severity", None)
    return suspects


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
    push_draft_grades: bool = True,
) -> dict[str, Any]:
    """Fetch → read → grade → persist for every student in one assignment.

    Returns {"results": [...]} where each item is one student's outcome:
        { student_id, name, email, file, status, composite?, error?, submission_id? }
    A failure on one student (bad scan, mis-aligned, etc.) is captured per-row
    so the run as a whole completes for the rest of the class.
    """
    template_path = Path(template_path)
    preferred_stem = sheet_stem_from_path(template_path)
    _template_cache: dict[str, dict] = {}

    def _template_for(stem: str) -> dict:
        if stem not in _template_cache:
            _template_cache[stem] = json.loads(sheet_paths(stem)[0].read_text())
        return _template_cache[stem]

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

        scans = dbmod.list_student_scans(course_id, coursework_id, student_id)
        if not scans:
            results.append({
                "student_id": student_id,
                "name": info.get("name"),
                "email": info.get("email"),
                "file": chosen.get("name"),
                "status": "download_failed",
                "error": "scan not in database (fetch again)",
            })
            continue

        # Try every attached file until one reads — a blurry first photo
        # shouldn't sink a submission whose second shot is fine.
        read_result, used_scan, used_stem, read_errors = None, None, None, []
        for scan in scans:
            tmp_path = scan_to_tempfile(scan)
            try:
                read_result, used_stem = _read_with_best_sheet(tmp_path, preferred_stem)
                used_scan = scan
                break
            except Exception as e:  # noqa: BLE001 — try the next file
                read_errors.append(f"{scan.get('name') or scan.get('file_id')}: {type(e).__name__}: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        if read_result is None:
            results.append({
                "student_id": student_id,
                "name": info.get("name"),
                "email": info.get("email"),
                "file": scans[0].get("name"),
                "status": "grade_failed",
                "error": f"none of {len(scans)} file(s) readable — " + " | ".join(read_errors),
            })
            continue

        try:
            template = _template_for(used_stem)
            answers = {int(k): v for k, v in read_result["answers"].items()}
            # Grade against the scored-only answer key — field-test answers live
            # separately and can never affect the score.
            report = full_grade(answers, template, test["answer_key"], test["scaler"])
            report["sheet_template"] = used_stem  # so overlays use the same geometry
            if scope and scope.get("type") == "partial":
                report["partial"] = partial_summary(report, scope)
            # Borderline reader calls, with row crops, for human review.
            try:
                report["review"] = _review_suspects(read_result, template)
            except Exception:  # noqa: BLE001 — review is best-effort, never blocks grading
                report["review"] = []
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
                "file": used_scan.get("name"),
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
                "file": used_scan.get("name"),
                "status": "grade_failed",
                "error": f"{type(e).__name__}: {e}",
            })

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

    # Push each new grade to Classroom as a DRAFT (teacher-visible only) so the
    # gradebook fills in by itself; "Return" later makes it official. Only
    # app-created assignments accept grade writes, so skip the rest quietly.
    graded_ids = [r["student_id"] for r in results if r.get("status") == "graded"]
    if push_draft_grades and graded_ids and app_asg is not None:
        try:
            rel = release_grades(
                email, course_id, coursework_id,
                draft_only=True, only_students=graded_ids,
            )
            by_sid = {x.get("student_id"): x for x in rel.get("results", [])}
            for r in results:
                if r.get("status") == "graded":
                    x = by_sid.get(r["student_id"], {})
                    r["draft_pushed"] = x.get("status") == "draft_set"
                    if not r["draft_pushed"]:
                        r["draft_error"] = x.get("error") or x.get("status") or "not pushed"
        except Exception as e:  # noqa: BLE001 — the grade itself is safe in our DB
            for r in results:
                if r.get("status") == "graded":
                    r["draft_pushed"] = False
                    r["draft_error"] = f"{type(e).__name__}: {e}"

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
