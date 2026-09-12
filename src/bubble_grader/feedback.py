"""Build per-student feedback reports + send them as personalized emails.

Report format:
    English: <scaled>, Missed questions: <q_in_test list>
    Math: ...
    Reading: ...
    Science: ...

"Missed" = incorrect + BLANK + MULTI (everything that didn't earn credit).

Each email optionally attaches a per-student PDF: the warped scan with green
circles over questions answered correctly and red Xs over questions that were
missed (wrong, BLANK, or MULTI), so the student can see exactly which ones
didn't earn credit — the same set listed as "Missed questions" in the body.
"""

import io
import json
import os
import tempfile
from pathlib import Path

from . import db as dbmod
from .config import DATA_DIR
from .gmail import send_email
from .scoring import grade_answers, merge_field_test

# NOTE: cv2 / PIL / the OMR reader are imported lazily inside
# build_overlay_pdf so that importing this module (and the web server)
# stays fast on serverless cold starts.


SECTION_DISPLAY = {
    "Test 1": "English",
    "Test 2": "Math",
    "Test 3": "Reading",
    "Test 4": "Science",
}
SECTION_ORDER = ["Test 1", "Test 2", "Test 3", "Test 4"]

# Default OMR template + reference. Could be made configurable later.
DEFAULT_TEMPLATE = DATA_DIR / "sheets" / "act_sheet.template.json"
DEFAULT_REFERENCE = DATA_DIR / "sheets" / "act_sheet.reference.png"


def _missed_clause(details: list[dict], q_start: int | None = None, q_end: int | None = None) -> str:
    """Build the "Missed questions: ..." clause from per-question details.

    Lists scored questions that didn't earn credit, then appends any field-test
    (not-scored) questions the student also got wrong as "(plus N, M not
    scored)" — so students see every wrong question while it's clear which ones
    actually counted. Optionally restrict to a q_in_test range (partial scope).
    """
    def _in_range(d: dict) -> bool:
        if q_start is None:
            return True
        q = d.get("q_in_test", 0)
        return q_start <= q <= q_end

    missed = [
        d for d in details
        if d.get("status") in ("incorrect", "blank", "multi") and _in_range(d)
    ]
    scored = sorted(d["q_in_test"] for d in missed if d.get("scored", True))
    unscored = sorted(d["q_in_test"] for d in missed if not d.get("scored", True))
    clause = "Missed questions: " + (", ".join(str(q) for q in scored) if scored else "none")
    if unscored:
        clause += f" (plus {', '.join(str(q) for q in unscored)} not scored)"
    return clause


def build_report(
    submission: dict,
    test_name: str | None = None,
    scope: dict | None = None,
) -> str:
    """Format a feedback report body from a submission's score data.

    When ``scope`` is a partial-scope dict, the email is rewritten to
    focus on the assigned passage/question range rather than dumping every
    section's scaled score and missed-question list.
    """
    score = submission.get("score") or {}

    # ----- partial scope ---------------------------------------------------
    if scope and scope.get("type") == "partial":
        return _build_partial_report(submission, test_name=test_name, scope=scope)

    # ----- full scope (existing layout) -----------------------------------
    sections = score.get("sections") or {}
    composite = score.get("composite")

    label = test_name or score.get("test_form") or "ACT Practice Test"
    lines: list[str] = [f"Here are your results from {label}:", ""]
    if composite is not None:
        lines.append(f"Composite: {composite}/36")
        lines.append("")

    for key in SECTION_ORDER:
        info = sections.get(key)
        if not info:
            continue
        display = SECTION_DISPLAY.get(key, key)
        scaled = info.get("scaled_score")
        score_str = f"{scaled}/36 scaled" if scaled is not None else "—"
        lines.append(f"{display}: {score_str}, {_missed_clause(info.get('details', []))}")

    return "\n".join(lines)


# Map of internal section key ("Test 1"..) → human label for emails.
_SECTION_FULL = {
    "Test 1": "English",
    "Test 2": "Mathematics",
    "Test 3": "Reading",
    "Test 4": "Science",
}


def _build_partial_report(
    submission: dict,
    *,
    test_name: str | None,
    scope: dict,
) -> str:
    """Compact email body for partial-scope assignments.

    Matches the teacher's requested phrasing:
        On {label} (questions a-b) in {Section} for {test name},
        you scored {raw}/{total} ({pct}%).
        Missed questions: ...
    """
    score = submission.get("score") or {}
    partial = score.get("partial") or {}
    sections = score.get("sections") or {}

    section_key = scope.get("section") or ""
    section_name = _SECTION_FULL.get(section_key, section_key)
    q_start = int(scope.get("q_start", 0))
    q_end = int(scope.get("q_end", 0))
    label = scope.get("label") or f"questions {q_start}–{q_end}"
    test_label = test_name or score.get("test_form") or "ACT Practice Test"

    raw = partial.get("raw")
    total = partial.get("total") or max(0, q_end - q_start + 1)
    pct = partial.get("percent")

    # Missed questions within the scope range only — same set the attachment
    # flags with a red X (incorrect, blank, or multi-marked).
    sec_info = sections.get(section_key) or {}
    missed_clause = _missed_clause(sec_info.get("details", []), q_start, q_end)

    score_line = (
        f"you scored {raw}/{total}" + (f" ({pct}%)" if pct is not None else "")
        if raw is not None else "your score wasn't computed"
    )

    return (
        f"Here are your results from {test_label}:\n"
        f"\n"
        f"On {label} (questions {q_start}–{q_end}) in {section_name}, {score_line}.\n"
        f"\n"
        f"{missed_clause}."
    )


def _student_scan_tempfile(
    teacher_email: str, course_id: str, coursework_id: str, student_id: str
) -> str | None:
    """Materialize the student's scan as a temp file for the overlay renderer.

    Scans normally sit in the DB scan store. If this assignment's grading
    cycle already deleted them, quietly refetch just this student's file from
    Drive — the original always lives there — and store it again. Returns a
    temp-file path (caller unlinks) or None if no scan can be had.
    """
    scan = dbmod.get_student_scan(course_id, coursework_id, student_id)
    if scan is None:
        from .submissions import fetch_assignment
        try:
            fetch_assignment(
                teacher_email, course_id, coursework_id,
                only_students=[student_id],
            )
        except Exception:  # noqa: BLE001 — fall through to "no scan"
            return None
        scan = dbmod.get_student_scan(course_id, coursework_id, student_id)
    if scan is None:
        return None
    from .submissions import scan_to_tempfile
    return scan_to_tempfile(scan)


def build_overlay_for_student(
    teacher_email: str, course_id: str, coursework_id: str, student_id: str
) -> tuple[bytes | None, str | None]:
    """(pdf_bytes, None) for the student's marked-up sheet, or (None, reason).

    Mirrors exactly what the feedback email attaches: latest submission,
    details refreshed against the current key (field-test included), scan
    pulled from the store (refetched from Drive if the cycle deleted it).
    """
    rows = dbmod.list_submissions(
        course_id=course_id, coursework_id=coursework_id, student_id=student_id
    )
    if not rows:
        return None, "No graded submission for this student yet."
    full = dbmod.get_submission(rows[0]["id"])
    if not full:
        return None, "Submission record could not be loaded."

    app_asg = dbmod.get_app_assignment(course_id, coursework_id)
    test_id = (app_asg or {}).get("test_id") or full.get("test_id")
    from .submissions import template_for_test
    if test_id:
        template_path, reference_path = template_for_test(test_id)
    else:
        template_path, reference_path = DEFAULT_TEMPLATE, DEFAULT_REFERENCE
    _test = dbmod.get_test(test_id) if test_id else None

    template_dict = json.loads(Path(template_path).read_text())
    email_score = _refresh_missed_details(
        full, template_dict, (_test or {}).get("answer_key"),
        field_test_answers=(_test or {}).get("field_test_answers"),
    )
    details = [
        d
        for sec in (email_score.get("sections") or {}).values()
        for d in (sec.get("details") or [])
    ]

    scan_path = _student_scan_tempfile(teacher_email, course_id, coursework_id, student_id)
    if scan_path is None:
        return None, "No scan on file for this student (and none fetchable from Drive)."
    try:
        return build_overlay_pdf(
            scan_path, template_path, reference_path, details=details
        ), None
    finally:
        try:
            os.unlink(scan_path)
        except OSError:
            pass


def build_overlay_pdf(
    scan_path: Path,
    template_path: Path = DEFAULT_TEMPLATE,
    reference_path: Path = DEFAULT_REFERENCE,
    dpi: int = 200,
    details: list[dict] | None = None,
) -> bytes:
    """Render the warped scan with a green-circle / red-X overlay; return PDF bytes.

    ``details`` is the flattened per-question score detail (every section's
    ``details`` entries concatenated). Each entry carries the *global* question
    number ``q``, the student's ``given`` answer, the ``correct`` answer, and a
    ``status`` of ``correct``/``incorrect``/``blank``/``multi``. The overlay
    reflects correctness against the answer key:

    * **correct**  → green circle on the (correctly filled) bubble
    * **incorrect**→ red X on the bubble the student wrongly filled
    * **blank/multi** → red X across every option in that question's row

    When ``details`` is ``None`` we fall back to the legacy behavior of simply
    circling whatever fill the reader detected (no correctness judgment).
    """
    import cv2  # lazy: heavy imaging deps load only when an overlay is built
    from PIL import Image
    from .omr import read_sheet_fm

    template = json.loads(template_path.read_text())
    px_per_mm = dpi / 25.4

    GREEN = (0, 200, 0)
    RED = (0, 0, 255)

    def _circle(img, cx, cy, r):
        cv2.circle(img, (cx, cy), r + 2, GREEN, 2)

    def _x(img, cx, cy, r):
        cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), RED, 2)
        cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), RED, 2)

    by_q = {d["q"]: d for d in details} if details is not None else None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        result = read_sheet_fm(scan_path, template_path, reference_path, debug_dir=tmp, dpi=dpi)
        warped = cv2.imread(str(tmp / "warped.png"))
        if warped is None:
            raise RuntimeError(f"could not load warped image for {scan_path}")
        answers = {int(k): v for k, v in result["answers"].items()}

        for b in template["bubbles"]:
            cx = int(b["center_mm"][0] * px_per_mm)
            cy = int(b["center_mm"][1] * px_per_mm)
            r = max(3, int(b["radius_mm"] * px_per_mm))

            if by_q is not None:
                d = by_q.get(b["q"])
                if not d:
                    # No graded detail for this row (e.g. a question past the
                    # test's length on a legacy sheet) — leave it unmarked.
                    continue
                status = d.get("status")
                if status == "correct":
                    if b["option"] == d.get("correct"):
                        _circle(warped, cx, cy, r)
                elif status == "incorrect":
                    if b["option"] == d.get("given"):
                        _x(warped, cx, cy, r)
                elif status in ("blank", "multi"):
                    # No single bubble to point at, so flag the whole row.
                    _x(warped, cx, cy, r)
                continue

            # Legacy fallback: mark detected fills without judging correctness.
            ans = answers.get(b["q"])
            if ans in ("BLANK", "MULTI"):
                _x(warped, cx, cy, r)
            elif ans == b["option"]:
                _circle(warped, cx, cy, r)

    # OpenCV uses BGR; PIL expects RGB. Then save the single page as a PDF.
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PDF", resolution=float(dpi))
    return buf.getvalue()


def _safe_filename(stem: str) -> str:
    keep = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in stem).strip()
    return (keep or "results").replace(" ", "_")


def _refresh_missed_details(
    full: dict, template: dict, answer_key: dict | None, field_test_answers: dict | None = None
) -> dict:
    """Return the submission's stored score with each section's per-question
    ``details`` re-graded against the CURRENT answer key.

    Why: when an answer key is completed *after* submissions were already graded
    (e.g. previously-missing question numbers are filled in), the stored
    ``details`` still reflect the key as it was at grade time — so the newly
    gradeable questions never show up in the "Missed questions" list or on the
    overlay. Re-grading the stored answers here refreshes only those two things.
    The scaled and composite scores are deliberately left exactly as originally
    calculated (we copy them through untouched), so completing a key never
    silently changes a student's reported score.
    """
    stored = full.get("score") or {}
    answers = {int(k): v for k, v in (full.get("answers") or {}).items()}
    if not answer_key or not answers:
        return stored
    # Grade against the scored key PLUS the field-test answers so `details`
    # covers every question — field-test ones flagged scored=False. The scaled/
    # composite scores are taken from the stored (scored-only) grade untouched.
    display_key = merge_field_test(answer_key, field_test_answers)
    fresh = grade_answers(answers, template, display_key, not_scored=field_test_answers)
    merged = dict(stored)
    merged_sections: dict[str, dict] = {}
    for sec, info in (stored.get("sections") or {}).items():
        new_info = dict(info)
        if sec in fresh:
            new_info["details"] = fresh[sec].get("details", [])
        merged_sections[sec] = new_info
    merged["sections"] = merged_sections
    return merged


def send_feedback_for_assignment(
    teacher_email: str,
    course_id: str,
    coursework_id: str,
    *,
    test_name: str | None = None,
    only_students: list[str] | None = None,
    teacher_name: str | None = None,
    include_overlay: bool = True,
    dry_run: bool = False,
    template_path: Path | str | None = None,
    reference_path: Path | str | None = None,
    delete_scans: bool = True,
) -> dict:
    """For each graded submission, compose + send (or preview) the student's email.

    ``delete_scans``: sending feedback is the end of a grading cycle, so by
    default the assignment's stored scans are deleted afterwards (only when
    every selected email actually sent). Chunked per-student callers pass
    False and delete once at the end of the whole batch instead.
    """
    # Look up the assignment's scope (partial vs full) so build_report can
    # tailor the email body. Missing app_assignments row = treat as full.
    app_asg = dbmod.get_app_assignment(course_id, coursework_id)
    scope = (app_asg or {}).get("scope")
    test_id = (app_asg or {}).get("test_id")

    # Pick the OMR template that matches the test's format. Callers can
    # still override via the template_path / reference_path kwargs.
    if (template_path is None or reference_path is None) and test_id:
        from .submissions import template_for_test
        auto_tpl, auto_ref = template_for_test(test_id)
        template_path = template_path or auto_tpl
        reference_path = reference_path or auto_ref
    template_path = Path(template_path or DEFAULT_TEMPLATE)
    reference_path = Path(reference_path or DEFAULT_REFERENCE)

    # Load the CURRENT answer key + template so each email's missed-question
    # list and overlay reflect the key as it stands now (which may have been
    # completed since these submissions were graded). Stored scores are left
    # as-is; see _refresh_missed_details.
    _test = dbmod.get_test(test_id) if test_id else None
    current_key = (_test or {}).get("answer_key")
    current_field_test = (_test or {}).get("field_test_answers")
    template_dict = json.loads(template_path.read_text())

    # Latest submission per student, optionally filtered by `only_students`.
    rows = dbmod.list_submissions(course_id=course_id, coursework_id=coursework_id)
    latest_per_student: dict[str, dict] = {}
    for r in rows:
        sid = r["student_id"]
        if sid and sid not in latest_per_student:
            latest_per_student[sid] = r

    if only_students:
        wanted = {s.strip() for s in only_students if s and s.strip()}
        if wanted:
            latest_per_student = {
                sid: r for sid, r in latest_per_student.items()
                if sid in wanted or (r.get("student_email") or "") in wanted
            }

    results: list[dict] = []
    for sid, r in latest_per_student.items():
        full = dbmod.get_submission(r["id"])
        if not full:
            continue
        student_email = full.get("student_email")
        student_name = full.get("student_name") or "there"
        if not student_email:
            results.append({
                "student_id": sid, "name": student_name,
                "status": "no_email_on_record",
            })
            continue

        first = student_name.split()[0] if student_name and student_name != "there" else "there"
        # Refresh the missed-question list/overlay against the current key
        # without disturbing the stored scaled/composite scores.
        email_score = _refresh_missed_details(
            full, template_dict, current_key, field_test_answers=current_field_test
        )
        email_full = {**full, "score": email_score}
        report = build_report(email_full, test_name=test_name, scope=scope)
        body_parts = [f"Hi {first},", "", report]
        if include_overlay:
            body_parts += [
                "",
                "Attached is a marked-up copy of your scanned answer sheet. Green "
                "circles mark the questions you got right; red Xs mark the ones "
                "you missed — answered incorrectly, left blank, or multi-marked.",
            ]
        if teacher_name:
            body_parts.extend(["", f"— {teacher_name}"])
        body = "\n".join(body_parts)
        subject = f"{test_name or 'ACT Practice Test'} — your results"

        # Build the overlay attachment (best-effort — empty list if it fails).
        attachments = []
        overlay_error = None
        if include_overlay and not dry_run:
            scan_path = _student_scan_tempfile(teacher_email, course_id, coursework_id, sid)
            if scan_path is None:
                overlay_error = "no scan available (not in DB, and Drive refetch failed)"
            else:
                # Flatten every section's per-question detail (refreshed against
                # the current key) so the overlay colors each bubble correctly.
                details = [
                    d
                    for sec in (email_score.get("sections") or {}).values()
                    for d in (sec.get("details") or [])
                ]
                try:
                    pdf_bytes = build_overlay_pdf(
                        Path(scan_path), template_path, reference_path, details=details
                    )
                    fname = f"{_safe_filename(student_name)}_results.pdf"
                    attachments.append((fname, pdf_bytes, "application/pdf"))
                except Exception as e:  # noqa: BLE001
                    overlay_error = f"{type(e).__name__}: {e}"
                finally:
                    try:
                        os.unlink(scan_path)
                    except OSError:
                        pass

        if dry_run:
            results.append({
                "student_id": sid, "name": student_name, "email": student_email,
                "status": "dry_run",
                "subject": subject, "body": body,
                "would_attach_overlay": include_overlay,
            })
            continue
        try:
            send_email(teacher_email, student_email, subject, body, attachments=attachments or None)
            results.append({
                "student_id": sid, "name": student_name, "email": student_email,
                "status": "sent",
                "attached_overlay": bool(attachments),
                "overlay_error": overlay_error,
            })
        except Exception as e:  # noqa: BLE001
            results.append({
                "student_id": sid, "name": student_name, "email": student_email,
                "status": "send_failed",
                "error": f"{type(e).__name__}: {e}",
            })

    # End of the grading cycle: once every selected email went out, the stored
    # scans have served their purpose — drop them to keep the DB tiny. (They
    # can always be refetched from Drive if feedback is ever re-sent.)
    scans_deleted = 0
    if delete_scans and not dry_run:
        any_sent = any(r["status"] == "sent" for r in results)
        any_failed = any(r["status"] == "send_failed" for r in results)
        if any_sent and not any_failed:
            scans_deleted = dbmod.delete_assignment_scans(course_id, coursework_id)

    return {"results": results, "scans_deleted": scans_deleted}
