"""Build per-student feedback reports + send them as personalized emails.

Report format:
    English: <scaled>, Missed questions: <q_in_test list>
    Math: ...
    Reading: ...
    Science: ...

"Missed" = incorrect + BLANK + MULTI (everything that didn't earn credit).

Each email optionally attaches a per-student PDF: the warped scan with green
circles over detected fills and red Xs over BLANK/MULTI questions, so the
student can see exactly what the reader thought they marked.
"""

import io
import json
import tempfile
from pathlib import Path

import cv2
from PIL import Image

from . import db as dbmod
from .config import DATA_DIR
from .gmail import send_email
from .omr import read_sheet_fm


SECTION_DISPLAY = {
    "Test 1": "English",
    "Test 2": "Math",
    "Test 3": "Reading",
    "Test 4": "Science",
}
SECTION_ORDER = ["Test 1", "Test 2", "Test 3", "Test 4"]

# Default OMR template + reference. Could be made configurable later.
DEFAULT_TEMPLATE = Path("data/sheets/act_sheet.template.json")
DEFAULT_REFERENCE = Path("data/sheets/act_sheet.reference.png")


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
        missed = sorted(
            d["q_in_test"] for d in info.get("details", [])
            if d.get("status") in ("incorrect", "blank", "multi")
        )
        missed_str = ", ".join(str(q) for q in missed) if missed else "none"
        lines.append(f"{display}: {score_str}, Missed questions: {missed_str}")

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

    # Missed questions within the scope range only.
    sec_info = sections.get(section_key) or {}
    missed = sorted(
        d["q_in_test"] for d in sec_info.get("details", [])
        if (d.get("status") in ("incorrect", "blank", "multi")
            and q_start <= d.get("q_in_test", 0) <= q_end)
    )
    missed_str = ", ".join(str(q) for q in missed) if missed else "none"

    score_line = (
        f"you scored {raw}/{total}" + (f" ({pct}%)" if pct is not None else "")
        if raw is not None else "your score wasn't computed"
    )

    return (
        f"Here are your results from {test_label}:\n"
        f"\n"
        f"On {label} (questions {q_start}–{q_end}) in {section_name}, {score_line}.\n"
        f"\n"
        f"Missed questions: {missed_str}."
    )


def _student_scan_path(course_id: str, coursework_id: str, student_id: str) -> Path | None:
    """Look up the most recently fetched scan file for a student from the manifest."""
    manifest_path = (
        DATA_DIR / "submissions" / course_id / coursework_id / "manifest.json"
    )
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:  # noqa: BLE001
        return None
    info = (manifest.get("students") or {}).get(student_id) or {}
    files = [f for f in info.get("files", []) if "error" not in f and f.get("path")]
    if not files:
        return None
    path = DATA_DIR / files[0]["path"]
    return path if path.exists() else None


def build_overlay_pdf(
    scan_path: Path,
    template_path: Path = DEFAULT_TEMPLATE,
    reference_path: Path = DEFAULT_REFERENCE,
    dpi: int = 200,
) -> bytes:
    """Render the warped scan with green-circle / red-X overlay; return as PDF bytes."""
    template = json.loads(template_path.read_text())
    px_per_mm = dpi / 25.4

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        result = read_sheet_fm(scan_path, template_path, reference_path, debug_dir=tmp, dpi=dpi)
        warped = cv2.imread(str(tmp / "warped.png"))
        if warped is None:
            raise RuntimeError(f"could not load warped image for {scan_path}")
        answers = {int(k): v for k, v in result["answers"].items()}

        for b in template["bubbles"]:
            ans = answers.get(b["q"])
            cx = int(b["center_mm"][0] * px_per_mm)
            cy = int(b["center_mm"][1] * px_per_mm)
            r = max(3, int(b["radius_mm"] * px_per_mm))
            if ans in ("BLANK", "MULTI"):
                cv2.line(warped, (cx - r, cy - r), (cx + r, cy + r), (0, 0, 255), 2)
                cv2.line(warped, (cx - r, cy + r), (cx + r, cy - r), (0, 0, 255), 2)
            elif ans == b["option"]:
                cv2.circle(warped, (cx, cy), r + 2, (0, 200, 0), 2)

    # OpenCV uses BGR; PIL expects RGB. Then save the single page as a PDF.
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PDF", resolution=float(dpi))
    return buf.getvalue()


def _safe_filename(stem: str) -> str:
    keep = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in stem).strip()
    return (keep or "results").replace(" ", "_")


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
    template_path: Path | str = DEFAULT_TEMPLATE,
    reference_path: Path | str = DEFAULT_REFERENCE,
) -> dict:
    """For each graded submission, compose + send (or preview) the student's email."""
    template_path = Path(template_path)
    reference_path = Path(reference_path)

    # Look up the assignment's scope (partial vs full) so build_report can
    # tailor the email body. Missing app_assignments row = treat as full.
    app_asg = dbmod.get_app_assignment(course_id, coursework_id)
    scope = (app_asg or {}).get("scope")

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
        report = build_report(full, test_name=test_name, scope=scope)
        body_parts = [f"Hi {first},", "", report]
        if include_overlay:
            body_parts += [
                "",
                "Attached is a marked-up copy of your scanned answer sheet. Green "
                "circles show the answers our system detected for you; red Xs "
                "mark questions where we couldn't tell (blank or multi-marked). "
                "IF YOU THINK ANY OF THESE ARE WRONG, REACH OUT TO SAILAJA AUNTIE "
                "TO GET THE CORRECT SCORE!!",
            ]
        if teacher_name:
            body_parts.extend(["", f"— {teacher_name}"])
        body = "\n".join(body_parts)
        subject = f"{test_name or 'ACT Practice Test'} — your results"

        # Build the overlay attachment (best-effort — empty list if it fails).
        attachments = []
        overlay_error = None
        if include_overlay and not dry_run:
            scan_path = _student_scan_path(course_id, coursework_id, sid)
            if scan_path is None:
                overlay_error = "no scan file on disk (run `grade-classroom` first)"
            else:
                try:
                    pdf_bytes = build_overlay_pdf(scan_path, template_path, reference_path)
                    fname = f"{_safe_filename(student_name)}_results.pdf"
                    attachments.append((fname, pdf_bytes, "application/pdf"))
                except Exception as e:  # noqa: BLE001
                    overlay_error = f"{type(e).__name__}: {e}"

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

    return {"results": results}
