"""Phone-camera submissions: HEIC conversion + trying every attached file."""

import io
import json

import pytest

from bubble_grader.config import DATA_DIR
from bubble_grader.submissions import _normalize_scan

SHEETS = DATA_DIR / "sheets"


def _make_heic_bytes() -> bytes:
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    img = Image.new("RGB", (64, 48), (200, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="HEIF")
    return buf.getvalue()


def test_heic_converts_to_jpeg():
    content, mime, name = _normalize_scan(_make_heic_bytes(), "image/heic", "IMG_0042.heic")
    assert mime == "image/jpeg" and name == "IMG_0042.jpg"
    from PIL import Image
    img = Image.open(io.BytesIO(content))
    assert img.format == "JPEG" and img.size == (64, 48)


def test_heic_detected_by_extension_alone():
    _, mime, name = _normalize_scan(_make_heic_bytes(), None, "photo.HEIF")
    assert mime == "image/jpeg" and name == "photo.jpg"


def test_non_heic_passes_through_untouched():
    data = b"%PDF-1.4 fake"
    content, mime, name = _normalize_scan(data, "application/pdf", "scan.pdf")
    assert (content, mime, name) == (data, "application/pdf", "scan.pdf")


@pytest.mark.parametrize("stem", ["act_sheet_new"])
def test_second_photo_rescues_a_blurry_first(db, tmp_path, stem):
    """Student attaches junk first and a clean shot second: grading must fall
    through to the readable file instead of failing the submission."""
    import cv2
    from bubble_grader.sheet import simulate_fill
    from bubble_grader.submissions import grade_classroom_assignment

    template_path = SHEETS / f"{stem}.template.json"
    template = json.loads(template_path.read_text())

    options_by_q: dict[int, list[str]] = {}
    sections: dict[int, tuple[str, int]] = {}
    for b in template["bubbles"]:
        options_by_q.setdefault(b["q"], []).append(b["option"])
        sections[b["q"]] = (b["section"], b["q_in_test"])
    answers = {q: opts[0] for q, opts in options_by_q.items()}

    # Key + scaler over every question (no field-test in this synthetic test).
    key: dict = {}
    counts: dict = {}
    for q, (sec, qit) in sections.items():
        key.setdefault(sec, {})[str(qit)] = answers[q]
        counts[sec] = counts.get(sec, 0) + 1
    scaler = {sec: {str(r): max(1, round(36 * r / n)) for r in range(n + 1)}
              for sec, n in counts.items()}
    db.upsert_test("t1", name="Synthetic")
    db.set_test_answer_key("t1", key)
    db.set_test_scaler("t1", scaler)

    reference = cv2.imread(str(SHEETS / f"{stem}.reference.png"), cv2.IMREAD_GRAYSCALE)
    filled = simulate_fill(template, answers,
                           dpi=int(template.get("source_dpi", 300)), base_image=reference)
    buf = io.BytesIO()
    filled.save(buf, format="PNG")

    meta = dict(student_name="Kid", student_email="k@x.com",
                classroom_submission_id="cs", state="TURNED_IN")
    db.store_scan_file("c", "cw", "s1", "f_blurry", name="blurry.png",
                       mime="image/png", content=b"this is not an image", error=None, **meta)
    db.store_scan_file("c", "cw", "s1", "f_good", name="good.png",
                       mime="image/png", content=buf.getvalue(), error=None, **meta)

    result = grade_classroom_assignment(
        "teacher@test", "c", "cw", "t1", template_path, refetch=False,
    )
    (r,) = result["results"]
    assert r["status"] == "graded", r.get("error")
    assert r["file"] == "good.png"
    assert r["composite"] == 36  # perfect synthetic sheet
    assert db.list_grade_errors("c", "cw") == {}


def test_all_files_unreadable_reports_each_attempt(db):
    from bubble_grader.submissions import grade_classroom_assignment

    db.upsert_test("t1", name="Synthetic")
    db.set_test_answer_key("t1", {"Test 1": {"1": "A"}})
    meta = dict(student_name="Kid", student_email="k@x.com",
                classroom_submission_id="cs", state="TURNED_IN")
    db.store_scan_file("c", "cw", "s1", "f1", name="one.png",
                       mime="image/png", content=b"junk1", error=None, **meta)
    db.store_scan_file("c", "cw", "s1", "f2", name="two.png",
                       mime="image/png", content=b"junk2", error=None, **meta)

    result = grade_classroom_assignment(
        "teacher@test", "c", "cw", "t1", SHEETS / "act_sheet_new.template.json",
        refetch=False,
    )
    (r,) = result["results"]
    assert r["status"] == "grade_failed"
    assert "none of 2 file(s) readable" in r["error"]
    assert db.list_grade_errors("c", "cw")["s1"]["status"] == "grade_failed"
