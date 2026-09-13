"""Which bubble sheet a test grades against, and refusing to grade a scan
that matched no sheet convincingly (the 'composite of 3' incident)."""

import pytest

from bubble_grader.config import DATA_DIR
from bubble_grader.submissions import (
    MIN_MATCH_INLIERS,
    SHEET_LAYOUTS,
    _match_ok,
    template_stem_for_test,
)

SHEETS = DATA_DIR / "sheets"


def _key(counts):
    return {sec: {str(q): "A" for q in range(1, n + 1)} for sec, n in counts.items()}


def test_every_known_sheet_has_template_and_reference():
    for stem in SHEET_LAYOUTS:
        assert (SHEETS / f"{stem}.template.json").exists(), stem
        assert (SHEETS / f"{stem}.reference.png").exists(), stem


@pytest.mark.parametrize("scored,field_test,expected", [
    ({"Test 1": 75, "Test 2": 60, "Test 3": 40, "Test 4": 40}, None, "act_sheet"),
    ({"Test 1": 40, "Test 2": 41, "Test 3": 27, "Test 4": 34}, None, "act_sheet_scored"),   # Seeing Streams
    ({"Test 1": 40, "Test 2": 41, "Test 3": 27, "Test 4": 34},                               # Alex Atala:
     {"Test 1": 10, "Test 2": 4, "Test 3": 9, "Test 4": 6}, "act_sheet_new"),                # scored + field-test = 50/45/36/40
])
def test_sheet_is_chosen_by_total_questions(db, scored, field_test, expected):
    db.upsert_test("t", name="T")
    db.set_test_answer_key("t", _key(scored))
    if field_test:
        db.set_test_field_test_answers("t", _key(field_test))
    assert template_stem_for_test("t") == expected


def test_unknown_key_shape_falls_back_to_format_heuristic(db):
    db.upsert_test("t", name="T")
    db.set_test_answer_key("t", _key({"Test 1": 45}))   # partial, odd count
    assert template_stem_for_test("t") == "act_sheet_new"


def test_match_quality_floor():
    assert _match_ok({"match_info": {"n_inliers": 49, "inlier_ratio": 0.31}})    # weakest real match seen
    assert not _match_ok({"match_info": {"n_inliers": 27, "inlier_ratio": 0.10}})  # graded a 3 — never again
    assert not _match_ok({"match_info": {"n_inliers": 38, "inlier_ratio": 0.25}})
    assert not _match_ok({"match_info": {"n_inliers": MIN_MATCH_INLIERS + 100, "inlier_ratio": 0.05}})


def test_wrong_sheet_is_refused_not_graded(db, monkeypatch):
    """Every known sheet matches weakly -> grade_failed with a clear reason,
    never a stored bogus score."""
    import bubble_grader.omr as omr
    from bubble_grader.submissions import grade_classroom_assignment
    monkeypatch.setattr(omr, "read_sheet_fm", lambda *a, **k: {
        "answers": {1: "A"}, "fills": {}, "match_info": {"n_inliers": 27, "inlier_ratio": 0.10}})
    db.upsert_test("t", name="T")
    db.set_test_answer_key("t", _key({"Test 1": 40, "Test 2": 41, "Test 3": 27, "Test 4": 34}))
    db.store_scan_file("c", "cw", "s1", "f1", name="x.png", mime="image/png", content=b"junk",
                       error=None, student_name="Kid", student_email="k@x", classroom_submission_id="cs",
                       state="TURNED_IN")
    result = grade_classroom_assignment("t@x", "c", "cw", "t", SHEETS / "act_sheet_scored.template.json",
                                        refetch=False)
    (r,) = result["results"]
    assert r["status"] == "grade_failed" and "didn't match any known layout" in r["error"]
    assert db.list_submissions(course_id="c", coursework_id="cw") == []


def test_scan_of_a_different_sheet_is_graded_with_that_sheet(db, monkeypatch):
    """Expected sheet matches weakly, another matches strongly -> use the other
    and record which one on the grade."""
    import bubble_grader.omr as omr
    from bubble_grader.submissions import grade_classroom_assignment
    def fake_read(path, tpl, ref, **k):
        strong = "act_sheet_new" in str(tpl)
        return {"answers": {1: "A"}, "fills": {},
                "match_info": {"n_inliers": 900 if strong else 30, "inlier_ratio": 0.6 if strong else 0.2}}
    monkeypatch.setattr(omr, "read_sheet_fm", fake_read)
    db.upsert_test("t", name="T")
    db.set_test_answer_key("t", _key({"Test 1": 40, "Test 2": 41, "Test 3": 27, "Test 4": 34}))
    db.set_test_scaler("t", {s: {str(r): 1 for r in range(0, n + 1)} for s, n in
                             {"Test 1": 40, "Test 2": 41, "Test 3": 27, "Test 4": 34}.items()})
    db.store_scan_file("c", "cw", "s1", "f1", name="x.png", mime="image/png", content=b"junk",
                       error=None, student_name="Kid", student_email="k@x", classroom_submission_id="cs",
                       state="TURNED_IN")
    result = grade_classroom_assignment("t@x", "c", "cw", "t", SHEETS / "act_sheet_scored.template.json",
                                        refetch=False)
    (r,) = result["results"]
    assert r["status"] == "graded", r.get("error")
    stored = db.get_submission(r["submission_id"])
    assert stored["score"]["sheet_template"] == "act_sheet_new"
