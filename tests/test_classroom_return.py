"""Auto-pushed draft grades + returning work inside Classroom (no email)."""

import bubble_grader.submissions as submissions
import bubble_grader.classroom_return as cr
from bubble_grader.config import DATA_DIR, SCOPES
from bubble_grader.server import _finish_rank

SHEETS = DATA_DIR / "sheets"
COUNTS = {"Test 1": 40, "Test 2": 41, "Test 3": 27, "Test 4": 34}


def _seed(db, with_assignment: bool):
    db.upsert_test("t", name="T")
    db.set_test_answer_key("t", {s: {str(q): "A" for q in range(1, n + 1)} for s, n in COUNTS.items()})
    db.set_test_scaler("t", {s: {str(r): 1 for r in range(0, n + 1)} for s, n in COUNTS.items()})
    if with_assignment:
        db.record_app_assignment("c", "cw", test_id="t", title="T")
    db.store_scan_file("c", "cw", "s1", "f1", name="x.png", mime="image/png", content=b"junk",
                       error=None, student_name="Kid", student_email="k@x",
                       classroom_submission_id="cs1", state="TURNED_IN")


def _fake_read(*a, **k):
    return {"answers": {1: "A"}, "fills": {},
            "match_info": {"n_inliers": 900, "inlier_ratio": 0.6}}


def test_new_scopes_present():
    assert "https://www.googleapis.com/auth/classroom.announcements" in SCOPES
    assert "https://www.googleapis.com/auth/drive.file" in SCOPES


def test_grading_pushes_draft_grade_for_app_owned_assignment(db, monkeypatch):
    import bubble_grader.omr as omr
    monkeypatch.setattr(omr, "read_sheet_fm", _fake_read)
    calls = []
    def fake_release(email, course_id, cw_id, **kw):
        calls.append(kw)
        return {"results": [{"student_id": sid, "status": "draft_set"} for sid in kw["only_students"]]}
    monkeypatch.setattr(submissions, "release_grades", fake_release)
    _seed(db, with_assignment=True)

    (r,) = submissions.grade_classroom_assignment(
        "t@x", "c", "cw", "t", SHEETS / "act_sheet_scored.template.json", refetch=False
    )["results"]
    assert r["status"] == "graded" and r["draft_pushed"] is True
    assert calls == [{"draft_only": True, "only_students": ["s1"]}]   # draft, never returned


def test_no_push_when_assignment_not_app_owned(db, monkeypatch):
    import bubble_grader.omr as omr
    monkeypatch.setattr(omr, "read_sheet_fm", _fake_read)
    monkeypatch.setattr(submissions, "release_grades",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    _seed(db, with_assignment=False)
    (r,) = submissions.grade_classroom_assignment(
        "t@x", "c", "cw", "t", SHEETS / "act_sheet_scored.template.json", refetch=False
    )["results"]
    assert r["status"] == "graded" and "draft_pushed" not in r


def test_return_posts_sheet_then_returns_grade(monkeypatch):
    order = []
    monkeypatch.setattr(cr, "student_feedback", lambda *a, **k: {
        "status": "ok", "student_name": "Kid Person", "student_email": "k@x",
        "test_name": "Seeing Streams", "report": "Composite: 30/36", "pdf": b"%PDF", "overlay_error": None})
    monkeypatch.setattr(cr, "upload_pdf", lambda email, name, data: (order.append("upload"), {"id": "FILE1"})[1])
    def fake_ann(email, course_id, text, *, student_ids, drive_file_id):
        order.append("announce")
        assert student_ids == ["s1"] and drive_file_id == "FILE1"
        assert "Hi Kid," in text and "Composite: 30/36" in text and "— Ms. T" in text
        return {"id": "ANN1"}
    monkeypatch.setattr(cr, "create_student_announcement", fake_ann)
    def fake_release(email, course_id, cw_id, **kw):
        order.append("release")
        assert kw == {"return_to_student": True, "only_students": ["s1"]}
        return {"results": [{"student_id": "s1", "status": "returned"}]}
    monkeypatch.setattr(cr, "release_grades", fake_release)

    r = cr.return_to_student("t@x", "c", "cw", "s1", test_name="Seeing Streams", teacher_name="Ms. T")
    assert order == ["upload", "announce", "release"]
    assert r["status"] == "returned" and r["attached_overlay"] and r["announcement_id"] == "ANN1"


def test_failed_post_returns_nothing(monkeypatch):
    """If the private post can't be created (e.g. missing permission), the grade
    must NOT be returned — the teacher fixes and retries with no half-done state."""
    monkeypatch.setattr(cr, "student_feedback", lambda *a, **k: {
        "status": "ok", "student_name": "Kid", "student_email": None, "test_name": "T",
        "report": "r", "pdf": None, "overlay_error": "no scan"})
    def boom(*a, **k): raise RuntimeError("403 insufficient scopes")
    monkeypatch.setattr(cr, "create_student_announcement", boom)
    monkeypatch.setattr(cr, "release_grades",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not release")))
    try:
        cr.return_to_student("t@x", "c", "cw", "s1")
        assert False, "expected the announcement error to propagate"
    except RuntimeError as e:
        assert "insufficient scopes" in str(e)


def test_finished_first_ranking():
    assert _finish_rank({"composite": 28, "classroom_state": "TURNED_IN"}) == 0
    assert _finish_rank({"composite": None, "partial": {"raw": 3}, "classroom_state": "CREATED"}) == 0
    assert _finish_rank({"composite": None, "partial": None, "classroom_state": "TURNED_IN"}) == 1
    assert _finish_rank({"composite": None, "partial": None, "classroom_state": "RETURNED"}) == 1
    assert _finish_rank({"composite": None, "partial": None, "classroom_state": "CREATED"}) == 2
