"""Grades go straight to students when a turn-in is graded; finished-first sort."""

import bubble_grader.submissions as submissions
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


def test_no_extra_scopes_needed():
    """Auto-release uses the grade-write scope teachers already granted."""
    assert "https://www.googleapis.com/auth/classroom.coursework.students" in SCOPES
    for s in SCOPES:
        assert "announcements" not in s and "courseworkmaterials" not in s and "drive.file" not in s


def test_grading_returns_grade_to_student_for_app_owned_assignment(db, monkeypatch):
    import bubble_grader.omr as omr
    monkeypatch.setattr(omr, "read_sheet_fm", _fake_read)
    calls = []
    def fake_release(email, course_id, cw_id, **kw):
        calls.append(kw)
        return {"results": [{"student_id": sid, "status": "returned"} for sid in kw["only_students"]]}
    monkeypatch.setattr(submissions, "release_grades", fake_release)
    _seed(db, with_assignment=True)

    (r,) = submissions.grade_classroom_assignment(
        "t@x", "c", "cw", "t", SHEETS / "act_sheet_scored.template.json", refetch=False
    )["results"]
    assert r["status"] == "graded" and r["grade_pushed"] is True
    assert calls == [{"return_to_student": True, "only_students": ["s1"]}]


def test_push_failure_keeps_grade_and_reports(db, monkeypatch):
    import bubble_grader.omr as omr
    monkeypatch.setattr(omr, "read_sheet_fm", _fake_read)
    monkeypatch.setattr(submissions, "release_grades",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Classroom down")))
    _seed(db, with_assignment=True)
    (r,) = submissions.grade_classroom_assignment(
        "t@x", "c", "cw", "t", SHEETS / "act_sheet_scored.template.json", refetch=False
    )["results"]
    assert r["status"] == "graded" and r["grade_pushed"] is False
    assert "Classroom down" in r["push_error"]
    assert db.list_submissions(course_id="c", coursework_id="cw")  # grade still saved


def test_no_push_when_assignment_not_app_owned(db, monkeypatch):
    import bubble_grader.omr as omr
    monkeypatch.setattr(omr, "read_sheet_fm", _fake_read)
    monkeypatch.setattr(submissions, "release_grades",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    _seed(db, with_assignment=False)
    (r,) = submissions.grade_classroom_assignment(
        "t@x", "c", "cw", "t", SHEETS / "act_sheet_scored.template.json", refetch=False
    )["results"]
    assert r["status"] == "graded" and "grade_pushed" not in r


def test_finished_first_ranking():
    assert _finish_rank({"composite": 28, "classroom_state": "TURNED_IN"}) == 0
    assert _finish_rank({"composite": None, "partial": {"raw": 3}, "classroom_state": "CREATED"}) == 0
    assert _finish_rank({"composite": None, "partial": None, "classroom_state": "TURNED_IN"}) == 1
    assert _finish_rank({"composite": None, "partial": None, "classroom_state": "RETURNED"}) == 1
    assert _finish_rank({"composite": None, "partial": None, "classroom_state": "CREATED"}) == 2
