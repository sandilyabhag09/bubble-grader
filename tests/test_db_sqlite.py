"""Storage-layer roundtrips on the SQLite fallback (CI has no Postgres)."""


def test_test_key_storage_roundtrip(db):
    db.upsert_test("t1", name="Test One", notes="n")
    db.set_test_answer_key("t1", {"Test 1": {"1": "A", "2": "F"}})
    db.set_test_field_test_answers("t1", {"Test 1": {"3": "B"}})
    db.set_test_scaler("t1", {"Test 1": {"0": 1, "2": 36}})
    t = db.get_test("t1")
    assert t["answer_key"]["Test 1"]["1"] == "A"
    assert t["field_test_answers"]["Test 1"]["3"] == "B"
    assert t["scaler"]["Test 1"]["2"] == 36
    # field-test answers must NOT be inside the grading key
    assert "3" not in t["answer_key"]["Test 1"]


def test_scan_store_roundtrip(db):
    db.store_scan_file("c", "cw", "s1", "f1", name="scan.pdf",
                       mime="application/pdf", content=b"%PDF-fake",
                       error=None, student_name="Kid", student_email="k@x.com",
                       classroom_submission_id="cs1", state="TURNED_IN")
    db.store_scan_file("c", "cw", "s1", "f2", name="bad.pdf", mime=None,
                       content=None, error="boom", student_name="Kid",
                       student_email="k@x.com", classroom_submission_id="cs1",
                       state="TURNED_IN")
    scan = db.get_student_scan("c", "cw", "s1")
    assert scan is not None and scan["content"] == b"%PDF-fake"

    students = db.list_scan_students("c", "cw")
    assert "s1" in students and len(students["s1"]["files"]) == 2

    assert db.delete_assignment_scans("c", "cw") >= 1
    assert db.get_student_scan("c", "cw", "s1") is None
    db.purge_old_scans()  # smoke: must not raise on empty store


def test_grade_errors_upsert_and_clear(db):
    db.record_grade_error("c", "cw", "s1", status="no_files", error=None)
    db.record_grade_error("c", "cw", "s1", status="grade_failed", error="blurry")
    errs = db.list_grade_errors("c", "cw")
    assert errs["s1"]["status"] == "grade_failed" and errs["s1"]["error"] == "blurry"
    db.clear_grade_error("c", "cw", "s1")
    assert db.list_grade_errors("c", "cw") == {}


def test_submission_insert_and_latest_first(db):
    db.upsert_test("t1", name="Test One")
    a = db.add_submission(test_id="t1", answers={1: "A"}, score={"composite": 20},
                          student_id="s1", student_name="Kid", student_email="k@x.com",
                          course_id="c", coursework_id="cw", classroom_submission_id="cs")
    b = db.add_submission(test_id="t1", answers={1: "B"}, score={"composite": 22},
                          student_id="s1", student_name="Kid", student_email="k@x.com",
                          course_id="c", coursework_id="cw", classroom_submission_id="cs")
    rows = db.list_submissions(course_id="c", coursework_id="cw")
    assert [r["id"] for r in rows[:2]] == [b, a]  # newest first
