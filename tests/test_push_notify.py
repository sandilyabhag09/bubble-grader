"""Classroom push notifications → grade the one student who just turned in."""

import base64
import json

import bubble_grader.push_notify as pn
from bubble_grader.config import SCOPES


def _envelope(event: dict) -> dict:
    return {"message": {"data": base64.b64encode(json.dumps(event).encode()).decode(),
                        "messageId": "1"}, "subscription": "projects/p/subscriptions/s"}


def _turnin_event(course="c", cw="cw", sub="sub1"):
    return {"collection": "courses.courseWork.studentSubmissions", "eventType": "MODIFIED",
            "resourceId": {"courseId": course, "courseWorkId": cw, "id": sub}}


def test_scope_present():
    assert "https://www.googleapis.com/auth/classroom.push-notifications" in SCOPES


def test_decode_envelope():
    assert pn.decode_pubsub_push(_envelope({"a": 1})) == {"a": 1}
    assert pn.decode_pubsub_push({}) is None
    assert pn.decode_pubsub_push({"message": {"data": "!!notbase64"}}) is None


def test_ignores_events_for_unowned_assignments(db):
    r = pn.handle_notification(_envelope(_turnin_event()))
    assert r["status"] == "ignored" and "app-owned" in r["reason"]


def test_ignores_coursework_events(db):
    ev = {"collection": "courses.courseWork", "eventType": "CREATED",
          "resourceId": {"courseId": "c", "id": "cw"}}
    assert pn.handle_notification(_envelope(ev))["status"] == "ignored"


def test_grades_fresh_turn_in_only(db, monkeypatch):
    db.record_app_assignment("c", "cw", test_id="t", title="T", created_by="t@x")
    monkeypatch.setattr(pn.dbmod, "list_teachers", lambda: ["t@x"])
    states = {"state": "TURNED_IN", "userId": "s1", "updateTime": "2026-09-13T10:00:00Z"}
    monkeypatch.setattr(pn, "get_submission", lambda email, c, cw, sid: dict(states))
    calls = []
    import bubble_grader.submissions as submissions
    monkeypatch.setattr(submissions, "template_for_test", lambda tid: ("tpl", "ref"))
    def fake_grade(email, c, cw, tid, tpl, **kw):
        calls.append((email, c, cw, tid, kw))
        return {"results": [{"student_id": "s1", "name": "Kid", "status": "graded",
                             "composite": 30, "grade_pushed": True}]}
    monkeypatch.setattr(submissions, "grade_classroom_assignment", fake_grade)

    r = pn.handle_notification(_envelope(_turnin_event()))
    assert r["status"] == "graded" and r["composite"] == 30 and r["grade_pushed"] is True
    assert calls == [("t@x", "c", "cw", "t", {"only_students": ["s1"]})]

    # Not turned in (e.g. we just RETURNED it) → nothing happens.
    states["state"] = "RETURNED"
    assert pn.handle_notification(_envelope(_turnin_event()))["status"] == "ignored"
    assert len(calls) == 1


def test_duplicate_delivery_is_ignored(db, monkeypatch):
    db.record_app_assignment("c", "cw", test_id="t", title="T", created_by="t@x")
    monkeypatch.setattr(pn.dbmod, "list_teachers", lambda: ["t@x"])
    db.add_submission("t", {1: "A"}, {"composite": 30},
                      student_id="s1", course_id="c", coursework_id="cw")
    monkeypatch.setattr(pn, "get_submission", lambda *a: {
        "state": "TURNED_IN", "userId": "s1", "updateTime": "2020-01-01T00:00:00Z"})
    import bubble_grader.submissions as submissions
    monkeypatch.setattr(submissions, "grade_classroom_assignment",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not grade")))
    r = pn.handle_notification(_envelope(_turnin_event()))
    assert r["status"] == "ignored" and "already graded" in r["reason"]


def test_ensure_registrations_creates_and_renews(db, monkeypatch):
    monkeypatch.setattr(pn, "PUBSUB_TOPIC", "projects/p/topics/t")
    db.record_app_assignment("c1", "cw", test_id="t", title="T", created_by="t@x")
    db.record_app_assignment("c2", "cw", test_id="t", title="T", created_by="t@x")
    monkeypatch.setattr(pn.dbmod, "list_teachers", lambda: ["t@x"])
    created, deleted = [], []
    monkeypatch.setattr(pn, "create_registration", lambda email, c, topic: (
        created.append((email, c, topic)) or {"registrationId": f"reg-{c}",
                                              "expiryTime": "2099-01-01T00:00:00Z"}))
    monkeypatch.setattr(pn, "delete_registration", lambda email, rid: deleted.append(rid))

    s = pn.ensure_registrations()
    assert s["registered"] == 2 and s["failed"] == 0
    assert {c for _, c, _ in created} == {"c1", "c2"}
    assert db.get_push_registration("c1")["registration_id"] == "reg-c1"

    # Fresh registrations are left alone on the next pass.
    s = pn.ensure_registrations()
    assert s["registered"] == 0 and len(created) == 2

    # One about to expire → old one deleted, new one created.
    db.upsert_push_registration("c1", registration_id="reg-c1", teacher_email="t@x",
                                topic="projects/p/topics/t", expiry_time="2000-01-01T00:00:00Z")
    s = pn.ensure_registrations()
    assert s["registered"] == 1 and deleted == ["reg-c1"] and len(created) == 3


def test_ensure_registrations_disabled_without_topic(db, monkeypatch):
    monkeypatch.setattr(pn, "PUBSUB_TOPIC", "")
    assert pn.ensure_registrations().get("disabled") is True


def test_webhook_requires_token(monkeypatch):
    from starlette.testclient import TestClient
    import bubble_grader.server as server
    monkeypatch.setattr(server, "CLASSROOM_PUSH_TOKEN", "sekrit")
    with TestClient(server.app) as client:
        assert client.post("/hooks/classroom?token=wrong", json={}).status_code == 404
        r = client.post("/hooks/classroom?token=sekrit", json={})
        assert r.status_code == 200 and r.json()["status"] == "ignored"


def test_cron_ping_purges_old_scans(db, monkeypatch):
    """Quiet stretches (nothing graded) must not let scans outlive retention."""
    from starlette.testclient import TestClient
    import bubble_grader.server as server
    import bubble_grader.auto_grade as auto_grade
    monkeypatch.setattr(server, "AUTO_GRADE_TOKEN", "tok")
    monkeypatch.setattr(auto_grade, "run_auto_grade_once", lambda **k: {
        "assignments_checked": 0, "students_graded": 0, "students_failed": 0, "details": []})
    db.store_scan_file("c", "cw", "s1", "old", name="x.png", mime="image/png", content=b"x",
                       error=None, student_name="Kid", student_email="k@x",
                       classroom_submission_id="cs1", state="TURNED_IN")
    db.store_scan_file("c", "cw", "s2", "new", name="y.png", mime="image/png", content=b"y",
                       error=None, student_name="Kid2", student_email="k2@x",
                       classroom_submission_id="cs2", state="TURNED_IN")
    from bubble_grader.db_backend import get_conn
    with get_conn() as conn:
        conn.execute("UPDATE scan_files SET fetched_at = '2020-01-01 00:00:00' WHERE file_id = 'old'")
    with TestClient(server.app) as client:
        r = client.get("/cron/auto-grade?token=tok")
    assert r.status_code == 200 and r.json()["scans_purged"] == 1
    assert [s for s in db.list_scan_students("c", "cw")] and db.get_student_scan("c", "cw", "s1") is None
    assert db.get_student_scan("c", "cw", "s2") is not None
