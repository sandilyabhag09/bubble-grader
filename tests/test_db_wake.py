"""The database must only be woken when there is real work.

Neon's free tier sleeps the database after 5 idle minutes and bills every
minute it's awake; the 5-minute keep-warm ping used to touch the database
every time, which kept it awake around the clock and exhausted the monthly
compute allowance.
"""

import pytest
from starlette.testclient import TestClient

import bubble_grader.db as dbmod
import bubble_grader.db_backend as dbb
import bubble_grader.server as server


def _forbid_db(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("this path must not touch the database")
    monkeypatch.setattr(dbmod, "get_conn", boom)
    monkeypatch.setattr(dbb, "get_conn", boom)


def _sweeps(pings, every=60):
    last, n = None, 0
    for m in pings:
        if server._sweep_due(m, every=every, last_sweep=last):
            last, n = m, n + 1
    return n


@pytest.mark.parametrize("offset", [0, 1, 2, 3, 4])
def test_exactly_one_sweep_per_hour_whatever_the_ping_phase(offset):
    pings = [offset + 5 * k for k in range(24 * 12)]  # one day of 5-minute pings
    assert _sweeps(pings) == 24


@pytest.mark.parametrize("seed", range(5))
def test_jittery_pings_still_sweep_every_hour(seed):
    import random
    rng = random.Random(seed)
    t, pings = rng.uniform(0, 5), []
    while t < 24 * 60:
        pings.append(int(t))
        t += 5 + rng.uniform(-0.75, 0.75)   # UptimeRobot drifts by tens of seconds
    assert _sweeps(pings) == 24


def test_off_hour_ping_does_not_touch_the_database(monkeypatch):
    monkeypatch.setattr(server, "AUTO_GRADE_TOKEN", "tok")
    monkeypatch.setattr(server, "IS_VERCEL", True)
    monkeypatch.setattr(server, "_now_minutes", lambda: 60 * 1000 + 30)
    _forbid_db(monkeypatch)
    import bubble_grader.auto_grade as auto_grade
    monkeypatch.setattr(auto_grade, "run_auto_grade_once",
                        lambda **k: (_ for _ in ()).throw(AssertionError("no sweep")))
    with TestClient(server.app) as client:
        r = client.get("/cron/auto-grade?token=tok")
    assert r.status_code == 200
    assert r.json()["swept"] is False and r.json()["next_sweep_in_minutes"] == 30


def test_top_of_hour_ping_sweeps(db, monkeypatch):
    monkeypatch.setattr(server, "AUTO_GRADE_TOKEN", "tok")
    monkeypatch.setattr(server, "_now_minutes", lambda: 60 * 1000 + 2)
    import bubble_grader.auto_grade as auto_grade
    monkeypatch.setattr(auto_grade, "run_auto_grade_once", lambda **k: {
        "assignments_checked": 0, "students_graded": 0, "students_failed": 0, "details": []})
    with TestClient(server.app) as client:
        r = client.get("/cron/auto-grade?token=tok")
    assert r.status_code == 200 and r.json()["swept"] is True


def test_serverless_cold_start_and_health_pings_do_not_touch_the_database(monkeypatch):
    monkeypatch.setattr(server, "IS_VERCEL", True)
    _forbid_db(monkeypatch)
    with TestClient(server.app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/healthz/deep").status_code == 200


def test_schema_is_created_on_first_real_use(tmp_path, monkeypatch):
    monkeypatch.setattr(dbb, "DB_PATH", tmp_path / "fresh.db")
    monkeypatch.setattr(dbb, "_first_connect_done", False)
    assert dbmod.list_teachers() == []          # no init_db() call: created lazily
    assert dbmod.list_push_registrations() == []


def test_lazy_schema_survives_a_failed_first_request(tmp_path, monkeypatch):
    monkeypatch.setattr(dbb, "DB_PATH", tmp_path / "fresh2.db")
    monkeypatch.setattr(dbb, "_first_connect_done", False)
    with pytest.raises(RuntimeError):
        with dbb.get_conn() as conn:
            raise RuntimeError("request blew up after connecting")
    assert dbmod.list_teachers() == []   # tables still there


# ----- review findings ------------------------------------------------------

def test_unauthenticated_json_api_is_gone(monkeypatch):
    """/teachers/... listed teacher emails, courses and student rosters with no
    sign-in; nothing used it. It must not exist (and must not wake the DB)."""
    _forbid_db(monkeypatch)
    monkeypatch.setattr(server, "IS_VERCEL", True)
    with TestClient(server.app) as client:
        for path in ("/teachers", "/teachers/a@b.c/courses",
                     "/teachers/a@b.c/courses/1/students",
                     "/teachers/a@b.c/courses/1/coursework",
                     "/teachers/a@b.c/courses/1/coursework/2/submissions"):
            assert client.get(path).status_code == 404, path
        assert client.post("/teachers/a@b.c/courses/1/coursework/2/fetch").status_code in (404, 405)


def test_students_left_over_by_a_capped_sweep_are_graded_next_time(monkeypatch):
    import bubble_grader.auto_grade as ag
    monkeypatch.setattr(ag, "_attempted", set())
    monkeypatch.setattr(ag.dbmod, "list_teachers", lambda: ["t@x"])
    monkeypatch.setattr(ag, "list_courses", lambda email: [{"id": "c"}])
    monkeypatch.setattr(ag.dbmod, "list_app_assignments",
                        lambda c: [{"test_id": "t", "coursework_id": "cw", "created_at": None}])
    monkeypatch.setattr(ag.dbmod, "list_submissions", lambda **k: [])
    monkeypatch.setattr(ag, "list_submissions", lambda e, c, cw: [
        {"userId": f"s{i}", "state": "TURNED_IN", "updateTime": "2026-10-17T10:00:00Z"} for i in range(12)])
    monkeypatch.setattr(ag, "template_for_test", lambda t: ("tpl", "ref"))
    graded = []
    def fake_grade(email, c, cw, t, tpl, only_students):
        graded.extend(only_students)
        return {"results": [{"student_id": s, "status": "graded"} for s in only_students]}
    monkeypatch.setattr(ag, "grade_classroom_assignment", fake_grade)

    first = ag.run_auto_grade_once(max_students=10)
    assert first["students_graded"] == 10 and first["truncated"] is True
    second = ag.run_auto_grade_once(max_students=10)
    assert second["students_graded"] == 2
    assert sorted(graded) == sorted(f"s{i}" for i in range(12))


def test_cron_sweep_is_capped_to_fit_the_function_limit(monkeypatch):
    monkeypatch.setattr(server, "AUTO_GRADE_TOKEN", "tok")
    import bubble_grader.auto_grade as ag
    import bubble_grader.push_notify as pn
    seen = {}
    def fake_run(**kw):
        seen.update(kw)
        return {"assignments_checked": 0, "students_graded": 0, "students_failed": 0, "details": []}
    monkeypatch.setattr(ag, "run_auto_grade_once", fake_run)
    monkeypatch.setattr(pn, "ensure_registrations", lambda **k: {})
    monkeypatch.setattr(server.dbmod, "purge_old_scans", lambda: 0)
    with TestClient(server.app) as client:
        assert client.get("/cron/auto-grade?token=tok&force=1").json()["swept"] is True
    assert seen == {"max_students": 25, "time_budget": 180}


def test_cron_without_token_is_404_even_with_odd_params():
    with TestClient(server.app) as client:
        assert client.get("/cron/auto-grade?force=abc").status_code == 404
        assert client.get("/cron/auto-grade?token=nope&force=1").status_code == 404


@pytest.mark.parametrize("raw,expected", [
    (None, 60), ("", 60), ("1h", 60), ("0", 15), ("-60", 15), ("5", 15),
    ("61", 60), ("90", 90), ("100000", 1440),
])
def test_sweep_interval_parsing_never_means_every_ping(raw, expected):
    from bubble_grader.config import _sweep_minutes
    assert _sweep_minutes(raw) == expected


def test_local_poller_stays_off_against_the_shared_database(monkeypatch):
    import bubble_grader.auto_grade as ag
    monkeypatch.setattr(server, "IS_VERCEL", False)
    monkeypatch.setattr(server, "AUTO_GRADE_ON_TURNIN", True)
    monkeypatch.setattr(server.dbmod, "init_db", lambda: None)
    monkeypatch.setattr(server, "start_background_poller", lambda: None)
    monkeypatch.setattr(dbb, "is_postgres", lambda: True)
    monkeypatch.setattr(ag, "start_auto_grade_poller",
                        lambda: (_ for _ in ()).throw(AssertionError("poller must not start")))
    with TestClient(server.app) as client:
        assert client.get("/healthz").status_code == 200


def test_repeat_notifications_do_not_wake_the_database(db, monkeypatch):
    import base64, json
    import bubble_grader.push_notify as pn
    pn.forget_ownership()
    db.record_app_assignment("c", "cw", test_id="t", title="T", created_by="t@x")
    monkeypatch.setattr(pn.dbmod, "list_teachers", lambda: ["t@x"])
    monkeypatch.setattr(pn, "get_submission",
                        lambda *a: {"state": "RETURNED", "userId": "s1"})
    def event(cw):
        ev = {"collection": "courses.courseWork.studentSubmissions", "eventType": "MODIFIED",
              "resourceId": {"courseId": "c", "courseWorkId": cw, "id": "sub1"}}
        return {"message": {"data": base64.b64encode(json.dumps(ev).encode()).decode()}}
    assert pn.handle_notification(event("cw"))["status"] == "ignored"       # owned, not a turn-in
    assert pn.handle_notification(event("other"))["status"] == "ignored"    # not ours
    _forbid_db(monkeypatch)
    assert pn.handle_notification(event("cw"))["reason"].startswith("state ")
    assert "app-owned" in pn.handle_notification(event("other"))["reason"]
    pn.forget_ownership()


def test_robots_txt_keeps_crawlers_off_sign_in():
    with TestClient(server.app) as client:
        r = client.get("/robots.txt")
    assert r.status_code == 200 and "Disallow: /oauth/" in r.text


def test_turn_in_is_graded_against_the_current_test_not_a_cached_one(db, monkeypatch):
    import base64, json
    import bubble_grader.push_notify as pn
    import bubble_grader.submissions as submissions
    pn.forget_ownership()
    db.record_app_assignment("c", "cw", test_id="old-test", title="T", created_by="t@x")
    monkeypatch.setattr(pn.dbmod, "list_teachers", lambda: ["t@x"])
    state = {"state": "RETURNED"}
    monkeypatch.setattr(pn, "get_submission", lambda *a: {
        "state": state["state"], "userId": "s1", "updateTime": "2026-10-17T10:00:00Z"})
    ev = {"collection": "courses.courseWork.studentSubmissions", "eventType": "MODIFIED",
          "resourceId": {"courseId": "c", "courseWorkId": "cw", "id": "sub1"}}
    body = {"message": {"data": base64.b64encode(json.dumps(ev).encode()).decode()}}
    pn.handle_notification(body)                      # caches ownership with old-test
    db.record_app_assignment("c", "cw", test_id="new-test", title="T")   # teacher switches test
    used = []
    monkeypatch.setattr(submissions, "template_for_test", lambda t: ("tpl", "ref"))
    monkeypatch.setattr(submissions, "grade_classroom_assignment",
                        lambda e, c, cw, t, tpl, **k: used.append(t) or {"results": []})
    state["state"] = "TURNED_IN"
    pn.handle_notification(body)
    assert used == ["new-test"]
    pn.forget_ownership()


def test_one_read_answers_every_event_in_a_course(db, monkeypatch):
    """Events for many different assignments the app doesn't own must not each
    wake the database: ownership is cached per course."""
    import base64, json
    import bubble_grader.push_notify as pn
    pn.forget_ownership()
    db.record_app_assignment("c", "ours", test_id="t", title="T", created_by="t@x")
    monkeypatch.setattr(pn.dbmod, "list_teachers", lambda: ["t@x"])
    monkeypatch.setattr(pn, "get_submission", lambda *a: {"state": "RETURNED", "userId": "s1"})
    calls = {"n": 0}
    real = pn.dbmod.list_app_assignments
    def counting(course_id):
        calls["n"] += 1
        return real(course_id)
    monkeypatch.setattr(pn.dbmod, "list_app_assignments", counting)
    def event(cw):
        ev = {"collection": "courses.courseWork.studentSubmissions", "eventType": "MODIFIED",
              "resourceId": {"courseId": "c", "courseWorkId": cw, "id": "sub"}}
        return {"message": {"data": base64.b64encode(json.dumps(ev).encode()).decode()}}
    pn.handle_notification(event("ours"))
    _forbid_db(monkeypatch)
    for i in range(8):
        assert pn.handle_notification(event(f"someone-elses-{i}"))["status"] == "ignored"
    assert calls["n"] == 1
    pn.forget_ownership()


def test_assignment_whose_test_is_picked_later_grades_immediately(db, monkeypatch):
    import base64, json
    import bubble_grader.push_notify as pn
    import bubble_grader.submissions as submissions
    pn.forget_ownership()
    db.record_app_assignment("c", "cw", test_id=None, title="T", created_by="t@x")
    monkeypatch.setattr(pn.dbmod, "list_teachers", lambda: ["t@x"])
    state = {"state": "CREATED"}
    monkeypatch.setattr(pn, "get_submission", lambda *a: {
        "state": state["state"], "userId": "s1", "updateTime": "2026-10-17T10:00:00Z"})
    ev = {"collection": "courses.courseWork.studentSubmissions", "eventType": "MODIFIED",
          "resourceId": {"courseId": "c", "courseWorkId": "cw", "id": "sub1"}}
    body = {"message": {"data": base64.b64encode(json.dumps(ev).encode()).decode()}}
    pn.handle_notification(body)                     # student opens it: no test yet
    db.record_app_assignment("c", "cw", test_id="t", title="T")   # teacher picks the test
    used = []
    monkeypatch.setattr(submissions, "template_for_test", lambda t: ("tpl", "ref"))
    monkeypatch.setattr(submissions, "grade_classroom_assignment",
                        lambda e, c, cw, t, tpl, **k: used.append(t) or {"results": []})
    state["state"] = "TURNED_IN"
    pn.handle_notification(body)
    assert used == ["t"]
    pn.forget_ownership()


def test_transient_error_lets_the_batch_retry(monkeypatch):
    import bubble_grader.auto_grade as ag
    monkeypatch.setattr(ag, "_attempted", set())
    monkeypatch.setattr(ag.dbmod, "list_teachers", lambda: ["t@x"])
    monkeypatch.setattr(ag, "list_courses", lambda email: [{"id": "c"}])
    monkeypatch.setattr(ag.dbmod, "list_app_assignments",
                        lambda c: [{"test_id": "t", "coursework_id": "cw", "created_at": None}])
    monkeypatch.setattr(ag.dbmod, "list_submissions", lambda **k: [])
    monkeypatch.setattr(ag, "list_submissions", lambda e, c, cw: [
        {"userId": "s1", "state": "TURNED_IN", "updateTime": "2026-10-17T10:00:00Z"}])
    monkeypatch.setattr(ag, "template_for_test", lambda t: ("tpl", "ref"))
    monkeypatch.setattr(ag, "_assignment_liveness", lambda *a: "unknown")
    outcome = {"fail": True}
    def grade(email, c, cw, t, tpl, only_students):
        if outcome["fail"]:
            raise RuntimeError("Drive hiccup")
        return {"results": [{"student_id": s, "status": "graded"} for s in only_students]}
    monkeypatch.setattr(ag, "grade_classroom_assignment", grade)
    assert ag.run_auto_grade_once()["students_graded"] == 0
    outcome["fail"] = False
    assert ag.run_auto_grade_once()["students_graded"] == 1


def _creds_fixture(monkeypatch, refresh_token_after="rt-1", fail=False):
    import datetime
    import bubble_grader.google_api as ga
    from google.oauth2.credentials import Credentials
    stored = {"token": "old", "refresh_token": "rt-1", "token_uri": "https://x", "client_id": "id",
              "client_secret": "s", "scopes": [], "id_token": None,
              "expiry": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)).isoformat()}
    writes, reads = [], []
    monkeypatch.setattr(ga, "load_credentials", lambda e: reads.append(e) or dict(stored))
    monkeypatch.setattr(ga, "store_credentials", lambda e, d: writes.append(d))
    def fake_refresh(self, request):
        if fail:
            from google.auth.exceptions import RefreshError
            raise RefreshError("invalid_grant")
        self.token = "new"
        self._refresh_token = refresh_token_after
        self.expiry = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(hours=1)
    monkeypatch.setattr(Credentials, "refresh", fake_refresh)
    ga.forget_credentials()
    return ga, writes, reads


def test_refreshing_an_access_token_does_not_write_to_the_database(monkeypatch):
    ga, writes, reads = _creds_fixture(monkeypatch)
    creds = ga._credentials_for("t@x")
    assert creds.token == "new"          # expired token refreshed up front
    ga._credentials_for("t@x")
    assert writes == [] and reads == ["t@x"]
    ga.forget_credentials()


def test_a_new_refresh_token_is_saved(monkeypatch):
    ga, writes, _ = _creds_fixture(monkeypatch, refresh_token_after="rt-2")
    ga._credentials_for("t@x")
    assert len(writes) == 1 and writes[0]["refresh_token"] == "rt-2"
    ga.forget_credentials()


def test_revoked_login_is_dropped_from_the_cache(monkeypatch):
    from google.auth.exceptions import RefreshError
    ga, _, reads = _creds_fixture(monkeypatch, fail=True)
    for _ in range(2):
        with pytest.raises(RefreshError):
            ga._credentials_for("t@x")
    assert reads == ["t@x", "t@x"]       # re-read each time, never served from cache
    ga.forget_credentials()


def test_api_docs_are_not_published():
    with TestClient(server.app) as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404, path
