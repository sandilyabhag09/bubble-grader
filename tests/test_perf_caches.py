"""In-process caches that turned page loads from seconds into sub-second."""

import bubble_grader.classroom as classroom
import bubble_grader.google_api as google_api


class _Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def test_classroom_lists_are_cached_and_invalidated(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(classroom.time, "monotonic", clock)
    classroom.invalidate()
    calls = {"n": 0}
    monkeypatch.setattr(classroom, "service_for", lambda *a, **k: object())
    def fake_paginate(request_fn, key):
        calls["n"] += 1
        return [{"id": "c1"}]
    monkeypatch.setattr(classroom, "_paginate", fake_paginate)

    assert classroom.list_courses("t@x") == [{"id": "c1"}]
    assert classroom.list_courses("t@x") == [{"id": "c1"}]
    assert calls["n"] == 1                                   # second call served from cache

    classroom.list_coursework("t@x", "course1")
    classroom.list_coursework("t@x", "course1")
    assert calls["n"] == 2

    classroom.invalidate("coursework", "t@x", "course1")     # what create/delete do
    classroom.list_coursework("t@x", "course1")
    assert calls["n"] == 3
    assert classroom.list_courses("t@x") and calls["n"] == 3  # unrelated entry untouched

    clock.t += classroom.TTL_COURSES + 1                     # expiry
    classroom.list_courses("t@x")
    assert calls["n"] == 4


def test_credentials_are_cached_per_teacher(monkeypatch):
    google_api.forget_credentials()
    loads = {"n": 0}

    class FakeCreds:
        token = "tok"
    def fake_load(email):
        loads["n"] += 1
        return {"token": "tok"}
    monkeypatch.setattr(google_api, "load_credentials", fake_load)
    monkeypatch.setattr(google_api, "credentials_from_dict", lambda raw: FakeCreds())
    monkeypatch.setattr(google_api, "ensure_fresh", lambda c: c)
    monkeypatch.setattr(google_api, "build", lambda *a, **k: "svc")

    assert google_api.service_for("t@x", "classroom", "v1") == "svc"
    assert google_api.service_for("t@x", "drive", "v3") == "svc"
    assert loads["n"] == 1                                   # one DB round trip, not two

    google_api.forget_credentials("t@x")                      # what a re-sign-in does
    google_api.service_for("t@x", "classroom", "v1")
    assert loads["n"] == 2
