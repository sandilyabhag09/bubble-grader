"""Re-grade dedup rules (student detail view): override-carrying grades are
sticky, otherwise closest-to-due wins, re-takes >1 week apart stay separate."""

from datetime import datetime

from bubble_grader.server import _dedupe_regrades


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


DUE = {"CW1": dt("2026-01-01 15:00:00+00:00")}


def sub(id, created, overrides=False, cw="CW1", test="T"):
    return {
        "id": id, "test_id": test, "coursework_id": cw,
        "created_at": dt(created),
        "score": {"overrides": [{"q": 1}]} if overrides else {},
    }


def kept_ids(subs, due=DUE):
    return [s["id"] for s in _dedupe_regrades(subs, due)]


def test_override_grade_is_sticky():
    subs = [  # newest-first, as list_submissions returns
        sub("C", "2026-01-03 09:25:00+00:00"),                    # later auto re-grade
        sub("B", "2026-01-01 15:10:00+00:00", overrides=True),    # manual override
        sub("A", "2026-01-01 14:55:00+00:00"),                    # closest to due
    ]
    assert kept_ids(subs) == ["B"]


def test_closest_to_due_wins_without_overrides():
    subs = [
        sub("C", "2026-01-03 09:25:00+00:00"),
        sub("B", "2026-01-01 15:10:00+00:00"),
        sub("A", "2026-01-01 14:55:00+00:00"),   # 5 min from due — closest
    ]
    assert kept_ids(subs) == ["A"]


def test_no_due_date_falls_back_to_newest():
    subs = [
        sub("C", "2026-01-03 09:25:00+00:00"),
        sub("A", "2026-01-01 14:55:00+00:00"),
    ]
    assert kept_ids(subs, due={}) == ["C"]


def test_retakes_a_week_apart_both_kept():
    subs = [
        sub("Y", "2026-02-01 10:00:00+00:00"),
        sub("A", "2026-01-01 14:55:00+00:00"),
    ]
    assert sorted(kept_ids(subs)) == ["A", "Y"]


def test_different_assignments_never_merge():
    subs = [
        sub("B", "2026-01-01 15:10:00+00:00", cw="CW2"),
        sub("A", "2026-01-01 14:55:00+00:00", cw="CW1"),
    ]
    assert sorted(kept_ids(subs)) == ["A", "B"]
