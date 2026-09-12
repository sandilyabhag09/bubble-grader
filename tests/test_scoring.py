"""Golden tests for the scoring rules this project has re-learned the hard way.

The load-bearing invariant (see project memory / git history): the grading
answer_key holds ONLY scored questions. Field-test answers live separately and
are merged in for DISPLAY only. If field-test answers ever leak into grading,
raw scores exceed the scaler's range and clamp to a perfect 36 — the bug that
recurred three times in summer 2026.
"""

import json
from pathlib import Path

import pytest

from bubble_grader.config import DATA_DIR
from bubble_grader.scoring import (
    _scale_lookup,
    full_grade,
    grade_answers,
    merge_field_test,
    partial_summary,
)
from bubble_grader.feedback import _missed_clause

TEMPLATE_PATH = DATA_DIR / "sheets" / "act_sheet_new.template.json"
SCORED = {"Test 1": 40, "Test 2": 41, "Test 3": 27, "Test 4": 34}


@pytest.fixture(scope="module")
def template() -> dict:
    return json.loads(Path(TEMPLATE_PATH).read_text())


def _first_option_by_q(template):
    """q_in_test -> (section, first option letter, global q) from the template."""
    seen = {}
    for b in template["bubbles"]:
        key = (b["section"], b["q_in_test"])
        if key not in seen:
            seen[key] = (b["option"], b["q"])
    return seen


@pytest.fixture(scope="module")
def keys(template):
    """Synthetic scored-only key + field-test answers + a linear scaler."""
    first = _first_option_by_q(template)
    scored_key, field_test = {}, {}
    for (section, q_in_test), (option, _g) in first.items():
        target = scored_key if q_in_test <= SCORED[section] else field_test
        target.setdefault(section, {})[str(q_in_test)] = option
    scaler = {
        sec: {str(raw): max(1, round(1 + raw * 35 / SCORED[sec])) for raw in range(SCORED[sec] + 1)}
        for sec in SCORED
    }
    return scored_key, field_test, scaler


@pytest.fixture(scope="module")
def all_correct_answers(template):
    """Global-q answers where the student bubbled the 'correct' first option
    for EVERY row on the sheet — scored and field-test alike."""
    first = _first_option_by_q(template)
    return {g: opt for (_sec, _q), (opt, g) in first.items()}


def test_field_test_never_counts(template, keys, all_correct_answers):
    """THE regression test: perfect sheet → raw equals the SCORED count
    (40/41/27/34), never the full row count (50/45/36/40)."""
    scored_key, _ft, scaler = keys
    report = full_grade(all_correct_answers, template, scored_key, scaler)
    for sec, expected in SCORED.items():
        assert report["sections"][sec]["raw_score"] == expected
        assert report["sections"][sec]["n_questions"] == expected
        assert report["sections"][sec]["scaled_score"] == 36
    assert report["composite"] == 36


def test_display_merge_flags_but_never_counts(template, keys, all_correct_answers):
    """Display grading (merged key + not_scored) shows field-test details
    flagged scored=False while every counter stays scored-only."""
    scored_key, field_test, _ = keys
    display_key = merge_field_test(scored_key, field_test)
    sections = grade_answers(all_correct_answers, template, display_key, not_scored=field_test)
    for sec, expected in SCORED.items():
        info = sections[sec]
        assert info["raw_score"] == expected              # counters: scored only
        flags = [d["scored"] for d in info["details"]]
        assert flags.count(False) == len(info["details"]) - expected
        assert all(not d["scored"] for d in info["details"] if d["q_in_test"] > SCORED[sec])


def test_merge_field_test_does_not_mutate(keys):
    scored_key, field_test, _ = keys
    before = json.dumps(scored_key, sort_keys=True)
    merged = merge_field_test(scored_key, field_test)
    assert json.dumps(scored_key, sort_keys=True) == before
    assert len(merged["Test 1"]) == 50 and len(scored_key["Test 1"]) == 40


def test_scale_lookup_exact_interpolate_clamp():
    table = {"0": 1, "10": 10, "20": 20, "40": 36}
    assert _scale_lookup(10, table) == 10                 # exact
    assert _scale_lookup(15, table) == 15                 # linear interpolation
    assert _scale_lookup(45, table) == 36                 # clamp high — the "perfect 36" mechanism
    assert _scale_lookup(-3, table) == 1                  # clamp low


def test_wrong_answers_lower_the_score(template, keys, all_correct_answers):
    scored_key, _ft, scaler = keys
    answers = dict(all_correct_answers)
    # Flip three scored English answers to BLANK.
    flipped = 0
    for b in template["bubbles"]:
        if b["section"] == "Test 1" and b["q_in_test"] <= 3 and b["q"] in answers:
            answers[b["q"]] = "BLANK"
            flipped += 1
    report = full_grade(answers, template, scored_key, scaler)
    assert report["sections"]["Test 1"]["raw_score"] == SCORED["Test 1"] - 3
    assert report["sections"]["Test 1"]["scaled_score"] < 36
    assert report["sections"]["Test 1"]["n_blank"] == 3


def test_partial_summary_ignores_unscored():
    report = {"sections": {"Test 1": {"details": [
        {"q_in_test": 1, "status": "correct", "scored": True},
        {"q_in_test": 2, "status": "correct", "scored": False},   # field-test
        {"q_in_test": 3, "status": "incorrect", "scored": True},
    ]}}}
    out = partial_summary(report, {"section": "Test 1", "q_start": 1, "q_end": 3})
    assert out["raw"] == 1 and out["total"] == 3


def test_missed_clause_labels_field_test():
    details = [
        {"q_in_test": 2, "status": "incorrect", "scored": True},
        {"q_in_test": 5, "status": "blank", "scored": True},
        {"q_in_test": 42, "status": "incorrect", "scored": False},
        {"q_in_test": 1, "status": "correct", "scored": True},
    ]
    clause = _missed_clause(details)
    assert clause == "Missed questions: 2, 5 (plus 42 not scored)"
    assert _missed_clause([{"q_in_test": 1, "status": "correct", "scored": True}]) == "Missed questions: none"
