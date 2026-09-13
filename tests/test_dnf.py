"""Did-not-finish: an all-blank section shows DNF in gradebook views while the
composite still uses the section's floor score. Emails/override rows untouched."""

from bubble_grader.scoring import section_is_dnf
from bubble_grader.server import _section_scaled_summary


def test_dnf_rule():
    assert section_is_dnf({"n_questions": 34, "n_blank": 34})
    assert not section_is_dnf({"n_questions": 34, "n_blank": 33, "n_incorrect": 1})
    assert not section_is_dnf({"n_questions": 0, "n_blank": 0})
    assert not section_is_dnf(None)


def test_summary_shows_dnf_but_composite_math_unchanged():
    score = {
        "scaled_per_section": {"Test 1": 33, "Test 2": 32, "Test 3": 26, "Test 4": 1},
        "sections": {
            "Test 1": {"n_questions": 40, "n_blank": 0},
            "Test 2": {"n_questions": 41, "n_blank": 0},
            "Test 3": {"n_questions": 27, "n_blank": 0},
            "Test 4": {"n_questions": 34, "n_blank": 34},
        },
        "composite": 23,
    }
    assert _section_scaled_summary(score) == "33 / 32 / 26 / DNF"
    assert score["scaled_per_section"]["Test 4"] == 1      # still scales to 1
    assert score["composite"] == 23                        # composite unchanged


def test_grade_answers_flags_dnf_section():
    from bubble_grader.scoring import grade_answers
    template = {"bubbles": [
        {"q": 1, "section": "Test 4", "q_in_test": 1, "option": "A"},
        {"q": 2, "section": "Test 4", "q_in_test": 2, "option": "F"},
        {"q": 3, "section": "Test 1", "q_in_test": 1, "option": "A"},
    ]}
    key = {"Test 4": {"1": "A", "2": "F"}, "Test 1": {"1": "A"}}
    out = grade_answers({1: "BLANK", 2: "BLANK", 3: "A"}, template, key)
    assert out["Test 4"]["dnf"] is True and out["Test 1"]["dnf"] is False
