"""Answer-key import validator — the deterministic gate behind the Claude
extraction. Every rule here was learned from a real form that broke grading."""

import pytest

from bubble_grader.key_import import sanity_check, validate_and_normalize


def _scaler(n):
    return {str(r): max(1, round(1 + 35 * r / n)) for r in range(n + 1)}


def test_full_form_booklet_splits_scored_and_field_test():
    """Alex-Atala style: all 50 positions printed, 41-50 marked Not Scored."""
    answers = {str(q): ("ABCD"[q % 4] if q % 2 else "FGHJ"[q % 4]) for q in range(1, 51)}
    raw = {"test_form": "67C", "sections": {
        "Test 1": {"answers": answers, "not_scored": list(range(41, 51))}},
        "scaler": {"Test 1": _scaler(40)}}
    out = validate_and_normalize(raw)
    assert len(out["answers"]["Test 1"]) == 40
    assert len(out["field_test_answers"]["Test 1"]) == 10
    assert "41" in out["field_test_answers"]["Test 1"]
    assert "41" not in out["answers"]["Test 1"]      # the load-bearing invariant
    assert out["conversions"] == []


def test_renumbered_booklet_converts_flipped_letters():
    """Seeing-Streams style: renumbered contiguously, test-day letters kept, so
    some rows carry the wrong letter set — converted by position."""
    answers = {"1": "B", "2": "F", "3": "H", "4": "B", "5": "A", "6": "G"}
    #                      Q3 odd row gets H (even-set) -> C; Q4 even row gets B -> G
    raw = {"sections": {"Test 3": {"answers": answers, "not_scored": []}},
           "scaler": {"Test 3": _scaler(6)}}
    out = validate_and_normalize(raw)
    assert out["answers"]["Test 3"] == {"1": "B", "2": "F", "3": "C", "4": "G", "5": "A", "6": "G"}
    assert len(out["conversions"]) == 2
    assert out["field_test_answers"] is None


def test_gaps_without_not_scored_markers_are_rejected():
    raw = {"sections": {"Test 1": {"answers": {"1": "A", "3": "C"}, "not_scored": []}},
           "scaler": {"Test 1": _scaler(2)}}
    with pytest.raises(ValueError, match="gaps"):
        validate_and_normalize(raw)


def test_scaler_must_cover_every_raw_exactly_once():
    answers = {"1": "A", "2": "F", "3": "C"}
    good = {"sections": {"Test 1": {"answers": answers, "not_scored": []}},
            "scaler": {"Test 1": {"0": 1, "1": 12, "2": 24, "3": 36}}}
    validate_and_normalize(good)  # baseline passes

    missing = {"sections": {"Test 1": {"answers": answers, "not_scored": []}},
               "scaler": {"Test 1": {"0": 1, "1": 12, "3": 36}}}
    with pytest.raises(ValueError, match="missing raw scores"):
        validate_and_normalize(missing)

    # Scaler reaching past the scored count = the perfect-36 clamp setup.
    over = {"sections": {"Test 1": {"answers": answers, "not_scored": []}},
            "scaler": {"Test 1": {str(r): r + 1 for r in range(0, 6)}}}
    with pytest.raises(ValueError, match="only 3 questions are scored"):
        validate_and_normalize(over)


def test_invalid_letters_rejected():
    raw = {"sections": {"Test 1": {"answers": {"1": "Z"}, "not_scored": []}},
           "scaler": {"Test 1": _scaler(1)}}
    with pytest.raises(ValueError, match="invalid answer letter"):
        validate_and_normalize(raw)


def test_sanity_check_catches_tampering():
    answers = {str(q): ("ABCD"[q % 4] if q % 2 else "FGHJ"[q % 4]) for q in range(1, 51)}
    raw = {"sections": {"Test 1": {"answers": answers, "not_scored": list(range(41, 51))}},
           "scaler": {"Test 1": _scaler(40)}}
    out = validate_and_normalize(raw)
    sanity_check(out)  # clean payload passes

    tampered = {**out, "answers": {"Test 1": {**out["answers"]["Test 1"], "1": "F"}}}
    with pytest.raises(ValueError, match="wrong letter set"):
        sanity_check(tampered)

    leaked = {**out, "field_test_answers": {"Test 1": {"40": "F", **out["field_test_answers"]["Test 1"]}}}
    with pytest.raises(ValueError, match="also in the scored key"):
        sanity_check(leaked)
