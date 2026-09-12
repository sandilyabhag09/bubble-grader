"""Review queue: borderline reader calls get flagged with row crops."""

import base64

import numpy as np

from bubble_grader.submissions import REVIEW_MAX_ITEMS, _review_suspects


def _template(n_questions=6, options=("A", "B", "C", "D")):
    bubbles = []
    for q in range(1, n_questions + 1):
        for i, opt in enumerate(options):
            bubbles.append({
                "q": q, "q_in_test": q, "section": "Test 1", "option": opt,
                "center_mm": [20 + i * 8, 10 * q], "radius_mm": 1.8,
            })
    return {"bubbles": bubbles}


def _read_result(per_q: dict, warped=True):
    """per_q: q -> (answer, top_fill, second_fill)."""
    fills, answers = {}, {}
    for q, (ans, top, second) in per_q.items():
        answers[q] = ans
        fills[q] = [
            {"option": "A", "fill": top},
            {"option": "B", "fill": second},
            {"option": "C", "fill": 0.0},
            {"option": "D", "fill": 0.0},
        ]
    out = {"answers": answers, "fills": fills}
    if warped:
        out["warped"] = np.full((700, 400), 255, dtype=np.uint8)
        out["warped_dpi"] = 200
    return out


def test_clean_reads_produce_no_suspects():
    rr = _read_result({1: ("A", 0.65, 0.02), 2: ("BLANK", 0.03, 0.01)})
    assert _review_suspects(rr, _template()) == []


def test_faint_blank_multi_and_borderline_are_flagged():
    rr = _read_result({
        1: ("BLANK", 0.15, 0.05),   # faint mark read as blank
        2: ("MULTI", 0.45, 0.40),   # two marks
        3: ("A", 0.33, 0.05),       # committed but under SOLID_FILL
        4: ("A", 0.60, 0.25),       # committed but dark runner-up
        5: ("A", 0.70, 0.03),       # clean — must NOT appear
    })
    out = _review_suspects(rr, _template())
    flagged_qs = {s["q"] for s in out}
    assert flagged_qs == {1, 2, 3, 4}
    # blank/multi outrank borderline singles
    assert [s["q"] for s in out[:2]] == [1, 2]
    for s in out:
        assert s["reason"] and s["section"] == "Test 1"


def test_crops_are_valid_jpegs():
    rr = _read_result({2: ("MULTI", 0.5, 0.45)})
    (sus,) = _review_suspects(rr, _template())
    raw = base64.b64decode(sus["crop_b64"])
    assert raw[:2] == b"\xff\xd8"  # JPEG magic
    assert len(raw) < 60_000


def test_suspects_are_capped():
    rr = _read_result({q: ("MULTI", 0.5, 0.45) for q in range(1, 30)})
    out = _review_suspects(rr, _template(n_questions=30))
    assert len(out) == REVIEW_MAX_ITEMS


def test_missing_warped_image_still_flags_without_crops():
    rr = _read_result({1: ("MULTI", 0.5, 0.45)}, warped=False)
    (sus,) = _review_suspects(rr, _template())
    assert "crop_b64" not in sus and sus["reason"]
