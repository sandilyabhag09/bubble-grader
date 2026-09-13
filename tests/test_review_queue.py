"""Review queue: borderline reader calls get flagged with row crops."""

import base64

import numpy as np

from bubble_grader.submissions import REVIEW_MAX_CROPS, _review_suspects


def _template(n_questions=6, options=("A", "B", "C", "D")):
    bubbles = []
    for q in range(1, n_questions + 1):
        for i, opt in enumerate(options):
            bubbles.append({
                "q": q, "q_in_test": q, "section": "Test 1", "option": opt,
                "center_mm": [20 + i * 8, 10 * q], "radius_mm": 1.8,
            })
    return {"bubbles": bubbles}


def _read_result(per_q: dict, warped=True, height_px=700):
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
        out["warped"] = np.full((height_px, 400), 255, dtype=np.uint8)
        out["warped_dpi"] = 200
    return out


def test_clean_reads_produce_no_suspects():
    rr = _read_result({1: ("A", 0.65, 0.02), 2: ("B", 0.71, 0.00)})
    assert _review_suspects(rr, _template()) == []


def test_true_blanks_are_included_with_crops():
    """A genuinely blank row costs a point, so it still gets a picture —
    the teacher confirms it's really blank instead of trusting the reader."""
    rr = _read_result({1: ("BLANK", 0.03, 0.01), 2: ("A", 0.7, 0.0)})
    (sus,) = _review_suspects(rr, _template())
    assert sus["q"] == 1 and "read as BLANK" in sus["reason"] and sus["crop_b64"]


def test_only_blank_and_multi_are_flagged():
    rr = _read_result({
        1: ("BLANK", 0.15, 0.05),   # blank (even a faint one is just "blank")
        2: ("MULTI", 0.45, 0.40),   # two marks
        3: ("A", 0.33, 0.05),       # committed, weak — trusted, NOT shown
        4: ("A", 0.60, 0.25),       # committed, dark runner-up — trusted, NOT shown
        5: ("A", 0.70, 0.03),       # clean
    })
    out = _review_suspects(rr, _template())
    assert [s["q"] for s in out] == [2, 1]          # multi first, then blank
    assert "MULTI" in out[0]["reason"] and "BLANK" in out[1]["reason"]
    assert not any(ch.isdigit() for ch in out[1]["reason"].split("BLANK")[1])  # no fill numbers


def test_crops_are_valid_jpegs():
    rr = _read_result({2: ("MULTI", 0.5, 0.45)})
    (sus,) = _review_suspects(rr, _template())
    raw = base64.b64decode(sus["crop_b64"])
    assert raw[:2] == b"\xff\xd8"  # JPEG magic
    assert len(raw) < 60_000


def test_every_blank_row_is_listed_only_crops_are_capped():
    """34 blanks must show 34 rows — a teacher noticed 20. Images are the
    heavy part, so only those are capped."""
    n = REVIEW_MAX_CROPS + 15
    # Rows sit at 10 mm * q; make the frame tall enough to hold every row.
    rr = _read_result({q: ("BLANK", 0.02, 0.01) for q in range(1, n + 1)},
                      height_px=int((10 * n + 30) * 200 / 25.4))
    out = _review_suspects(rr, _template(n_questions=n))
    assert len(out) == n
    assert sum(1 for s in out if "crop_b64" in s) == REVIEW_MAX_CROPS
    assert all("crop_b64" not in s for s in out[REVIEW_MAX_CROPS:])


def test_missing_warped_image_still_flags_without_crops():
    rr = _read_result({1: ("MULTI", 0.5, 0.45)}, warped=False)
    (sus,) = _review_suspects(rr, _template())
    assert "crop_b64" not in sus and sus["reason"]
