"""End-to-end optical-mark-reading golden test, fully synthetic.

Renders a known answer pattern onto the committed reference sheet, then runs
the real feature-match reader on it. Any drift in warping, thresholds, or
template geometry shows up here — with zero student data involved.
"""

import json
from pathlib import Path

import pytest

from bubble_grader.config import DATA_DIR

SHEETS = DATA_DIR / "sheets"


@pytest.mark.parametrize("stem", ["act_sheet_new", "act_sheet"])
def test_simulated_sheet_reads_back_exactly(tmp_path, stem):
    import cv2
    from bubble_grader.sheet import simulate_fill
    from bubble_grader.omr import read_sheet_fm

    template_path = SHEETS / f"{stem}.template.json"
    reference_path = SHEETS / f"{stem}.reference.png"
    template = json.loads(template_path.read_text())

    # Deterministic non-trivial pattern: rotate through each row's options.
    options_by_q: dict[int, list[str]] = {}
    for b in template["bubbles"]:
        options_by_q.setdefault(b["q"], []).append(b["option"])
    expected = {q: opts[q % len(opts)] for q, opts in options_by_q.items()}

    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    filled = simulate_fill(template, expected,
                           dpi=int(template.get("source_dpi", 300)),
                           base_image=reference)
    scan = tmp_path / "filled.png"
    filled.save(scan)

    result = read_sheet_fm(scan, template_path, reference_path)
    got = {int(q): a for q, a in result["answers"].items()}

    mismatches = {q: (expected[q], got.get(q)) for q in expected if got.get(q) != expected[q]}
    assert not mismatches, f"{len(mismatches)} misread: {dict(list(mismatches.items())[:8])}"
