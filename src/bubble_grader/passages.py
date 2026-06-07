"""Standard ACT passage definitions for partial-scope assignments.

Maps (format, section) → list of (start_q, end_q, label) tuples. Used by
the new-assignment form's Passage dropdown so the teacher can pick
"Reading Passage 2" without typing question numbers.

Two formats are supported:

* ``legacy`` — the long-running 75/60/40/40 ACT used 2016-mid-2025.
  Passage breakdowns here are standardized across all our 25 legacy tests.

* ``new`` — the shortened 50/45/36/40 ACT introduced in 2025.
  Passage counts and per-passage question counts vary slightly more
  between tests; the defaults here match the published "Practice Test 3"
  (Double the Manta Rays) and most other new-format releases.

Math has no passages on the legacy form (just individual questions), so
``Test 2`` is intentionally absent from both maps — the new-assignment
form falls back to free-form question range entry for Math.

Science passage counts can vary across tests (5 vs 6 vs 7 passages with
different question counts each). The defaults below cover the typical
case; if a specific test has different breakdowns the teacher can always
use "Custom range" instead.
"""

from __future__ import annotations


# Each entry: (q_start, q_end, label)
PASSAGES_BY_FORMAT_AND_SECTION: dict[str, dict[str, list[tuple[int, int, str]]]] = {
    "legacy": {
        # English: 5 passages × 15 questions
        "Test 1": [
            (1, 15, "Passage I"),
            (16, 30, "Passage II"),
            (31, 45, "Passage III"),
            (46, 60, "Passage IV"),
            (61, 75, "Passage V"),
        ],
        # Reading: 4 passages × 10 questions
        "Test 3": [
            (1, 10, "Passage I — Prose Fiction / Literary Narrative"),
            (11, 20, "Passage II — Social Science"),
            (21, 30, "Passage III — Humanities"),
            (31, 40, "Passage IV — Natural Science"),
        ],
        # Science: typically 6 passages of 5-7 questions each.
        # These ranges are the most common breakdown; if a particular
        # test differs the teacher can use Custom range.
        "Test 4": [
            (1, 6, "Passage I"),
            (7, 12, "Passage II"),
            (13, 19, "Passage III"),
            (20, 26, "Passage IV"),
            (27, 33, "Passage V"),
            (34, 40, "Passage VI"),
        ],
    },
    "new": {
        # English (new format): 5 passages × 10 questions
        "Test 1": [
            (1, 10, "Passage I"),
            (11, 20, "Passage II"),
            (21, 30, "Passage III"),
            (31, 40, "Passage IV"),
            (41, 50, "Passage V"),
        ],
        # Reading (new format): 3 passages × 12 questions
        "Test 3": [
            (1, 12, "Passage I"),
            (13, 24, "Passage II"),
            (25, 36, "Passage III"),
        ],
        # Science (new format): similar to legacy; ranges may shift slightly
        "Test 4": [
            (1, 6, "Passage I"),
            (7, 12, "Passage II"),
            (13, 19, "Passage III"),
            (20, 26, "Passage IV"),
            (27, 33, "Passage V"),
            (34, 40, "Passage VI"),
        ],
    },
}


def passages_for(fmt: str, section: str) -> list[dict]:
    """Return passage records for the given test format + section.

    Format defaults to legacy when unrecognized — covers any test JSON
    that doesn't carry a format flag yet. Empty list means "no presets;
    use Custom range."
    """
    table = PASSAGES_BY_FORMAT_AND_SECTION.get(fmt) or PASSAGES_BY_FORMAT_AND_SECTION["legacy"]
    return [
        {"q_start": s, "q_end": e, "label": label}
        for (s, e, label) in (table.get(section) or [])
    ]
