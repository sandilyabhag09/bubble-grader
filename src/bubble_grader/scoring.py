"""Score a {q: answer} dict against an ACT answer key and a per-section scaler.

File formats
------------
Answer key JSON:
    {
      "test_form": "67C",
      "answers": {
        "Test 1": {"1": "B", "2": "G", ...},
        "Test 2": {"1": "D", "2": "K", ...},
        "Test 3": {"1": "A", ...},
        "Test 4": {"1": "C", ...}
      }
    }

Scaler JSON (per-section raw → scaled lookup table; missing raws are linearly
interpolated):
    {
      "Test 1": {"0": 1, "1": 1, "2": 2, ..., "75": 36},
      "Test 2": {"0": 1, ..., "60": 36},
      "Test 3": {"0": 1, ..., "40": 36},
      "Test 4": {"0": 1, ..., "40": 36}
    }
"""

import json
from pathlib import Path

# Sections in the order they appear on the sheet — used when iterating composites.
DEFAULT_SECTION_ORDER = ["Test 1", "Test 2", "Test 3", "Test 4"]


def load_answer_key(path: Path | str) -> dict:
    data = json.loads(Path(path).read_text())
    if "answers" not in data:
        raise ValueError(
            "Answer key missing top-level 'answers' field — see scoring.py docstring."
        )
    return {
        "test_form": data.get("test_form", "?"),
        "answers": {
            section: {int(q): opt.upper() for q, opt in qs.items()}
            for section, qs in data["answers"].items()
        },
    }


def load_scaler(path: Path | str) -> dict[str, dict[int, int]]:
    data = json.loads(Path(path).read_text())
    return {
        section: {int(raw): int(scaled) for raw, scaled in scales.items()}
        for section, scales in data.items()
    }


def _scale_lookup(raw: int, table: dict) -> int | None:
    """Look up `raw` in `table`. Linear-interpolate or clamp at extremes.

    Tolerates either int- or str-keyed tables (DB persists JSON → str keys).
    """
    if not table:
        return None
    norm = {int(k): int(v) for k, v in table.items()}
    if raw in norm:
        return norm[raw]
    keys = sorted(norm)
    if raw <= keys[0]:
        return norm[keys[0]]
    if raw >= keys[-1]:
        return norm[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= raw <= hi:
            ratio = (raw - lo) / (hi - lo)
            return int(round(norm[lo] + ratio * (norm[hi] - norm[lo])))
    return None


def _composite(scaled_per_section: dict[str, int]) -> int | None:
    """ACT composite = mean of 4 section scaled scores, half-up rounding."""
    values = [v for v in scaled_per_section.values() if v is not None]
    if len(values) != 4:
        return None
    # Half-up: round(x + 0.5 - epsilon) ≡ math.floor(x + 0.5). Avoids banker's rounding.
    return int((sum(values) / 4) + 0.5)


def _inner_key(answer_key: dict) -> dict[str, dict]:
    """Accept either the wrapped form ({'test_form', 'answers': {...}}) or just
    the inner {section: {q: option}} dict, and return the inner dict."""
    inner = answer_key.get("answers") if isinstance(answer_key, dict) else None
    if isinstance(inner, dict) and all(isinstance(v, dict) for v in inner.values()):
        return inner
    return answer_key


def merge_field_test(answer_key: dict | None, field_test_answers: dict | None) -> dict:
    """Overlay field-test answers onto the scored answer key.

    Used ONLY for display (email + overlay + detail table): grading always runs
    on the scored-only ``answer_key`` so field-test questions can never move a
    score, but for display we want their answers too so students see every wrong
    question. Returns a new dict; inputs are not mutated.
    """
    out: dict = {sec: dict(qs) for sec, qs in (answer_key or {}).items()}
    for sec, qs in (field_test_answers or {}).items():
        out.setdefault(sec, {}).update(qs)
    return out


def _not_scored_set(not_scored: dict | None) -> set[tuple[str, int]]:
    """Normalize ``{section: [q_in_test, ...]}`` to a set of (section, int)."""
    out: set[tuple[str, int]] = set()
    if not not_scored:
        return out
    for section, qs in not_scored.items():
        for q in qs or []:
            try:
                out.add((section, int(q)))
            except (TypeError, ValueError):
                continue
    return out


def grade_answers(
    answers: dict[int, str],
    template: dict,
    answer_key: dict,
    not_scored: dict | None = None,
) -> dict[str, dict]:
    """Per-section breakdown: counts, raw score, per-question detail.

    `answers` is keyed by *global* question number (1..215).
    `template` provides q → (section, q_in_test) mapping via its `bubbles`.
    `answer_key` is either the wrapped form from `load_answer_key` or the inner
    {section: {q_in_test: option}} dict (e.g. as stored in the DB).
    `not_scored` is ``{section: [q_in_test, ...]}`` of field-test questions ACT
    doesn't count. Their answers are still in the key (so we can grade them for
    display), but they're excluded from the counts and raw score — every detail
    carries a ``scored`` flag so the email/overlay can show them as wrong while
    the scaled score reflects only the scored questions.
    """
    key = _inner_key(answer_key)
    ns = _not_scored_set(not_scored)
    q_meta: dict[int, tuple[str, int]] = {}
    for b in template.get("bubbles", []):
        if "section" in b and "q_in_test" in b:
            q_meta[b["q"]] = (b["section"], b["q_in_test"])

    sections: dict[str, dict] = {}
    for q, given in answers.items():
        if q not in q_meta:
            continue
        section, q_in_test = q_meta[q]
        # Lookup tolerates both string-keyed (DB) and int-keyed dicts.
        section_key = key.get(section, {}) if isinstance(key, dict) else {}
        correct = section_key.get(q_in_test) or section_key.get(str(q_in_test))
        if correct is None:
            # Question isn't in this test's answer key. Happens for
            # new-format tests scanned on the legacy 75/60/40/40 bubble
            # sheet — rows past the test's actual question count have no
            # correct answer to compare against, so skip entirely rather
            # than miscount blanks or mark stray marks as wrong.
            continue
        is_scored = (section, q_in_test) not in ns
        sec = sections.setdefault(
            section,
            {"n_correct": 0, "n_incorrect": 0, "n_blank": 0, "n_multi": 0, "details": []},
        )

        given_norm = given.upper() if isinstance(given, str) else given
        if given_norm == "BLANK":
            status = "blank"
        elif given_norm == "MULTI":
            status = "multi"
        elif correct is not None and given_norm == correct:
            status = "correct"
        else:
            status = "incorrect"

        # Field-test questions are graded (so we know they're wrong/right) but
        # never counted — only scored questions move the counters/raw score.
        if is_scored:
            sec[{"blank": "n_blank", "multi": "n_multi",
                 "correct": "n_correct", "incorrect": "n_incorrect"}[status]] += 1

        sec["details"].append(
            {
                "q": q,
                "q_in_test": q_in_test,
                "given": given_norm,
                "correct": correct,
                "status": status,
                "scored": is_scored,
            }
        )

    for sec in sections.values():
        sec["details"].sort(key=lambda d: d["q_in_test"])
        sec["raw_score"] = sec["n_correct"]
        sec["n_questions"] = (
            sec["n_correct"] + sec["n_incorrect"] + sec["n_blank"] + sec["n_multi"]
        )
    return sections


def partial_summary(report: dict, scope: dict) -> dict:
    """Compute raw / total / percent for the in-scope question range only.

    ``scope`` is the dict stored on app_assignments: it must have
    ``section`` (one of "Test 1".."Test 4"), ``q_start``, ``q_end``.

    Returns ``{"raw": N, "total": M, "percent": round_to_1_decimal}``
    where ``raw`` counts in-scope questions marked "correct" and
    ``total = q_end - q_start + 1``. Questions in the range that don't
    appear in the answer key (e.g. new-format field-test items) are
    treated as un-gradeable rather than incorrect — but for the
    teacher's reporting we still divide by the full range size, so
    the percent reflects the passage as it was assigned. Toggle this
    interpretation later if the teacher prefers raw/scorable_total.
    """
    section = scope.get("section")
    qs = int(scope.get("q_start", 1))
    qe = int(scope.get("q_end", 0))
    sec_info = (report.get("sections") or {}).get(section) or {}
    details = sec_info.get("details") or []
    in_range = [d for d in details if qs <= d.get("q_in_test", 0) <= qe]
    raw = sum(
        1 for d in in_range
        if d.get("status") == "correct" and d.get("scored", True)
    )
    total = max(0, qe - qs + 1)
    pct = round((raw / total) * 100, 1) if total else 0.0
    return {"raw": raw, "total": total, "percent": pct}


def full_grade(
    answers: dict[int, str],
    template: dict,
    answer_key: dict,
    scaler: dict[str, dict[int, int]] | None = None,
    not_scored: dict | None = None,
) -> dict:
    """End-to-end report: per-section raw + scaled + composite."""
    sections = grade_answers(answers, template, answer_key, not_scored=not_scored)

    scaled_per_section: dict[str, int] = {}
    if scaler is not None:
        for section, info in sections.items():
            scaled = _scale_lookup(info["raw_score"], scaler.get(section, {}))
            info["scaled_score"] = scaled
            if scaled is not None:
                scaled_per_section[section] = scaled

    return {
        "test_form": answer_key.get("test_form", "?"),
        "sections": sections,
        "scaled_per_section": scaled_per_section,
        "composite": _composite(scaled_per_section) if scaler else None,
    }
