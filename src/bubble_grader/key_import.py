"""Answer-key PDF import: Claude extracts, deterministic code verifies.

The model reads the PDF text and returns structured data; nothing it says is
trusted until `validate_and_normalize` has enforced every rule this project
learned the hard way across real ACT forms:

* Field-test ("Not Scored") questions NEVER enter the grading key — they're
  split into `field_test_answers` (the scored-only-key invariant that keeps
  scaled scores from clamping to a perfect 36).
* "My Answer Key" booklets renumber questions after removing non-scored ones
  while keeping test-day letters, so some rows carry the wrong letter set for
  their position on the bubble sheet. Those are converted positionally
  (F→A, G→B, H→C, J→D, K→E and vice versa), exactly how students bubble them.
* The scaler must cover every raw score 0..N exactly once, and its max raw
  must equal the scored question count — the clamp guard.
"""

from __future__ import annotations

import io
import json
import re

ODD_SET = "ABCDE"   # odd-numbered rows on the sheet
EVEN_SET = "FGHJK"  # even-numbered rows

SECTION_LABELS = {
    "Test 1": "English", "Test 2": "Math", "Test 3": "Reading", "Test 4": "Science",
}

_EXTRACTION_PROMPT = """You are extracting an official ACT answer key booklet for an automated grading system. Read the booklet text and return ONLY a JSON object (no prose, no code fences) with exactly this shape:

{
  "test_form": "<form code like J08 or 67C, or null>",
  "name": "<a short human test name if one is evident, else null>",
  "sections": {
    "Test 1": {"answers": {"1": "B", "2": "F", ...}, "not_scored": [41, 42, ...]},
    "Test 2": {...}, "Test 3": {...}, "Test 4": {...}
  },
  "scaler": {
    "Test 1": {"40": 36, "39": 35, ...}, "Test 2": {...}, "Test 3": {...}, "Test 4": {...}
  }
}

Rules:
- "Test 1" is English, "Test 2" Mathematics, "Test 3" Reading, "Test 4" Science.
- answers: EVERY question number printed in the scoring key, with its answer letter EXACTLY as printed. Do not renumber, do not convert letters, do not skip questions marked "Not Scored" — include them in answers AND list their numbers in not_scored.
- not_scored: ONLY question numbers the booklet explicitly marks as not scored (e.g. "Not Scored"). If the booklet says non-scored questions were REMOVED (a renumbered "My Answer Key" booklet), not_scored must be an empty list.
- scaler: expand the conversion table to one entry PER RAW SCORE. A row like "35: 37–39" means raws 37, 38 and 39 all map to scale 35. Rows marked "—" contribute nothing. Every raw from 0 to the section maximum must appear exactly once.
- If a section is missing from the booklet, omit it from both objects.

Return the JSON object and nothing else."""


def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 16) -> str:
    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages[:max_pages]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 — a bad page shouldn't kill the import
            pages.append("")
    text = "\n\n".join(pages).strip()
    if len(text) < 400:
        raise ValueError(
            "Couldn't read text from this PDF (it may be a pure image scan). "
            "Try the original digital answer-key PDF."
        )
    return text


def claude_extract(booklet_text: str, api_key: str) -> dict:
    """One structured-extraction call. Output is UNTRUSTED until validated."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=16000,
            system=_EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": booklet_text}],
        )
    except anthropic.AuthenticationError:
        raise ValueError("ANTHROPIC_API_KEY was rejected — check the key in the server settings.")
    except anthropic.RateLimitError:
        raise ValueError("Anthropic rate limit hit — wait a minute and retry.")
    except anthropic.APIConnectionError:
        raise ValueError("Couldn't reach the Anthropic API — network issue; retry.")

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    # Tolerate accidental fencing; the prompt forbids it but belts and suspenders.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("The extractor returned no JSON — is this really an answer-key PDF?")
    return json.loads(m.group(0))


def _letter_for_row(q: int, letter: str) -> tuple[str, bool]:
    """Convert a booklet letter to the SHEET row's letter set, positionally."""
    row = ODD_SET if q % 2 == 1 else EVEN_SET
    if letter in row:
        return letter, False
    other = EVEN_SET if row is ODD_SET else ODD_SET
    if letter in other:
        return row[other.index(letter)], True
    raise ValueError(f"invalid answer letter {letter!r} for question {q}")


def validate_and_normalize(raw: dict) -> dict:
    """Enforce every invariant; returns the storable, trusted payload.

    Output: {test_id_hint, name, test_form, answers, field_test_answers|None,
             scaler, warnings, conversions}
    """
    warnings: list[str] = []
    conversions: list[str] = []
    sections_in = raw.get("sections") or {}
    scaler_in = raw.get("scaler") or {}
    if not sections_in:
        raise ValueError("No sections found in the extracted data.")

    answers_out: dict[str, dict[str, str]] = {}
    field_test_out: dict[str, dict[str, str]] = {}

    for sec, blob in sections_in.items():
        if sec not in SECTION_LABELS:
            raise ValueError(f"Unknown section {sec!r} (expected Test 1..Test 4).")
        label = SECTION_LABELS[sec]
        printed = {int(q): str(a).strip().upper() for q, a in (blob.get("answers") or {}).items()}
        if not printed:
            raise ValueError(f"{label}: no answers extracted.")
        not_scored = {int(q) for q in (blob.get("not_scored") or [])}
        unknown_ns = not_scored - set(printed)
        if unknown_ns:
            raise ValueError(f"{label}: not_scored lists questions missing from answers: {sorted(unknown_ns)}")

        scored_qs = sorted(q for q in printed if q not in not_scored)
        if not_scored:
            # Full-form booklet: positions are the real sheet positions, and
            # letters must already sit in the right parity set.
            for q in sorted(printed):
                conv, changed = _letter_for_row(q, printed[q])
                if changed:
                    raise ValueError(
                        f"{label} Q{q}: letter {printed[q]} doesn't match its row's letter "
                        "set, but this booklet keeps original positions — that combination "
                        "means the extraction is wrong. Re-check the PDF."
                    )
            answers_out[sec] = {str(q): printed[q] for q in scored_qs}
            field_test_out[sec] = {str(q): printed[q] for q in sorted(not_scored)}
        else:
            # Renumbered "My Answer Key" booklet (non-scored removed): must be
            # contiguous from 1, and letters convert positionally where the
            # renumbering flipped their parity.
            if scored_qs != list(range(1, len(scored_qs) + 1)):
                gaps = sorted(set(range(1, max(scored_qs) + 1)) - set(scored_qs))
                raise ValueError(
                    f"{label}: no questions are marked Not Scored, yet the numbering has "
                    f"gaps ({gaps[:10]}…) — either the booklet marks non-scored questions "
                    "(and extraction missed the markers) or the extraction dropped rows."
                )
            out = {}
            for q in scored_qs:
                conv, changed = _letter_for_row(q, printed[q])
                if changed:
                    conversions.append(f"{label} Q{q}: {printed[q]}→{conv}")
                out[str(q)] = conv
            answers_out[sec] = out

    # ----- scaler --------------------------------------------------------
    scaler_out: dict[str, dict[str, int]] = {}
    for sec, answers in answers_out.items():
        label = SECTION_LABELS[sec]
        table = scaler_in.get(sec)
        if not table:
            raise ValueError(f"{label}: no conversion (scaler) table extracted.")
        norm: dict[str, int] = {}
        for k, v in table.items():
            r, s = int(k), int(v)
            if str(r) in norm:
                raise ValueError(f"{label}: raw score {r} appears twice in the scaler.")
            if not (1 <= s <= 36):
                raise ValueError(f"{label}: scale score {s} out of range for raw {r}.")
            norm[str(r)] = s
        n_scored = len(answers)
        raws = sorted(int(k) for k in norm)
        missing = sorted(set(range(0, n_scored + 1)) - set(raws))
        extra = [r for r in raws if r > n_scored]
        if extra:
            raise ValueError(
                f"{label}: scaler covers raw scores up to {max(raws)} but only {n_scored} "
                "questions are scored — the scored/not-scored split is wrong somewhere."
            )
        if missing:
            raise ValueError(f"{label}: scaler is missing raw scores {missing[:10]} (0..{n_scored} required).")
        # THE clamp guard: max raw must equal the scored question count.
        assert max(raws) == n_scored
        scaler_out[sec] = norm

    missing_sections = [s for s in SECTION_LABELS if s not in answers_out]
    if missing_sections:
        warnings.append(
            "Missing sections: " + ", ".join(SECTION_LABELS[s] for s in missing_sections)
            + " — the test will grade only the sections present."
        )
    if conversions:
        warnings.append(
            f"{len(conversions)} answer letter(s) converted to sheet positions "
            "(renumbered booklet kept test-day letters)."
        )

    name = (raw.get("name") or "").strip() or None
    form = (raw.get("test_form") or "").strip() or None
    base = name or (f"Form {form}" if form else "imported test")
    slug = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_") or "imported"
    return {
        "test_id_hint": f"act_import_{slug}"[:60],
        "name": name or (f"ACT {form}" if form else "Imported ACT test"),
        "test_form": form,
        "answers": answers_out,
        "field_test_answers": field_test_out or None,
        "scaler": scaler_out,
        "warnings": warnings,
        "conversions": conversions,
    }


def sanity_check(payload: dict) -> None:
    """Cheap invariant re-check on the confirm step (payload round-trips the browser)."""
    for sec, answers in (payload.get("answers") or {}).items():
        for q, letter in answers.items():
            row = ODD_SET if int(q) % 2 == 1 else EVEN_SET
            if letter not in row:
                raise ValueError(f"{sec} Q{q}={letter}: wrong letter set for its row.")
        table = (payload.get("scaler") or {}).get(sec) or {}
        raws = sorted(int(k) for k in table)
        if not raws or max(raws) != len(answers) or raws != list(range(0, len(answers) + 1)):
            raise ValueError(f"{sec}: scaler no longer matches the scored question count.")
    for sec, ft in (payload.get("field_test_answers") or {}).items():
        overlap = set(ft) & set((payload.get("answers") or {}).get(sec, {}))
        if overlap:
            raise ValueError(f"{sec}: field-test questions {sorted(overlap)[:5]} also in the scored key.")
