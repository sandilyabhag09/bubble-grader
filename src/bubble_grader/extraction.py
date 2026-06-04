"""Extract ACT answer keys and scale-score tables from PDFs via Tesseract OCR.

The PDFs we get from prep books (e.g. The Real ACT Prep Guide) are scanned
images with no embedded text. We rasterize each page, run Tesseract, then
parse the OCR output with regular expressions tuned to the known layouts.

Output formats match `scoring.py` expectations:
  Answer key:  {section_name: {q_in_test: option}}
  Scaler:      {section_name: {raw_score: scaled_score}}

Sections are mapped to our internal canonical names:
  English      → "Test 1"
  Mathematics  → "Test 2"
  Reading      → "Test 3"
  Science      → "Test 4"
"""

import re
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytesseract
from PIL import Image

# Maps the section-header text we look for in OCR'd pages → our canonical name.
# Includes subscore-area keywords because the actual banner headers (e.g.
# "Mathematics ■ Scoring Key") are styled images Tesseract often can't read,
# while the subscore footnotes are plain text and reliably extracted.
_SECTION_PATTERNS: dict[str, re.Pattern] = {
    "Test 1": re.compile(r"\benglish\b|usage\s*/?\s*mechanics|rhetorical\s+skills", re.IGNORECASE),
    "Test 2": re.compile(
        r"\bmathematics\b|pre[\s-]?algebra|elementary\s+algebra|coordinate\s+geometry|"
        r"plane\s+geometry|trigonometry",
        re.IGNORECASE,
    ),
    "Test 3": re.compile(r"\breading\b|social\s+studies|arts\s*/\s*literature", re.IGNORECASE),
    "Test 4": re.compile(r"\bscience\b(?!s)", re.IGNORECASE),  # exclude "Sciences"
}
_SECTION_ORDER = ["Test 1", "Test 2", "Test 3", "Test 4"]

# Sections in the order they appear as columns in a scale-score table.
_SCALER_COLUMNS = ["Test 1", "Test 2", "Test 3", "Test 4"]

# Per-section answer-key page layouts (rows_per_col, in left-to-right column order).
# Question N is at column floor((N-1) / rows_in_first_col) — but column lengths
# can differ, so we walk columns and accumulate.
_SECTION_LAYOUT: dict[str, list[int]] = {
    "Test 1": [25, 25, 25],   # English: 75q, 3 even columns
    "Test 2": [30, 30],       # Math: 60q, 2 even columns
    "Test 3": [14, 14, 12],   # Reading: 40q, ragged 3-col
    "Test 4": [14, 14, 12],   # Science: 40q, ragged 3-col
}


def _grid_to_qnum(col_idx: int, row_idx: int, layout: list[int]) -> int:
    """Translate (column, row) into the per-section question number."""
    return sum(layout[:col_idx]) + row_idx + 1

# ACT-valid option letters. Odd-numbered Qs use A-E, even use F-K. Letters
# like "I", "L", "M"-"Z" never appear, so they're useful negative filters
# against OCR noise (e.g. dashes/blanks mis-read as letter shapes).
_OPT_RE = re.compile(r"[ABCDEFGHJK]")
_CELL_RE = re.compile(r"\d+-\d+|\d+|[-—=]")


def _rasterize_pages(pdf_path: Path | str, dpi: int = 300) -> list[np.ndarray]:
    pdf = pdfium.PdfDocument(str(pdf_path))
    return [
        np.array(p.render(scale=dpi / 72, grayscale=True).to_pil()) for p in pdf
    ]


def _ocr(image: np.ndarray, config: str = "") -> str:
    return pytesseract.image_to_string(Image.fromarray(image), config=config)


def _identify_section(text: str) -> str | None:
    """Return our canonical section name if a page's text reveals which test it is."""
    for name, pat in _SECTION_PATTERNS.items():
        if pat.search(text):
            return name
    return None


def _parse_answers(text: str) -> dict[int, str]:
    """Pull {q: option} pairs out of OCR'd answer-key text.

    A "row" in the source PDF holds multiple questions side-by-side
    (e.g. `1. D    26. J    51. B`). After OCR this collapses to one
    text line, so we treat the whole text as a stream of `<num>. <stuff>`
    markers and grab the LAST ACT-valid letter inside each segment between
    markers. The "last valid letter" heuristic compensates for OCR noise
    on the left of each cell (underline blanks mis-read as `O`/`C`).
    """
    out: dict[int, str] = {}
    markers = list(re.finditer(r"\b(\d{1,3})[.,]?\s", text))
    for i, m in enumerate(markers):
        q = int(m.group(1))
        if not (1 <= q <= 80):
            continue
        seg_start = m.end()
        seg_end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        segment_full = text[seg_start:seg_end]
        # Each answer cell lives on a single line. Trim to first newline so the
        # footer after the last marker (e.g. "...Mechanics RH = Rhetorical...")
        # doesn't bleed in.
        nl = segment_full.find("\n")
        segment = segment_full[:nl] if nl != -1 else segment_full
        # Answer cells are 1-3 chars after OCR. Anything longer is prose
        # (e.g. "STEP 1." / "Practice Test 4" matched as a digit-marker).
        if len(segment.strip()) > 8:
            continue
        candidates = _OPT_RE.findall(segment)
        if candidates and q not in out:
            out[q] = candidates[-1].upper()
    return out


def _collect_letter_positions(image: np.ndarray) -> list[dict]:
    """Use Tesseract's per-word output with bounding boxes to find every ACT-valid letter.

    Returns a list of {letter, x, y} dicts where (x, y) is the center pixel of the
    detected word. We deliberately ignore long words (likely prose, not key cells)
    and short words containing no A-K letter (e.g. dashes, digits).
    """
    data = pytesseract.image_to_data(
        Image.fromarray(image), output_type=pytesseract.Output.DICT
    )
    valid = set("ABCDEFGHJK")
    out: list[dict] = []
    for i, txt in enumerate(data["text"]):
        s = (txt or "").strip().upper()
        if not s or len(s) > 3:
            continue
        cands = [c for c in s if c in valid]
        if not cands:
            continue
        out.append(
            {
                "letter": cands[-1],
                "x": data["left"][i] + data["width"][i] // 2,
                "y": data["top"][i] + data["height"][i] // 2,
            }
        )
    return out


def _position_based_fill(
    image: np.ndarray, layout: list[int], existing: dict[int, str]
) -> dict[int, str]:
    """Recover missing questions by snapping letter positions onto the expected grid.

    Algorithm:
      1. Get every ACT-valid letter in the page with its (x, y).
      2. Cluster Y values → row positions (max(layout) of them).
      3. Cluster X values → column positions (len(layout) of them).
      4. For each (col, row) grid slot, if no answer exists yet for that
         question, pick the closest detected letter.
    """
    n_cols = len(layout)
    max_rows = max(layout)
    total_q = sum(layout)
    expected_qs = set(range(1, total_q + 1))
    missing = expected_qs - set(existing)
    if not missing:
        return {}

    letters = _collect_letter_positions(image)
    if len(letters) < total_q // 2:
        return {}  # not enough signal to reconstruct the grid

    # Cluster Y values into row positions. We look for groups of letters with
    # similar Y; the centroid of each group is a row's vertical position.
    def _cluster_1d(values: list[float], target_k: int, tol: float) -> list[float]:
        if not values:
            return []
        values = sorted(values)
        clusters: list[list[float]] = [[values[0]]]
        for v in values[1:]:
            if v - clusters[-1][-1] <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        # Pick the target_k most-populated clusters (drop header/footer noise).
        clusters.sort(key=lambda c: -len(c))
        return sorted(sum(c) / len(c) for c in clusters[:target_k])

    row_ys = _cluster_1d([L["y"] for L in letters], max_rows, tol=25)
    col_xs = _cluster_1d([L["x"] for L in letters], n_cols, tol=80)
    if len(row_ys) < max_rows or len(col_xs) < n_cols:
        return {}

    # First, assign each detected letter to its nearest grid slot. Letters
    # that don't sit tightly inside one cell get discarded — they're noise.
    # Then we only fill MISSING slots whose assigned letter is unique to them.
    row_pitch = (row_ys[-1] - row_ys[0]) / max(1, len(row_ys) - 1) if len(row_ys) > 1 else 30
    y_tol = max(8.0, row_pitch * 0.40)  # ~40% of row pitch — can't bleed into next row
    x_tol = 50.0                        # columns are far apart; loose is fine

    # Map slot (col_idx, row_idx) → list of detected letters claiming it.
    slot_claims: dict[tuple[int, int], list[str]] = {}
    for L in letters:
        # Nearest column
        col_idx = min(range(n_cols), key=lambda i: abs(L["x"] - col_xs[i]))
        if abs(L["x"] - col_xs[col_idx]) > x_tol:
            continue
        # Nearest row
        row_idx = min(range(max_rows), key=lambda i: abs(L["y"] - row_ys[i]))
        if abs(L["y"] - row_ys[row_idx]) > y_tol:
            continue
        # Skip slots that don't exist (ragged last column)
        if row_idx >= layout[col_idx]:
            continue
        slot_claims.setdefault((col_idx, row_idx), []).append(L["letter"])

    # Recover only missing slots where exactly one letter claims that slot.
    recovered: dict[int, str] = {}
    for (col_idx, row_idx), claims in slot_claims.items():
        q = _grid_to_qnum(col_idx, row_idx, layout)
        if q not in missing:
            continue
        # Pick one letter only if all claimants agree (or there's exactly one).
        unique = set(claims)
        if len(unique) == 1:
            recovered[q] = next(iter(unique))
    return recovered


def extract_answer_key(
    pdf_path: Path | str,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    """Return {test_form, answers: {section: {q_in_test: option}}}.

    Strategy:
      1. OCR every page and try header-based section detection.
      2. Drop pages with no question/answer pairs (intro pages, scaler page).
      3. For answer pages whose section is still unknown, fall back to the
         conventional ACT order (English → Math → Reading → Science).
      4. Merge answers per section; flag mismatched counts in `_warnings`.
    """
    pages = _rasterize_pages(pdf_path)
    per_page: list[dict] = []
    for i, img in enumerate(pages):
        text = _ocr(img)
        per_page.append(
            {
                "page": i + 1,
                "image": img,
                "section": _identify_section(text),
                "answers": _parse_answers(text),
            }
        )

    # Only pages with meaningfully many parsed answers are "answer pages".
    # Scaler pages and prose intro pages will fall below this threshold.
    answer_pages = [p for p in per_page if len(p["answers"]) >= 10]

    # If exactly 4 answer pages and some are unlabeled, assign by page order.
    if len(answer_pages) == 4:
        for p, default_section in zip(answer_pages, _SECTION_ORDER):
            if p["section"] not in _SECTION_ORDER:
                p["section"] = default_section

    # First pass: gather text-OCR answers per section.
    answers: dict[str, dict[int, str]] = {}
    page_for_section: dict[str, np.ndarray] = {}
    for p in answer_pages:
        if p["section"] not in _SECTION_ORDER:
            continue
        answers.setdefault(p["section"], {}).update(p["answers"])
        page_for_section[p["section"]] = p["image"]

    # Second pass: position-based recovery of any still-missing questions.
    recovery_log: dict[str, dict[int, str]] = {}
    if expected_counts:
        for section, want in expected_counts.items():
            got = answers.get(section, {})
            if len(got) >= want:
                continue
            img = page_for_section.get(section)
            if img is None:
                continue
            recovered = _position_based_fill(img, _SECTION_LAYOUT[section], got)
            if recovered:
                recovery_log[section] = recovered
                answers[section].update(recovered)

    warnings: list[str] = []
    if expected_counts:
        for section, want in expected_counts.items():
            got = len(answers.get(section, {}))
            if got != want:
                warnings.append(
                    f"{section}: extracted {got} answers, expected {want}"
                )
    # Also record how each page was assigned, useful for debugging.
    assignments = [
        f"page {p['page']}: section={p['section']}, parsed {len(p['answers'])} answers"
        for p in per_page
    ]

    return {
        "test_form": Path(pdf_path).stem,
        "answers": {
            section: {str(q): opt for q, opt in qs.items()}
            for section, qs in answers.items()
        },
        "_warnings": warnings,
        "_page_assignments": assignments,
        "_recovered_by_position": {
            section: {str(q): opt for q, opt in qs.items()}
            for section, qs in recovery_log.items()
        },
    }


def _expand_cell(cell: str) -> list[int]:
    """A scaler cell can be `42`, `45-46`, or `-`. Return raw scores it covers."""
    if cell in ("-", "—", "="):
        return []
    if "-" in cell:
        lo, hi = (int(x) for x in cell.split("-"))
        return list(range(lo, hi + 1))
    return [int(cell)]


def _parse_scaler_rows(text: str) -> dict[str, dict[int, int]]:
    """Parse the scale-score conversion table OCR text.

    Expected line shape after tokenization:
        <scale>  <english_cell>  <math_cell>  <reading_cell>  <science_cell>  <scale>
    where each cell is a number, a range like `47-48`, or a dash placeholder.
    """
    scaler: dict[str, dict[int, int]] = {s: {} for s in _SCALER_COLUMNS}

    for raw_line in text.split("\n"):
        tokens = raw_line.strip().split()
        if len(tokens) < 6:
            continue
        # First and last tokens must both parse as the same integer scale 1..36.
        try:
            scale_l = int(tokens[0])
            scale_r = int(tokens[-1])
        except ValueError:
            continue
        if scale_l != scale_r or not (1 <= scale_l <= 36):
            continue
        # Keep only middle tokens that look like cell content; filters stray letters.
        middle = [t for t in tokens[1:-1] if _CELL_RE.fullmatch(t)]
        if len(middle) != 4:
            continue
        for section, cell in zip(_SCALER_COLUMNS, middle):
            for raw in _expand_cell(cell):
                scaler[section][raw] = scale_l
    return scaler


def extract_scaler(pdf_path: Path | str) -> dict:
    """Return {section: {raw: scaled}} parsed from any page containing a scale table."""
    pages = _rasterize_pages(pdf_path)
    scaler: dict[str, dict[int, int]] = {s: {} for s in _SCALER_COLUMNS}

    for img in pages:
        text = _ocr(img)
        if "Scale" not in text and "scale" not in text:
            continue
        page_scaler = _parse_scaler_rows(text)
        for section, table in page_scaler.items():
            scaler[section].update(table)

    # Sanity-check: each section should have a contiguous range of raws including 0.
    warnings: list[str] = []
    for section, table in scaler.items():
        if not table:
            warnings.append(f"{section}: no scaler rows extracted")
            continue
        raws = sorted(table)
        if raws[0] != 0:
            warnings.append(f"{section}: scaler starts at raw={raws[0]}, expected 0")
        # Find max gap in coverage
        gaps = [r for prev, r in zip(raws, raws[1:]) if r - prev > 1]
        if gaps:
            warnings.append(
                f"{section}: coverage has gaps at raws={gaps[:5]}{'...' if len(gaps)>5 else ''}"
            )

    return {
        "test_form": Path(pdf_path).stem,
        "scaler": {
            section: {str(raw): scaled for raw, scaled in table.items()}
            for section, table in scaler.items()
        },
        "_warnings": warnings,
    }
