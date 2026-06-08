"""Adapt an existing ACT-format answer sheet PDF for our OMR pipeline.

Strategy (Path B):
  1. Rasterize the PDF at high DPI.
  2. Auto-detect every printed bubble (outlined oval) via contour analysis.
  3. Group the detections into the ACT's 4-section layout and assign question
     numbers and option letters (A-D / F-J alternation, with 5 options for math).
  4. Overlay 4 ArUco markers in the page corners so our existing reader can
     compute a homography on photographed copies.
  5. Save: <name>.pdf (printable, with markers), <name>.png, <name>.template.json.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from .sheet import ARUCO_DICT_NAME, MARGIN, MARKER_SIZE, _aruco_image

# Page geometry — these match `sheet.py` so the same reader works.
PAGE_W_MM = 215.9
PAGE_H_MM = 279.4

# ACT-format layout. rows_per_col describes how questions stack down each
# column for that section; the last column is often shorter so the section fits.

# Legacy 75/60/40/40 ACT — used 2016 through mid-2025.
ACT_SECTIONS = [
    {"name": "Test 1", "n_questions": 75, "n_options": 4, "rows_per_col": [13, 13, 13, 13, 13, 10]},
    {"name": "Test 2", "n_questions": 60, "n_options": 5, "rows_per_col": [10, 10, 10, 10, 10, 10]},
    {"name": "Test 3", "n_questions": 40, "n_options": 4, "rows_per_col": [7, 7, 7, 7, 7, 5]},
    {"name": "Test 4", "n_questions": 40, "n_options": 4, "rows_per_col": [7, 7, 7, 7, 7, 5]},
]

# Shortened ACT (50/45/36/40) introduced 2025. Same bubble convention
# (odd → A-D[E], even → F-J[K]), different column counts per section.
NEW_FORMAT_SECTIONS = [
    {"name": "Test 1", "n_questions": 50, "n_options": 4, "rows_per_col": [13, 13, 13, 11]},
    {"name": "Test 2", "n_questions": 45, "n_options": 5, "rows_per_col": [10, 10, 10, 10, 5]},
    {"name": "Test 3", "n_questions": 36, "n_options": 4, "rows_per_col": [7, 7, 7, 7, 7, 1]},
    {"name": "Test 4", "n_questions": 40, "n_options": 4, "rows_per_col": [7, 7, 7, 7, 7, 5]},
]

# Named registry so callers can ask for a layout by string.
LAYOUTS = {
    "legacy": ACT_SECTIONS,
    "new":    NEW_FORMAT_SECTIONS,
}


def _options_for(question_num: int, n_options: int) -> str:
    """ACT convention: odd Qs use A-D(E), even Qs use F-J(K)."""
    base = "ABCDE" if question_num % 2 == 1 else "FGHJK"
    return base[:n_options]


def _cluster_by_y(bubbles: list[dict], tol_px: float) -> list[list[dict]]:
    """Group bubbles into visual rows by Y proximity. Returns rows in top-down order."""
    sorted_bs = sorted(bubbles, key=lambda b: b["cy"])
    rows: list[list[dict]] = []
    cur: list[dict] = []
    cur_cy: float | None = None
    for b in sorted_bs:
        if cur_cy is None or abs(b["cy"] - cur_cy) <= tol_px:
            cur.append(b)
            cur_cy = sum(x["cy"] for x in cur) / len(cur)
        else:
            rows.append(cur)
            cur = [b]
            cur_cy = float(b["cy"])
    if cur:
        rows.append(cur)
    return rows


def _split_into_sections(
    bubbles: list[dict],
    dpi: int,
    n_sections: int = 4,
    *,
    row_tol_mm: float = 2.0,
    section_gap_mm: float = 10.0,
    min_row_width: int = 18,
) -> list[list[list[dict]]]:
    """Drop header bubbles, group test bubbles into N section row-groups.

    Heuristic: bubbles cluster into rows. Test rows are wide (≥``min_row_width``
    bubbles); header rows (booklet number grid, examples) have far fewer so we
    drop rows below the threshold. A vertical gap >``section_gap_mm`` between
    consecutive test rows marks a section boundary.

    The new-format sheet has tighter row spacing and a narrower English test
    section (16 bubbles per row instead of 24+), so callers running the new
    layout pass smaller ``row_tol_mm`` and ``min_row_width`` than the legacy
    defaults.
    """
    px_per_mm = dpi / 25.4
    row_tol_px = row_tol_mm * px_per_mm
    section_gap_px = section_gap_mm * px_per_mm

    rows = _cluster_by_y(bubbles, tol_px=row_tol_px)
    # Track each row's mean Y so we can group rows into "section bands".
    rows_with_cy: list[tuple[float, list[dict]]] = [
        (sum(b["cy"] for b in r) / len(r), r) for r in rows
    ]

    # Stage 1 — pick out the wide rows. These reliably ARE test-question rows
    # (header bubbles like booklet number / form columns are narrower). The
    # Y-gap between them tells us where one section ends and the next begins.
    wide = [(cy, r) for (cy, r) in rows_with_cy if len(r) >= min_row_width]

    # Stage 2 — collapse the wide rows into N sections by Y-gap.
    wide_bands: list[list[tuple[float, list[dict]]]] = []
    band: list[tuple[float, list[dict]]] = []
    last_cy: float | None = None
    for cy, r in wide:
        if last_cy is not None and (cy - last_cy) > section_gap_px:
            wide_bands.append(band)
            band = []
        band.append((cy, r))
        last_cy = cy
    if band:
        wide_bands.append(band)

    # Stage 3 — extend each band's Y range to absorb adjacent narrow rows
    # (e.g. the new-format Test 1 has 12-bubble trailing rows where only
    # 3 of 4 columns reach the bottom of the section). We grow the band
    # downwards until either the Y-gap to the next row exceeds
    # section_gap_px or we land on a wide row that belongs to the next
    # band. Going upwards is symmetric.
    sections: list[list[list[dict]]] = []
    consumed: set[int] = set()  # row indices we've already assigned to a band
    rows_indexed = list(enumerate(rows_with_cy))
    rows_indexed.sort(key=lambda x: x[1][0])

    for wb in wide_bands:
        wb_top = min(cy for cy, _ in wb)
        wb_bot = max(cy for cy, _ in wb)
        in_band: list[tuple[float, list[dict]]] = list(wb)
        # Try to absorb rows above and below the band.
        for idx, (cy, r) in rows_indexed:
            if idx in consumed:
                continue
            if (cy, r) in wb:
                continue
            # Skip clearly-header rows (very narrow) by a softer floor —
            # bubble-grid rows for any plausible ACT section have at least
            # 4 bubbles per visible column × 1 column = 4. Below 4 is noise.
            if len(r) < 4:
                continue
            # Allow the band to swallow rows whose Y is within section_gap_px
            # of the band's current extent.
            if wb_top - section_gap_px <= cy <= wb_bot + section_gap_px:
                in_band.append((cy, r))
                wb_top = min(wb_top, cy)
                wb_bot = max(wb_bot, cy)
        # Sort the band's rows top-down for downstream labeling.
        in_band.sort(key=lambda x: x[0])
        for idx, (cy, r) in enumerate(rows_with_cy):
            for (bcy, br) in in_band:
                if br is r:
                    consumed.add(idx)
        sections.append([r for _, r in in_band])

    if len(sections) != n_sections:
        # Fall back: pick the n_sections largest groups by total bubble count.
        sections.sort(key=lambda s: -sum(len(r) for r in s))
        sections = sections[:n_sections]
        # Re-sort top-down for caller.
        sections.sort(key=lambda s: sum(b["cy"] for r in s for b in r) / sum(len(r) for r in s))
    return sections


def _find_canonical_xs(
    section_rows: list[list[dict]], n_canonical: int, tol_px: float = 8.0
) -> list[float]:
    """Across all rows in a section, return the `n_canonical` most-frequent X positions.

    Real bubble columns appear in (almost) every row. False-positive intruders (e.g.
    "0" digit characters in question numbers like 40, 50, 60) appear only in rows
    whose question number contains that digit, so they cluster at lower frequency.
    """
    all_xs: list[int] = [b["cx"] for r in section_rows for b in r]
    if not all_xs:
        return []
    all_xs.sort()
    clusters: list[list[int]] = [[all_xs[0]]]
    for x in all_xs[1:]:
        if x - clusters[-1][-1] <= tol_px:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    clusters.sort(key=lambda c: -len(c))
    top = clusters[:n_canonical]
    return sorted(sum(c) / len(c) for c in top)


def _snap_row_to_canonical(
    row: list[dict], canonical_xs: list[float], tol_px: float = 18.0
) -> list[dict]:
    """Drop bubbles not near a canonical X, keep one bubble per canonical slot."""
    used_slots: set[int] = set()
    kept: list[dict] = []
    for b in sorted(row, key=lambda b: b["cx"]):
        slot = min(range(len(canonical_xs)), key=lambda i: abs(canonical_xs[i] - b["cx"]))
        if abs(canonical_xs[slot] - b["cx"]) > tol_px:
            continue
        if slot in used_slots:
            continue
        kept.append(b)
        used_slots.add(slot)
    return kept


def _split_row_into_groups(row: list[dict], n_groups: int) -> list[list[dict]] | None:
    """Split a row's bubbles into `n_groups` question-groups via X-gap analysis."""
    sorted_row = sorted(row, key=lambda b: b["cx"])
    total = len(sorted_row)
    if total < n_groups:
        return None
    if total % n_groups != 0:
        # Unequal options per group — shouldn't happen on a well-detected row.
        return None
    # Find the (n_groups - 1) largest X-gaps between adjacent bubbles.
    gaps = [
        (sorted_row[i + 1]["cx"] - sorted_row[i]["cx"], i)
        for i in range(total - 1)
    ]
    gaps.sort(reverse=True, key=lambda g: g[0])
    boundary_indices = sorted(i for _, i in gaps[: n_groups - 1])
    groups: list[list[dict]] = []
    start = 0
    for bi in boundary_indices:
        groups.append(sorted_row[start : bi + 1])
        start = bi + 1
    groups.append(sorted_row[start:])
    return groups


def _label_section(
    section_rows: list[list[dict]],
    cfg: dict,
    q_offset: int = 0,
) -> tuple[list[dict], list[str]]:
    """Assign q-number + option letter to every bubble in a section.

    Returns (labeled_bubbles, warnings).
    """
    rows_per_col: list[int] = cfg["rows_per_col"]
    n_opts: int = cfg["n_options"]
    n_cols: int = len(rows_per_col)
    max_rows = max(rows_per_col)
    warnings: list[str] = []

    if len(section_rows) != max_rows:
        warnings.append(
            f"[{cfg['name']}] expected {max_rows} rows, got {len(section_rows)}"
        )

    labeled: list[dict] = []
    for row_idx, row in enumerate(section_rows[:max_rows]):
        # Which columns have a question in this row?
        cols_with_q = [c for c, rpc in enumerate(rows_per_col) if row_idx < rpc]
        n_groups = len(cols_with_q)
        expected_bubbles = n_groups * n_opts

        if len(row) != expected_bubbles:
            warnings.append(
                f"[{cfg['name']}] row {row_idx}: got {len(row)} bubbles, "
                f"expected {expected_bubbles} ({n_groups} groups × {n_opts} opts)"
            )
            continue

        groups = _split_row_into_groups(row, n_groups)
        if groups is None:
            warnings.append(f"[{cfg['name']}] row {row_idx}: split failed")
            continue

        for col_idx, group in zip(cols_with_q, groups):
            # Question number: sum of rows in earlier columns + (row_idx + 1)
            q_num_in_section = sum(rows_per_col[:col_idx]) + row_idx + 1
            q_num = q_offset + q_num_in_section
            opts = _options_for(q_num_in_section, n_opts)
            for opt, bubble in zip(opts, group):
                labeled.append(
                    {
                        **bubble,
                        "q": q_num,                       # global, 1..215
                        "q_in_test": q_num_in_section,    # what teachers see in answer keys
                        "section": cfg["name"],
                        "option": opt,
                    }
                )

    return labeled, warnings


def label_bubbles(
    detected: list[dict],
    dpi: int,
    sections_config: list[dict] | None = None,
) -> tuple[list[dict], list[str]]:
    """Top-level labeler: split into ACT sections and assign q/option to each bubble.

    Question numbering is global across all four tests so the existing reader
    (which keys by a single ``q``) works without modification. The per-test
    number is preserved in ``q_in_test`` for answer-key mapping.

    ``sections_config`` selects which layout to apply. Defaults to the
    legacy 75/60/40/40 ``ACT_SECTIONS``; pass ``NEW_FORMAT_SECTIONS`` for
    the shortened 50/45/36/40 format.
    """
    cfg_list = sections_config if sections_config is not None else ACT_SECTIONS
    # Pick the row-width floor from the SMALLEST section width — anything
    # narrower must be header noise. For the legacy layout the smallest
    # section is 24 bubbles wide (6×4); the new layout's English section
    # is just 16 (4×4), so we'd reject real rows with the legacy default.
    min_row_width = min(s["n_options"] * len(s["rows_per_col"]) for s in cfg_list)
    # New-format rows are also packed tighter vertically — about 4-5mm
    # apart vs ~6mm on the legacy sheet. Halve the Y-tolerance for "new"
    # so two adjacent rows don't merge into one.
    row_tol_mm = 1.0 if cfg_list is NEW_FORMAT_SECTIONS else 2.0
    sections = _split_into_sections(
        detected,
        dpi=dpi,
        n_sections=len(cfg_list),
        row_tol_mm=row_tol_mm,
        min_row_width=min_row_width,
    )
    all_warnings: list[str] = []
    labeled: list[dict] = []
    q_offset = 0
    for cfg, section_rows in zip(cfg_list, sections):
        section_labeled, warnings = _label_section(section_rows, cfg, q_offset=q_offset)
        labeled.extend(section_labeled)
        all_warnings.extend(warnings)
        q_offset += cfg["n_questions"]
    return labeled, all_warnings


def rasterize_pdf(pdf_path: Path | str, dpi: int = 300) -> np.ndarray:
    """Rasterize page 1 of `pdf_path` as a uint8 grayscale numpy array."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[0]
    scale = dpi / 72.0  # PDF units are points
    bitmap = page.render(scale=scale, grayscale=True)
    pil = bitmap.to_pil()
    return np.array(pil)


def _extract_bubble_template(gray: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Pick a clean bubble from the blank sheet and return it as a template image.

    Bootstrap step: we do one cheap contour-based scan, keep candidates with the
    right hollow-oval signature, and crop the one whose dimensions best match an
    ACT bubble (≈ 3.1mm × 2.2mm). The cropped chip is then the canonical pattern
    we hand to `cv2.matchTemplate`.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        raise RuntimeError("No contours found while extracting bubble template.")
    hierarchy = hierarchy[0]

    target_w = 3.1 * px_per_mm     # ~37 px at 300 DPI
    target_h = 2.2 * px_per_mm     # ~26 px
    best: tuple[int, int, int, int] | None = None
    best_score = float("inf")
    for i, c in enumerate(contours):
        if hierarchy[i][3] != -1 or hierarchy[i][2] == -1:
            continue  # require outer-with-hole = hollow oval
        x, y, w, h = cv2.boundingRect(c)
        # Loose size sanity, then pick by distance to ideal dimensions.
        if not (15 <= w <= 60 and 12 <= h <= 40):
            continue
        score = abs(w - target_w) + abs(h - target_h)
        if score < best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        raise RuntimeError("Could not extract a bubble template from this PDF.")
    x, y, w, h = best
    pad = max(2, int(0.3 * px_per_mm))
    h_img, w_img = gray.shape
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)
    return gray[y0:y1, x0:x1].copy()


def detect_bubbles(
    gray: np.ndarray,
    px_per_mm: float,
    template: np.ndarray | None = None,
    match_threshold: float = 0.65,
) -> list[dict]:
    """Find every bubble in `gray` via `cv2.matchTemplate`.

    Returns one dict per bubble with cx, cy in pixel coords and the template's
    width/height. Digits, dots, smudges score poorly against the bubble template
    and don't survive the threshold — no aspect/hole/size filters needed.
    """
    if template is None:
        template = _extract_bubble_template(gray, px_per_mm=px_per_mm)
    th_h, th_w = template.shape

    # Normalized cross-correlation: similarity map in [-1, 1]; bubble locations
    # appear as compact high-similarity blobs (one per real bubble).
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    mask = (result >= match_threshold).astype(np.uint8) * 255

    # Each surviving blob is one bubble; its centroid in `result` coords is the
    # top-left of where the template fits — add half-template to get bubble center.
    n_labels, _labels, _stats, centroids = cv2.connectedComponentsWithStats(mask)

    bubbles: list[dict] = []
    for i in range(1, n_labels):
        cx_corr, cy_corr = centroids[i]
        bubbles.append(
            {
                "cx": int(round(cx_corr + th_w / 2)),
                "cy": int(round(cy_corr + th_h / 2)),
                "w": int(th_w),
                "h": int(th_h),
            }
        )
    return bubbles


def annotate_detections(gray: np.ndarray, bubbles: list[dict]) -> np.ndarray:
    """Return a BGR image with detected bubbles outlined in red for visual review."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for b in bubbles:
        cv2.circle(bgr, (b["cx"], b["cy"]), max(b["w"], b["h"]) // 2 + 2, (0, 0, 255), 2)
    return bgr


def overlay_aruco_markers(gray: np.ndarray, dpi: int) -> tuple[np.ndarray, list[dict]]:
    """Draw 4 ArUco markers in the page corners; return (image, marker_template_entries)."""
    out = gray.copy()
    px_per_mm = dpi / 25.4
    marker_size_px = int(MARKER_SIZE * px_per_mm)
    centers_mm = [
        (MARGIN + MARKER_SIZE / 2, MARGIN + MARKER_SIZE / 2),
        (PAGE_W_MM - MARGIN - MARKER_SIZE / 2, MARGIN + MARKER_SIZE / 2),
        (PAGE_W_MM - MARGIN - MARKER_SIZE / 2, PAGE_H_MM - MARGIN - MARKER_SIZE / 2),
        (MARGIN + MARKER_SIZE / 2, PAGE_H_MM - MARGIN - MARKER_SIZE / 2),
    ]
    marker_entries = []
    for mid, (cx_mm, cy_mm) in enumerate(centers_mm):
        marker = _aruco_image(mid, marker_size_px)
        cx_px = int(cx_mm * px_per_mm)
        cy_px = int(cy_mm * px_per_mm)
        x0 = cx_px - marker_size_px // 2
        y0 = cy_px - marker_size_px // 2
        out[y0 : y0 + marker_size_px, x0 : x0 + marker_size_px] = marker
        marker_entries.append(
            {
                "id": mid,
                "corner": ["TL", "TR", "BR", "BL"][mid],
                "center_mm": [round(cx_mm, 3), round(cy_mm, 3)],
            }
        )
    return out, marker_entries


def prepare_act_sheet(
    src_pdf: Path | str,
    out_dir: Path | str,
    name: str = "act_sheet",
    dpi: int = 300,
    layout: str = "legacy",
) -> dict:
    """Full pipeline: rasterize → detect → filter+label → overlay markers → save.

    ``layout`` selects the section config: ``"legacy"`` for the 75/60/40/40
    sheet (default), ``"new"`` for the 50/45/36/40 short-format sheet.

    Outputs under out_dir/:
      <name>.pdf            — printable, with ArUco corner markers
      <name>.png            — same as PDF, raster
      <name>.detected.png   — debug overlay: red circles around every detection
      <name>.labeled.png    — debug overlay: kept bubbles, green
      <name>.template.json  — compatible with omr.read_sheet
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    px_per_mm = dpi / 25.4
    sections_config = LAYOUTS.get(layout)
    if sections_config is None:
        raise ValueError(f"Unknown layout {layout!r}; choose from {list(LAYOUTS)}.")
    expected_total = sum(s["n_questions"] * s["n_options"] for s in sections_config)

    gray = rasterize_pdf(src_pdf, dpi=dpi)
    # Save the unmodified rasterization — used as the reference image for the
    # feature-matching reader (`read-fm`) which doesn't rely on ArUco markers.
    cv2.imwrite(str(out_dir / f"{name}.reference.png"), gray)
    detected = detect_bubbles(gray, px_per_mm=px_per_mm)
    labeled, warnings = label_bubbles(detected, dpi=dpi, sections_config=sections_config)

    # Diagnostic overlays
    cv2.imwrite(str(out_dir / f"{name}.detected.png"), annotate_detections(gray, detected))
    label_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for b in labeled:
        cv2.circle(label_overlay, (b["cx"], b["cy"]), max(b["w"], b["h"]) // 2 + 2, (0, 180, 0), 2)
    cv2.imwrite(str(out_dir / f"{name}.labeled.png"), label_overlay)

    # Printable sheet with ArUco markers overlaid
    with_markers, marker_entries = overlay_aruco_markers(gray, dpi=dpi)
    cv2.imwrite(str(out_dir / f"{name}.png"), with_markers)
    Image.fromarray(with_markers).convert("RGB").save(
        out_dir / f"{name}.pdf", "PDF", resolution=float(dpi)
    )

    # Build the OMR-compatible template. Use a single radius per bubble derived
    # from the median detected size for stability.
    if labeled:
        median_radius_px = float(np.median([max(b["w"], b["h"]) / 2 for b in labeled]))
    else:
        median_radius_px = 2.0 * px_per_mm
    radius_mm = round(median_radius_px / px_per_mm, 3)

    bubbles_out: list[dict] = []
    for b in labeled:
        bubbles_out.append(
            {
                "q": b["q"],
                "q_in_test": b["q_in_test"],
                "section": b["section"],
                "option": b["option"],
                "center_mm": [round(b["cx"] / px_per_mm, 3), round(b["cy"] / px_per_mm, 3)],
                "radius_mm": radius_mm,
            }
        )

    # Compute per-bubble baseline fill from the unmodified rasterization.
    # ACT bubbles have a printed letter inside (F/G/H/J etc.) whose ink
    # density varies — a printed G covers more pixels than a printed F. If we
    # threshold student scans directly, the heavier-printed letter can read
    # "fuller" than a lightly-marked neighbour. Storing the baseline lets the
    # OMR reader subtract it to isolate the actual pencil mark.
    SAMPLE_SHRINK = 0.70  # must match omr.SAMPLE_SHRINK
    _, baseline_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    h_ref, w_ref = baseline_binary.shape
    for b in bubbles_out:
        cx = int(b["center_mm"][0] * px_per_mm)
        cy = int(b["center_mm"][1] * px_per_mm)
        r = max(1, int(b["radius_mm"] * px_per_mm * SAMPLE_SHRINK))
        y0, y1 = max(0, cy - r), min(h_ref, cy + r)
        x0, x1 = max(0, cx - r), min(w_ref, cx + r)
        roi = baseline_binary[y0:y1, x0:x1]
        if roi.size == 0:
            b["baseline_fill"] = 0.0
            continue
        mask = np.zeros_like(roi)
        cv2.circle(mask, (roi.shape[1] // 2, roi.shape[0] // 2), r, 255, -1)
        inside = cv2.bitwise_and(roi, mask)
        denom = int(mask.sum())
        b["baseline_fill"] = round(
            float(inside.sum() / denom) if denom > 0 else 0.0, 4
        )

    template = {
        "version": 1,
        "kind": "act",
        "layout": layout,
        "page_size_mm": [PAGE_W_MM, PAGE_H_MM],
        "sections": sections_config,
        "n_questions": sum(s["n_questions"] for s in sections_config),
        "source_dpi": dpi,                          # DPI of <name>.reference.png
        "reference_image": f"{name}.reference.png",
        "fiducials": {
            "dict": ARUCO_DICT_NAME,
            "size_mm": MARKER_SIZE,
            "markers": marker_entries,
        },
        "bubbles": bubbles_out,
        "_detected_count": len(detected),
        "_labeled_count": len(labeled),
        "_expected_count": expected_total,
        "_warnings": warnings,
    }
    (out_dir / f"{name}.template.json").write_text(json.dumps(template, indent=2))

    return {
        "detected": len(detected),
        "labeled": len(labeled),
        "expected": expected_total,
        "warnings": warnings,
        "out_dir": str(out_dir),
    }
