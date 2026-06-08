"""Read a photograph of a filled bubble sheet into {question: answer}.

Two paths to the homography that takes a photo into the canonical sheet frame:

  * read_sheet     — uses 4 ArUco corner markers. Requires markered sheet.
  * read_sheet_fm  — uses ORB feature matching against a reference image of
                     the unmodified sheet. No markers needed.

Once the homography is known, the post-warp logic is identical: Otsu-threshold,
sample each bubble's fill ratio, decide answer per question.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .feature_match import FeatureMatchError, compute_homography


# Image-or-PDF inputs. Scanned submissions arrive as PDFs from school scanners;
# phone uploads arrive as JPG/PNG. Both routes converge to a grayscale ndarray.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _load_grayscale(path: Path | str, pdf_dpi: int = 300) -> np.ndarray:
    """Load an image, or rasterize page 1 of a PDF, into a grayscale numpy array.

    PDFs declared at a normal letter-size page get rendered at the requested
    DPI directly. Some scans (especially phone-photo PDFs from Drive) report
    page dimensions that match the photo's pixel count rather than the
    physical sheet, so rendering at "300 DPI" would produce a 7000+ px image
    that pushes ORB out of its scale-pyramid range. When that happens we
    fall back to rendering at a fixed pixel HEIGHT (~3300 px) so the result
    has roughly the same dimensions as the 300-DPI reference rasterization.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium  # local import: keeps non-PDF path cheap to start
        pdf = pdfium.PdfDocument(str(p))
        if len(pdf) == 0:
            raise ValueError(f"PDF has no pages: {p}")
        page = pdf[0]
        # PDF user-space units are 1/72 inch. For a normal 8.5×11 page that
        # gives a page height of 11 inches → 792 pt. Anything much taller
        # than that means the embedded source already has its own resolution
        # baked in, and we should target a pixel-height instead of DPI.
        w_pt, h_pt = page.get_size()
        scale = pdf_dpi / 72.0
        target_h_px = int(11.0 * pdf_dpi)  # ≈ 3300 at 300 DPI
        if h_pt * scale > target_h_px * 2:
            scale = target_h_px / h_pt
        bitmap = page.render(scale=scale, grayscale=True)
        return np.array(bitmap.to_pil())
    if suffix in _IMAGE_SUFFIXES:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {p}")
        return img
    raise ValueError(
        f"Unsupported input type {suffix!r}. Expected PDF or one of: "
        + ", ".join(sorted(_IMAGE_SUFFIXES))
    )

# Tuneables — the bubble-recognition thresholds.
# Fills are AFTER per-bubble baseline subtraction (the baseline removes the
# pre-printed letter ink), so a blank bubble reads ≈ 0 and even a light pencil
# mark cleanly clears MIN_FILL. AMBIGUOUS_REL rescues high-baseline scans where
# every bubble reads a touch dark but one bubble still stands out proportionally.
MIN_FILL = 0.10           # below this adjusted ratio → BLANK
                          # was 0.04; raised after the OMR was reporting
                          # false-positive fills on visibly empty bubbles
                          # (scan noise + light printed-letter density
                          # could nudge an unmarked bubble past 0.04).
AMBIGUOUS_DELTA = 0.05    # absolute gap between top and runner-up
                          # was 0.03; raised in tandem with MIN_FILL so a
                          # narrowly-edging top option can't commit when
                          # both top and runner-up are weakly marked.
AMBIGUOUS_REL = 0.25      # OR relative gap (top - second)/top
SAMPLE_SHRINK = 0.70      # sample inside this fraction of the printed radius


def _load_template(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def _detect_markers(gray: np.ndarray, dict_name: str) -> dict[int, np.ndarray]:
    """Return {marker_id: 4×2 corner array (clockwise from TL of the marker)}."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c[0] for i, c in zip(ids.flatten(), corners)}


def _sample_warped(
    warped: np.ndarray,
    template: dict,
    dpi: int,
    min_fill: float,
    ambiguous_delta: float,
) -> tuple[dict, dict, np.ndarray]:
    """Given a canonical-frame warped image, sample every bubble.

    Returns (answers, fills_per_q, binary_image).
    """
    px_per_mm = dpi / 25.4
    page_w_mm, page_h_mm = template["page_size_mm"]
    canon_w_px = int(page_w_mm * px_per_mm)
    canon_h_px = int(page_h_mm * px_per_mm)

    _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    fills_per_q: dict[int, list[dict]] = {}
    for b in template["bubbles"]:
        cx_mm, cy_mm = b["center_mm"]
        r_mm = b["radius_mm"]
        cx = int(cx_mm * px_per_mm)
        cy = int(cy_mm * px_per_mm)
        r = max(1, int(r_mm * px_per_mm * SAMPLE_SHRINK))
        y0, y1 = max(0, cy - r), min(canon_h_px, cy + r)
        x0, x1 = max(0, cx - r), min(canon_w_px, cx + r)
        roi = binary[y0:y1, x0:x1]
        if roi.size == 0:
            fill = 0.0
        else:
            mask = np.zeros_like(roi)
            cv2.circle(mask, (roi.shape[1] // 2, roi.shape[0] // 2), r, 255, -1)
            inside = cv2.bitwise_and(roi, mask)
            denom = int(mask.sum())
            fill = float(inside.sum() / denom) if denom > 0 else 0.0
        # Subtract the per-bubble printed-letter baseline so all options compete
        # on student pencil ink alone, not letter density.
        baseline = float(b.get("baseline_fill", 0.0))
        adjusted = max(0.0, fill - baseline)
        fills_per_q.setdefault(b["q"], []).append({"option": b["option"], "fill": adjusted})

    answers: dict[int, str] = {}
    for q in sorted(fills_per_q):
        ranked = sorted(fills_per_q[q], key=lambda b: -b["fill"])
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else {"fill": 0.0}
        abs_gap = top["fill"] - second["fill"]
        rel_gap = abs_gap / top["fill"] if top["fill"] > 0 else 0.0
        if top["fill"] < min_fill:
            answers[q] = "BLANK"
        elif abs_gap >= ambiguous_delta or rel_gap >= AMBIGUOUS_REL:
            # Either a clear absolute gap OR a clear relative gap — accept top.
            # The rel-gap branch rescues high-baseline scans where every bubble
            # reads a bit dark, but one bubble still stands out proportionally.
            answers[q] = top["option"]
        else:
            answers[q] = "MULTI"

    return answers, fills_per_q, binary


def _maybe_write_debug(
    debug_dir: Path | str | None,
    warped: np.ndarray,
    binary: np.ndarray,
    extras: dict[str, np.ndarray] | None = None,
) -> None:
    if debug_dir is None:
        return
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "warped.png"), warped)
    cv2.imwrite(str(debug_dir / "binary.png"), binary)
    for name, img in (extras or {}).items():
        cv2.imwrite(str(debug_dir / f"{name}.png"), img)


def read_sheet(
    image_path: Path | str,
    template_path: Path | str,
    debug_dir: Path | str | None = None,
    dpi: int = 200,
    min_fill: float = MIN_FILL,
    ambiguous_delta: float = AMBIGUOUS_DELTA,
) -> dict:
    """ArUco-based reader. Requires the photographed sheet to have corner markers.

    `image_path` can be a JPG/PNG photo or a scanned PDF.
    """
    template = _load_template(template_path)
    source_dpi = int(template.get("source_dpi", 300))
    gray = _load_grayscale(image_path, pdf_dpi=source_dpi)

    markers = _detect_markers(gray, template["fiducials"]["dict"])
    expected_ids = {m["id"] for m in template["fiducials"]["markers"]}
    if not expected_ids.issubset(markers.keys()):
        missing = expected_ids - markers.keys()
        raise ValueError(
            f"Missing ArUco markers: {sorted(missing)}. Detected: {sorted(markers.keys())}."
        )

    px_per_mm = dpi / 25.4
    page_w_mm, page_h_mm = template["page_size_mm"]
    canon_w_px = int(page_w_mm * px_per_mm)
    canon_h_px = int(page_h_mm * px_per_mm)
    marker_size_mm = template["fiducials"]["size_mm"]

    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    for m in template["fiducials"]["markers"]:
        detected = markers[m["id"]]
        cx_mm, cy_mm = m["center_mm"]
        half = marker_size_mm / 2
        canonical_corners = [
            [(cx_mm - half) * px_per_mm, (cy_mm - half) * px_per_mm],
            [(cx_mm + half) * px_per_mm, (cy_mm - half) * px_per_mm],
            [(cx_mm + half) * px_per_mm, (cy_mm + half) * px_per_mm],
            [(cx_mm - half) * px_per_mm, (cy_mm + half) * px_per_mm],
        ]
        for src, dst in zip(detected, canonical_corners):
            src_pts.append(src.tolist())
            dst_pts.append(dst)

    H, _ = cv2.findHomography(
        np.array(src_pts, dtype=np.float32),
        np.array(dst_pts, dtype=np.float32),
        method=0,
    )
    warped = cv2.warpPerspective(gray, H, (canon_w_px, canon_h_px))
    answers, fills, binary = _sample_warped(warped, template, dpi, min_fill, ambiguous_delta)
    _maybe_write_debug(debug_dir, warped, binary)
    return {"answers": answers, "fills": fills, "mode": "aruco"}


def read_sheet_fm(
    image_path: Path | str,
    template_path: Path | str,
    reference_path: Path | str,
    debug_dir: Path | str | None = None,
    dpi: int = 200,
    min_fill: float = MIN_FILL,
    ambiguous_delta: float = AMBIGUOUS_DELTA,
) -> dict:
    """Marker-free reader. Computes the homography by ORB-matching the photo
    against a reference rasterization of the unmodified sheet.

    `image_path` can be a JPG/PNG photo, a scanned PDF, or any other image format
    OpenCV understands. Scanned PDFs are rasterized at the template's source DPI.
    """
    template = _load_template(template_path)
    source_dpi = int(template.get("source_dpi", 300))
    gray = _load_grayscale(image_path, pdf_dpi=source_dpi)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise ValueError(f"Could not read reference: {reference_path}")

    # H_pr maps photo pixels → reference-image pixels (at the reference's DPI).
    H_pr, info = compute_homography(gray, reference)

    # We sample in a canonical frame at `dpi`. Reference is at `source_dpi`.
    source_dpi = int(template.get("source_dpi", 300))
    scale = dpi / source_dpi

    page_w_mm, page_h_mm = template["page_size_mm"]
    canon_w_px = int(page_w_mm * dpi / 25.4)
    canon_h_px = int(page_h_mm * dpi / 25.4)

    # Compose: photo → reference (at source_dpi) → scaled canon (at `dpi`).
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    H_final = S @ H_pr
    warped = cv2.warpPerspective(gray, H_final, (canon_w_px, canon_h_px))

    answers, fills, binary = _sample_warped(warped, template, dpi, min_fill, ambiguous_delta)
    _maybe_write_debug(debug_dir, warped, binary)
    return {
        "answers": answers,
        "fills": fills,
        "mode": "feature_match",
        "match_info": info,
    }
