"""Generate printable bubble-sheet PDFs and their machine-readable template JSON.

Coordinate system: millimeters, origin at top-left of the page (matches image coords).
ArUco markers live in the four corners. The template JSON records every bubble
center so the OMR reader is purely a lookup against detected fiducial geometry.
"""

import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# US Letter
PAGE_W_MM = 215.9
PAGE_H_MM = 279.4

# Layout constants (mm). Tweak here only.
MARGIN = 12
MARKER_SIZE = 12          # ArUco pattern is 6×6 modules; this is its drawn size
TITLE_H = 12
BUBBLE_RADIUS = 2.0
BUBBLE_GAP = 1.6          # horizontal gap between bubbles in a row
ROW_HEIGHT = 5.2          # vertical pitch between question rows
QNUM_WIDTH = 9            # horizontal space reserved for the "001." label

ARUCO_DICT_NAME = "DICT_4X4_50"


def build_template(n_questions: int = 215, n_options: int = 5) -> dict:
    """Compute bubble + fiducial geometry for a sheet of n questions × n options.

    Returns a dict with everything the renderer and the OMR reader need.
    """
    assert 1 <= n_options <= 5, "n_options must be 1..5"
    options = "ABCDE"[:n_options]

    grid_top = MARGIN + MARKER_SIZE + 4 + TITLE_H
    grid_bottom = PAGE_H_MM - MARGIN - MARKER_SIZE - 4
    grid_left = MARGIN
    grid_right = PAGE_W_MM - MARGIN
    grid_h = grid_bottom - grid_top
    grid_w = grid_right - grid_left

    rows_per_col = int(grid_h / ROW_HEIGHT)
    n_cols = math.ceil(n_questions / rows_per_col)
    col_pitch = grid_w / n_cols

    bubbles = []
    qnums = []
    bubble_diameter = BUBBLE_RADIUS * 2
    for q in range(1, n_questions + 1):
        col_idx = (q - 1) // rows_per_col
        row_idx = (q - 1) % rows_per_col
        col_x = grid_left + col_idx * col_pitch
        row_y = grid_top + (row_idx + 0.5) * ROW_HEIGHT

        qnums.append({"q": q, "anchor_mm": [round(col_x, 3), round(row_y, 3)]})

        for i, opt in enumerate(options):
            cx = (
                col_x
                + QNUM_WIDTH
                + BUBBLE_RADIUS
                + i * (bubble_diameter + BUBBLE_GAP)
            )
            cy = row_y
            bubbles.append(
                {
                    "q": q,
                    "option": opt,
                    "center_mm": [round(cx, 3), round(cy, 3)],
                    "radius_mm": BUBBLE_RADIUS,
                }
            )

    marker_centers = [
        (MARGIN + MARKER_SIZE / 2, MARGIN + MARKER_SIZE / 2),                                # TL  id=0
        (PAGE_W_MM - MARGIN - MARKER_SIZE / 2, MARGIN + MARKER_SIZE / 2),                    # TR  id=1
        (PAGE_W_MM - MARGIN - MARKER_SIZE / 2, PAGE_H_MM - MARGIN - MARKER_SIZE / 2),        # BR  id=2
        (MARGIN + MARKER_SIZE / 2, PAGE_H_MM - MARGIN - MARKER_SIZE / 2),                    # BL  id=3
    ]

    return {
        "version": 1,
        "page_size_mm": [PAGE_W_MM, PAGE_H_MM],
        "options": options,
        "n_questions": n_questions,
        "rows_per_col": rows_per_col,
        "n_cols": n_cols,
        "fiducials": {
            "dict": ARUCO_DICT_NAME,
            "size_mm": MARKER_SIZE,
            "markers": [
                {
                    "id": i,
                    "corner": ["TL", "TR", "BR", "BL"][i],
                    "center_mm": [round(c[0], 3), round(c[1], 3)],
                }
                for i, c in enumerate(marker_centers)
            ],
        },
        "qnums": qnums,
        "bubbles": bubbles,
    }


def _aruco_image(marker_id: int, size_px: int) -> np.ndarray:
    """Render a single ArUco marker as a uint8 grayscale array of size_px × size_px."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, ARUCO_DICT_NAME)
    )
    return cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)


def render_sheet(
    template: dict,
    fills: dict[int, str] | None = None,
    dpi: int = 300,
) -> Image.Image:
    """Render the sheet as a PIL grayscale image. Pass `fills` to draw filled bubbles."""
    page_w_mm, page_h_mm = template["page_size_mm"]
    px_per_mm = dpi / 25.4
    w_px = int(page_w_mm * px_per_mm)
    h_px = int(page_h_mm * px_per_mm)

    # White canvas
    arr = 255 * np.ones((h_px, w_px), dtype=np.uint8)

    # ArUco markers
    marker_size_mm = template["fiducials"]["size_mm"]
    marker_size_px = int(marker_size_mm * px_per_mm)
    for m in template["fiducials"]["markers"]:
        marker = _aruco_image(m["id"], marker_size_px)
        cx_px = int(m["center_mm"][0] * px_per_mm)
        cy_px = int(m["center_mm"][1] * px_per_mm)
        x0 = cx_px - marker_size_px // 2
        y0 = cy_px - marker_size_px // 2
        arr[y0 : y0 + marker_size_px, x0 : x0 + marker_size_px] = marker

    # Bubbles: hollow circles, then dark fill if requested
    for b in template["bubbles"]:
        cx = int(b["center_mm"][0] * px_per_mm)
        cy = int(b["center_mm"][1] * px_per_mm)
        r = int(b["radius_mm"] * px_per_mm)
        cv2.circle(arr, (cx, cy), r, 0, thickness=max(1, int(0.4 * px_per_mm / 4)))
        if fills and fills.get(b["q"]) == b["option"]:
            cv2.circle(arr, (cx, cy), int(r * 0.78), 0, thickness=-1)

    # Question numbers + per-bubble option letters: use PIL for text (cleaner than cv2.putText)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font_q = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=int(2.6 * px_per_mm))
        font_o = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=int(1.6 * px_per_mm))
    except OSError:
        font_q = ImageFont.load_default()
        font_o = ImageFont.load_default()

    for qn in template["qnums"]:
        ax, ay = qn["anchor_mm"]
        x_px = int(ax * px_per_mm)
        y_px = int(ay * px_per_mm) - int(1.6 * px_per_mm)
        draw.text((x_px, y_px), f"{qn['q']}.", fill=0, font=font_q)

    # Option letters under bubbles
    for b in template["bubbles"]:
        cx = int(b["center_mm"][0] * px_per_mm)
        cy = int(b["center_mm"][1] * px_per_mm)
        r = int(b["radius_mm"] * px_per_mm)
        # Drawn under each bubble center
        draw.text(
            (cx - int(0.7 * px_per_mm), cy + r + int(0.3 * px_per_mm)),
            b["option"],
            fill=0,
            font=font_o,
        )

    # Title
    draw.text(
        (int((page_w_mm / 2 - 30) * px_per_mm), int((MARGIN + MARKER_SIZE + 4) * px_per_mm)),
        f"Bubble Sheet  —  {template['n_questions']} questions × {len(template['options'])} options",
        fill=0,
        font=font_q,
    )

    return img


def save_pdf(image: Image.Image, out_path: Path) -> None:
    image.convert("RGB").save(out_path, "PDF", resolution=300.0)


def generate_sheet(
    out_dir: Path,
    name: str = "sheet",
    n_questions: int = 215,
    n_options: int = 5,
    dpi: int = 300,
) -> tuple[Path, Path, Path]:
    """Write {name}.pdf, {name}.png, and {name}.template.json under out_dir.

    Returns (pdf_path, png_path, template_json_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    tpl_path = out_dir / f"{name}.template.json"

    template = build_template(n_questions=n_questions, n_options=n_options)
    img = render_sheet(template, dpi=dpi)
    save_pdf(img, pdf_path)
    img.save(png_path, "PNG")
    tpl_path.write_text(json.dumps(template, indent=2))
    return pdf_path, png_path, tpl_path


def simulate_fill(
    template: dict,
    answers: dict[int, str],
    dpi: int = 300,
    base_image: Image.Image | np.ndarray | None = None,
) -> Image.Image:
    """Return a sheet image with `answers` bubbled in.

    If `base_image` is provided (e.g. a pre-rasterized ACT sheet that we don't
    have qnum geometry for), we overlay fills onto it. Otherwise we render the
    sheet from scratch using the template's geometry.
    """
    if base_image is None:
        return render_sheet(template, fills=answers, dpi=dpi)

    if isinstance(base_image, Image.Image):
        arr = np.array(base_image.convert("L"))
    else:
        arr = base_image.copy()

    px_per_mm = dpi / 25.4
    for b in template["bubbles"]:
        if answers.get(b["q"]) != b["option"]:
            continue
        cx = int(b["center_mm"][0] * px_per_mm)
        cy = int(b["center_mm"][1] * px_per_mm)
        r = max(2, int(b["radius_mm"] * px_per_mm * 0.78))
        cv2.circle(arr, (cx, cy), r, 0, thickness=-1)
    return Image.fromarray(arr)
