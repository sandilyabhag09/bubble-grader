"""Regenerate the About page's example images from a made-up answer sheet.

Fills the current ACT answer sheet (act_sheet_scored) with invented answers,
runs it through the real reader, scorer and marked-up-sheet renderer, and
saves matching "before" and "after" crops of Tests 1-2. No real student data.

    uv run python scripts/make_about_images.py
"""

import json
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from bubble_grader.config import DATA_DIR
from bubble_grader.feedback import build_overlay_pdf
from bubble_grader.omr import read_sheet_fm
from bubble_grader.scoring import grade_answers

SHEETS = DATA_DIR / "sheets"
STEM = "act_sheet_scored"
OUT = Path(__file__).resolve().parents[1] / "src/bubble_grader/static/about"
CROP_MM = (22.0, 70.5, 176.0, 153.0)   # x0, y0, x1, y1: Test 1 + Test 2 with headers
OUT_WIDTH = 1200
OVERLAY_DPI = 200


def main() -> None:
    template_path = SHEETS / f"{STEM}.template.json"
    reference_path = SHEETS / f"{STEM}.reference.png"
    template = json.loads(template_path.read_text())
    src_dpi = int(template.get("source_dpi", 300))

    rows: dict[int, list[dict]] = defaultdict(list)
    for b in template["bubbles"]:
        rows[b["q"]].append(b)

    rng = random.Random(20261019)
    key: dict[str, dict[int, str]] = defaultdict(dict)
    marks: dict[int, list[str]] = {}
    for q, bubbles in sorted(rows.items()):
        opts = [b["option"] for b in bubbles]
        sec, qit = bubbles[0]["section"], bubbles[0]["q_in_test"]
        right = rng.choice(opts)
        key[sec][qit] = right
        r = rng.random()
        if r < 0.03:
            marks[q] = []                                   # left blank
        elif r < 0.045:
            marks[q] = [right, rng.choice([o for o in opts if o != right])]   # two filled
        elif r < 0.17:
            marks[q] = [rng.choice([o for o in opts if o != right])]          # wrong
        else:
            marks[q] = [right]

    # Pencil-ish marks on the blank sheet: dark grey, slightly off-centre and uneven.
    sheet = np.array(Image.open(reference_path).convert("L"))
    px = src_dpi / 25.4
    by_q_opt = {(b["q"], b["option"]): b for b in template["bubbles"]}
    for q, chosen in marks.items():
        for opt in chosen:
            b = by_q_opt[(q, opt)]
            cx = int(b["center_mm"][0] * px + rng.uniform(-2, 2))
            cy = int(b["center_mm"][1] * px + rng.uniform(-2, 2))
            r = b["radius_mm"] * px * rng.uniform(0.70, 0.82)
            cv2.ellipse(sheet, (cx, cy), (int(r * rng.uniform(1.0, 1.15)), int(r)),
                        rng.uniform(0, 180), 0, 360, int(rng.uniform(40, 75)), -1)
    paper = np.clip(sheet.astype(np.int16) * 0.94 + 10, 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        scan = Path(tmp) / "made-up-sheet.png"
        Image.fromarray(paper).save(scan)

        read = read_sheet_fm(scan, template_path, reference_path)
        got = {int(q): a for q, a in read["answers"].items()}
        expected = {q: ("BLANK" if not m else "MULTI" if len(m) > 1 else m[0]) for q, m in marks.items()}
        wrong_reads = {q: (expected[q], got.get(q)) for q in expected if got.get(q) != expected[q]}
        if wrong_reads:
            raise SystemExit(f"reader disagreed with the made-up sheet: {wrong_reads}")

        sections = grade_answers(got, template, key)
        details = [d for s in sections.values() for d in s["details"]]
        pdf = build_overlay_pdf(scan, template_path, reference_path, dpi=OVERLAY_DPI, details=details)
        page = pdfium.PdfDocument(pdf)[0]
        after = page.render(scale=OVERLAY_DPI / 72).to_pil().convert("RGB")

    def crop(img: Image.Image, dpi: int) -> Image.Image:
        k = dpi / 25.4
        x0, y0, x1, y1 = (int(v * k) for v in CROP_MM)
        c = img.crop((x0, y0, x1, y1))
        return c.resize((OUT_WIDTH, int(c.height * OUT_WIDTH / c.width)), Image.LANCZOS)

    OUT.mkdir(parents=True, exist_ok=True)
    before = crop(Image.fromarray(paper).convert("RGB"), src_dpi)
    marked = crop(after, OVERLAY_DPI)
    before.save(OUT / "sheet-before.webp", "WEBP", quality=80, method=6)
    marked.save(OUT / "sheet-after.webp", "WEBP", quality=82, method=6)

    shown = {q for q, ms in marks.items()
             if any(CROP_MM[1] <= by_q_opt[(q, o)]["center_mm"][1] <= CROP_MM[3] for o in [r["option"] for r in rows[q]])}
    tally = defaultdict(int)
    for d in details:
        if d["q"] in shown:
            tally[d["status"]] += 1
    print("all", len(expected), "made-up answers read correctly")
    print("in the crop:", dict(tally))
    for f in ("sheet-before.webp", "sheet-after.webp"):
        p = OUT / f
        print(f, Image.open(p).size, f"{p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
