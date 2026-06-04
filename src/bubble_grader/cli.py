"""CLI: bubble-grader admin + sheet generation + OMR commands."""

import json
from pathlib import Path

import click

from .classroom import (
    create_coursework,
    list_courses,
    list_coursework,
    list_roster,
    list_submissions,
)
from .config import DATA_DIR
from .db import init_db, list_teachers
from .act_sheet import prepare_act_sheet
from .extraction import extract_answer_key, extract_scaler
from .scoring import full_grade
from . import db as dbmod
from .omr import read_sheet, read_sheet_fm
from .sheet import build_template, generate_sheet, simulate_fill
from .feedback import send_feedback_for_assignment
from .submissions import fetch_assignment, grade_classroom_assignment, release_grades


@click.group()
def cli() -> None:
    """bubble-grader admin CLI."""
    init_db()


@cli.command("serve")
def cmd_serve() -> None:
    """Run the OAuth + admin web server."""
    from .server import main

    main()


@cli.command("update")
def cmd_update() -> None:
    """Pull the latest code from GitHub and re-sync dependencies.

    Run this when the project maintainer says they've shipped a new feature.
    Safe to run anytime — it never touches your local DB, OAuth tokens, or
    secrets directory.
    """
    import shutil
    import subprocess
    from .config import PROJECT_ROOT

    if not (PROJECT_ROOT / ".git").exists():
        click.echo("Not a git checkout — `bubble-grader update` only works when "
                   "the project was installed via `git clone`. Skipping.")
        raise click.exceptions.Exit(1)

    click.echo("Fetching latest code…")
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only"],
            stderr=subprocess.STDOUT, text=True,
        )
        click.echo(out.strip())
    except subprocess.CalledProcessError as e:
        click.echo(f"git pull failed:\n{e.output}")
        raise click.exceptions.Exit(1)

    if not shutil.which("uv"):
        click.echo("uv not on PATH; skipping dependency sync. Run `uv sync` manually.")
        return
    click.echo("\nSyncing dependencies…")
    subprocess.check_call(["uv", "sync"], cwd=str(PROJECT_ROOT))
    click.echo("\n✓ Updated. Restart the server to pick up changes.")


@cli.command("setup")
@click.option("--force-key", is_flag=True,
              help="Regenerate FERNET_KEY even if one already exists in .env (DESTROYS prior tokens).")
def cmd_setup(force_key: bool) -> None:
    """First-run wizard: verify prerequisites, generate .env, initialize the DB.

    Safe to re-run — pure verification when everything's already in place.
    """
    import shutil
    import sys
    from .config import CLIENT_SECRET_PATH, PROJECT_ROOT, DATA_DIR

    click.echo("== bubble-grader setup ==\n")
    failures: list[str] = []

    # 1. Python version (we required 3.12 in pyproject, but verify at runtime).
    if sys.version_info >= (3, 12):
        click.echo(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    else:
        failures.append(f"Python 3.12+ required (have {sys.version.split()[0]}).")

    # 2. Tesseract OCR (needed for answer-key extraction from scanned PDFs).
    tess = shutil.which("tesseract")
    if tess:
        click.echo(f"  ✓ Tesseract at {tess}")
    else:
        failures.append("Tesseract not found. macOS install: `brew install tesseract`")

    # 3. client_secret.json from the project owner.
    if CLIENT_SECRET_PATH.exists():
        click.echo(f"  ✓ {CLIENT_SECRET_PATH.relative_to(PROJECT_ROOT)} present")
    else:
        failures.append(
            f"{CLIENT_SECRET_PATH.relative_to(PROJECT_ROOT)} not found. "
            "Place the file you were given at that path, then re-run setup."
        )

    # 4. FERNET_KEY — generate if absent (or if --force-key).
    env_path = PROJECT_ROOT / ".env"
    have_key = False
    existing_lines: list[str] = []
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("FERNET_KEY=") and line.split("=", 1)[1].strip():
                have_key = True
            existing_lines.append(line)

    if not have_key or force_key:
        from cryptography.fernet import Fernet
        new_key = Fernet.generate_key().decode()
        out_lines: list[str] = []
        wrote_key = False
        for line in existing_lines:
            if line.startswith("FERNET_KEY="):
                out_lines.append(f"FERNET_KEY={new_key}")
                wrote_key = True
            else:
                out_lines.append(line)
        if not wrote_key:
            out_lines.append(f"FERNET_KEY={new_key}")
        # Ensure the other env defaults exist so the server can boot.
        defaults = {
            "OAUTH_REDIRECT_URI": "http://localhost:8765/oauth/callback",
            "SERVER_PORT": "8765",
        }
        for k, v in defaults.items():
            if not any(line.startswith(f"{k}=") for line in out_lines):
                out_lines.append(f"{k}={v}")
        env_path.write_text("\n".join(out_lines) + "\n")
        click.echo(f"  ✓ Wrote FERNET_KEY + defaults to {env_path.relative_to(PROJECT_ROOT)}")
    else:
        click.echo(f"  ✓ FERNET_KEY already present in {env_path.relative_to(PROJECT_ROOT)}")

    # 5. Initialize the SQLite database (safe to call repeatedly).
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from . import db as dbmod
    dbmod.init_db()
    click.echo(f"  ✓ Database initialized at data/bubble_grader.db")

    if failures:
        click.echo("\n⚠ Setup incomplete:")
        for f in failures:
            click.echo(f"   • {f}")
        click.echo("\nFix the above, then re-run `uv run bubble-grader setup`.")
        sys.exit(1)

    click.echo("\nAll set! Start the web UI with:")
    click.echo("    uv run bubble-grader serve")
    click.echo("…then open http://localhost:8765 in your browser and sign in.\n")


@cli.command("teachers")
def cmd_teachers() -> None:
    """List teachers with stored credentials."""
    for t in list_teachers():
        click.echo(t)


@cli.command("courses")
@click.argument("email")
def cmd_courses(email: str) -> None:
    """List active Classroom courses for a stored teacher."""
    for c in list_courses(email):
        click.echo(f"{c['id']}\t{c['name']}")


@cli.command("coursework")
@click.argument("email")
@click.argument("course_id")
def cmd_coursework(email: str, course_id: str) -> None:
    """List assignments/questions in a course."""
    for cw in list_coursework(email, course_id):
        click.echo(
            f"{cw['id']}\t{cw.get('workType', '?'):20s}\t"
            f"{cw.get('state', '?'):10s}\t{cw.get('title', '')}"
        )


@cli.command("students")
@click.argument("email")
@click.argument("course_id")
def cmd_students(email: str, course_id: str) -> None:
    """List students enrolled in a course."""
    for s in list_roster(email, course_id):
        profile = s.get("profile") or {}
        name = (profile.get("name") or {}).get("fullName", "?")
        click.echo(
            f"{s['userId']}\t{profile.get('emailAddress', '?'):40s}\t{name}"
        )


@cli.command("submissions")
@click.argument("email")
@click.argument("course_id")
@click.argument("cw_id")
def cmd_submissions(email: str, course_id: str, cw_id: str) -> None:
    """List submissions for one assignment."""
    for s in list_submissions(email, course_id, cw_id):
        atts = (s.get("assignmentSubmission") or {}).get("attachments", []) or []
        click.echo(
            f"{s['userId']}\t{s.get('state', '?'):15s}\t{len(atts)} attachment(s)"
        )


@cli.command("fetch")
@click.argument("email")
@click.argument("course_id")
@click.argument("cw_id")
@click.option(
    "--all", "fetch_all", is_flag=True,
    help="Include submissions not yet in TURNED_IN state.",
)
def cmd_fetch(email: str, course_id: str, cw_id: str, fetch_all: bool) -> None:
    """Download every Drive attachment for an assignment's submissions."""
    manifest = fetch_assignment(
        email, course_id, cw_id, only_turned_in=not fetch_all
    )
    n_students = len(manifest["students"])
    n_files = sum(len(s["files"]) for s in manifest["students"].values())
    click.echo(f"Downloaded {n_files} file(s) for {n_students} student(s).")
    click.echo(
        f"Manifest written to data/submissions/{course_id}/{cw_id}/manifest.json"
    )
    # Also dump a brief summary to stdout for quick eyeballing.
    click.echo(json.dumps(
        {
            sid: {"name": s["name"], "files": [f.get("name") for f in s["files"]]}
            for sid, s in manifest["students"].items()
        },
        indent=2,
    ))


@cli.command("generate-sheet")
@click.option("--questions", "n_questions", default=215, show_default=True,
              help="Number of questions on the sheet.")
@click.option("--options", "n_options", default=5, show_default=True,
              help="Number of answer choices per question (1..5).")
@click.option("--name", default="sheet", show_default=True,
              help="Output filename stem.")
@click.option("--out", "out_dir", default=None,
              help="Output directory (default: data/sheets/).")
def cmd_generate_sheet(n_questions: int, n_options: int, name: str, out_dir: str | None) -> None:
    """Generate sheet.pdf + sheet.png + sheet.template.json."""
    target = Path(out_dir) if out_dir else DATA_DIR / "sheets"
    pdf, png, tpl = generate_sheet(target, name=name, n_questions=n_questions, n_options=n_options)
    click.echo(f"PDF:      {pdf}")
    click.echo(f"PNG:      {png}")
    click.echo(f"Template: {tpl}")


@cli.command("prepare-act")
@click.argument("src_pdf", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", default="act_sheet", show_default=True,
              help="Output filename stem.")
@click.option("--out", "out_dir", default=None,
              help="Output directory (default: data/sheets/).")
@click.option("--dpi", default=300, show_default=True,
              help="Rasterization DPI.")
def cmd_prepare_act(src_pdf: str, name: str, out_dir: str | None, dpi: int) -> None:
    """Detect bubbles in an existing ACT-format PDF and overlay ArUco markers."""
    target = Path(out_dir) if out_dir else DATA_DIR / "sheets"
    result = prepare_act_sheet(src_pdf, target, name=name, dpi=dpi)
    click.echo(f"Detected:  {result['detected']} bubbles total")
    click.echo(f"Labeled:   {result['labeled']} (kept after section filter)")
    click.echo(f"Expected:  {result['expected']} (4 ACT sections)")
    click.echo(f"Output:    {result['out_dir']}/{name}.{{pdf,png,detected.png,labeled.png,template.json}}")
    if result["warnings"]:
        click.echo("\nWarnings:")
        for w in result["warnings"]:
            click.echo(f"  - {w}")
    elif result["labeled"] != result["expected"]:
        click.echo(
            f"\n⚠  Labeled count ({result['labeled']}) ≠ expected "
            f"({result['expected']}). Review {name}.labeled.png."
        )


@cli.command("simulate")
@click.argument("template_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--answers", "answers_json", default=None,
              help='Inline JSON like \'{"1":"A","2":"C"}\'. If omitted, fills a deterministic pattern.')
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False),
              help="Where to write the simulated filled PNG.")
def cmd_simulate(template_path: str, answers_json: str | None, out_path: str) -> None:
    """Render a sheet with bubbles filled in (for round-trip testing the reader)."""
    template = json.loads(Path(template_path).read_text())
    if answers_json:
        raw = json.loads(answers_json)
        answers = {int(k): v for k, v in raw.items()}
    else:
        # Deterministic pattern: cycle through options based on q index.
        opts = template["options"]
        answers = {q: opts[(q - 1) % len(opts)] for q in range(1, template["n_questions"] + 1)}
    img = simulate_fill(template, answers)
    img.save(out_path, "PNG")
    click.echo(f"Wrote {out_path} with {len(answers)} bubbles filled.")


@cli.command("read")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--template", "template_path", required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--debug", "debug_dir", default=None,
              help="If set, writes warped + binary intermediates here.")
@click.option("--json-out", "json_out", default=None,
              help="If set, writes the full {answers, fills} JSON here.")
@click.option("--min-fill", default=0.30, show_default=True,
              help="Below this fill ratio a bubble is treated as BLANK.")
@click.option("--ambiguous-delta", default=0.10, show_default=True,
              help="If top and runner-up fills are within this delta the question is MULTI.")
def cmd_read(image_path: str, template_path: str, debug_dir: str | None,
             json_out: str | None, min_fill: float, ambiguous_delta: float) -> None:
    """Read a filled bubble sheet and print {question: answer}."""
    template = json.loads(Path(template_path).read_text())
    result = read_sheet(
        image_path, template_path,
        debug_dir=debug_dir, min_fill=min_fill, ambiguous_delta=ambiguous_delta,
    )
    if json_out:
        Path(json_out).write_text(json.dumps(result, indent=2))
    answers = result["answers"]

    # Map global q → section + q_in_test if available (ACT case)
    q_to_section: dict[int, tuple[str, int]] = {}
    for b in template.get("bubbles", []):
        if "section" in b and "q_in_test" in b:
            q_to_section[b["q"]] = (b["section"], b["q_in_test"])

    by_answer: dict[str, int] = {}
    for v in answers.values():
        by_answer[v] = by_answer.get(v, 0) + 1
    click.echo(f"Read {len(answers)} questions:")
    for k in sorted(by_answer):
        click.echo(f"  {k:6s}: {by_answer[k]}")

    if q_to_section:
        click.echo("\nPer-section breakdown:")
        per_section: dict[str, dict[str, int]] = {}
        for q, ans in answers.items():
            section = q_to_section[q][0]
            per_section.setdefault(section, {})
            per_section[section][ans] = per_section[section].get(ans, 0) + 1
        for sec in sorted(per_section):
            buckets = per_section[sec]
            click.echo(f"  {sec}: " + ", ".join(f"{k}={v}" for k, v in sorted(buckets.items())))

    anomalies = [q for q, a in answers.items() if a in ("BLANK", "MULTI")]
    if anomalies:
        click.echo(f"\n⚠  {len(anomalies)} BLANK/MULTI questions:")
        for q in anomalies[:30]:
            if q_to_section:
                sec, qit = q_to_section[q]
                click.echo(f"  Q{q} ({sec} #{qit}): {answers[q]}")
            else:
                click.echo(f"  Q{q}: {answers[q]}")
        if len(anomalies) > 30:
            click.echo(f"  ... +{len(anomalies) - 30} more")


@cli.command("import-key")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", default=None,
              help="JSON output path (default: data/scoring/<stem>.key.json).")
@click.option("--counts", default="75,60,40,40", show_default=True,
              help="Comma-separated expected question counts for sections in order Test 1..4.")
def cmd_import_key(pdf_path: str, out_path: str | None, counts: str) -> None:
    """Extract an answer key from a (possibly scanned) PDF via OCR.

    Output JSON has shape `{test_form, answers: {section: {q: option}}}`.
    Questions filled in via position-based recovery are listed in
    `_recovered_by_position` so a teacher knows where to spot-check.
    """
    expected = dict(zip(["Test 1", "Test 2", "Test 3", "Test 4"], (int(c) for c in counts.split(","))))
    result = extract_answer_key(pdf_path, expected_counts=expected)

    if out_path is None:
        target = DATA_DIR / "scoring" / f"{Path(pdf_path).stem}.key.json"
    else:
        target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))

    click.echo(f"Wrote {target}")
    for section in sorted(result["answers"]):
        got = len(result["answers"][section])
        want = expected.get(section, "?")
        marker = "✓" if got == want else "✗"
        click.echo(f"  {marker} {section}: {got}/{want}")

    recovered = result.get("_recovered_by_position", {})
    if recovered:
        click.echo("\n⚠  These were filled in via position-based recovery — VERIFY before grading:")
        for section, qs in recovered.items():
            for q, opt in sorted(qs.items(), key=lambda kv: int(kv[0])):
                click.echo(f"    {section} Q{q} = {opt}")

    if result["_warnings"]:
        click.echo("\nWarnings:")
        for w in result["_warnings"]:
            click.echo(f"  - {w}")


@cli.command("import-scaler")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", default=None,
              help="JSON output path (default: data/scoring/<stem>.scaler.json).")
def cmd_import_scaler(pdf_path: str, out_path: str | None) -> None:
    """Extract the raw-to-scaled conversion table from a PDF."""
    result = extract_scaler(pdf_path)
    if out_path is None:
        target = DATA_DIR / "scoring" / f"{Path(pdf_path).stem}.scaler.json"
    else:
        target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))

    click.echo(f"Wrote {target}")
    for section, table in result["scaler"].items():
        if table:
            raws = sorted(int(r) for r in table.keys())
            click.echo(f"  {section}: raws {raws[0]}..{raws[-1]} ({len(raws)} entries)")
        else:
            click.echo(f"  {section}: empty")
    if result["_warnings"]:
        click.echo("\nWarnings:")
        for w in result["_warnings"]:
            click.echo(f"  - {w}")


@cli.command("read-fm")
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--template", "template_path", required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--reference", "reference_path", default=None,
              help="Path to the unmodified sheet rasterization. Defaults to the "
                   "reference_image field in the template, resolved next to it.")
@click.option("--debug", "debug_dir", default=None,
              help="If set, writes warped + binary intermediates here.")
@click.option("--json-out", "json_out", default=None,
              help="If set, writes the full {answers, fills, match_info} JSON here.")
@click.option("--min-fill", default=0.30, show_default=True)
@click.option("--ambiguous-delta", default=0.10, show_default=True)
def cmd_read_fm(
    image_path: str,
    template_path: str,
    reference_path: str | None,
    debug_dir: str | None,
    json_out: str | None,
    min_fill: float,
    ambiguous_delta: float,
) -> None:
    """Read a filled bubble sheet by feature-matching against a reference (no markers needed)."""
    template = json.loads(Path(template_path).read_text())
    if reference_path is None:
        ref_name = template.get("reference_image")
        if not ref_name:
            raise click.ClickException(
                "Template has no `reference_image`; pass --reference explicitly."
            )
        reference_path = str(Path(template_path).parent / ref_name)

    result = read_sheet_fm(
        image_path, template_path, reference_path,
        debug_dir=debug_dir, min_fill=min_fill, ambiguous_delta=ambiguous_delta,
    )
    if json_out:
        Path(json_out).write_text(json.dumps(result, indent=2))

    answers = result["answers"]
    info = result["match_info"]
    click.echo(
        f"Feature match: {info['n_inliers']}/{info['n_good_matches']} inliers "
        f"({info['inlier_ratio']*100:.1f}%), keypoints photo={info['n_keypoints_photo']} ref={info['n_keypoints_reference']}"
    )

    q_to_section: dict[int, tuple[str, int]] = {}
    for b in template.get("bubbles", []):
        if "section" in b and "q_in_test" in b:
            q_to_section[b["q"]] = (b["section"], b["q_in_test"])

    by_answer: dict[str, int] = {}
    for v in answers.values():
        by_answer[v] = by_answer.get(v, 0) + 1
    click.echo(f"\nRead {len(answers)} questions:")
    for k in sorted(by_answer):
        click.echo(f"  {k:6s}: {by_answer[k]}")

    if q_to_section:
        click.echo("\nPer-section breakdown:")
        per_section: dict[str, dict[str, int]] = {}
        for q, ans in answers.items():
            section = q_to_section[q][0]
            per_section.setdefault(section, {})
            per_section[section][ans] = per_section[section].get(ans, 0) + 1
        for sec in sorted(per_section):
            buckets = per_section[sec]
            click.echo(f"  {sec}: " + ", ".join(f"{k}={v}" for k, v in sorted(buckets.items())))

    anomalies = [q for q, a in answers.items() if a in ("BLANK", "MULTI")]
    if anomalies:
        click.echo(f"\n⚠  {len(anomalies)} BLANK/MULTI questions (showing first 30):")
        for q in anomalies[:30]:
            sec, qit = q_to_section.get(q, ("?", q))
            click.echo(f"  Q{q} ({sec} #{qit}): {answers[q]}")


### `test` subgroup: manage stored tests + their keys + scalers ----------------

@cli.group("test")
def test_group() -> None:
    """Manage saved ACT practice tests (answer keys + scalers in the DB)."""


@test_group.command("add")
@click.argument("test_id")
@click.option("--name", required=True, help="Display name for the test.")
@click.option("--notes", default=None, help="Free-text notes about the test.")
def cmd_test_add(test_id: str, name: str, notes: str | None) -> None:
    """Register a new test."""
    dbmod.upsert_test(test_id, name=name, notes=notes)
    click.echo(f"Saved test '{test_id}': {name}")


@test_group.command("list")
def cmd_test_list() -> None:
    """List all registered tests + which artifacts they have."""
    rows = dbmod.list_tests()
    if not rows:
        click.echo("(no tests registered yet; use `bubble-grader test add` to start)")
        return
    click.echo(f"{'ID':<25} {'Name':<35} {'Key':<5} {'Scaler':<6}")
    click.echo("-" * 75)
    for r in rows:
        key = "✓" if r["has_key"] else "·"
        scl = "✓" if r["has_scaler"] else "·"
        click.echo(f"{r['id']:<25} {(r['name'] or ''):<35} {key:<5} {scl:<6}")


@test_group.command("show")
@click.argument("test_id")
def cmd_test_show(test_id: str) -> None:
    """Show a test's metadata + completeness."""
    t = dbmod.get_test(test_id)
    if t is None:
        raise click.ClickException(f"No test '{test_id}'.")
    click.echo(f"id:        {t['id']}")
    click.echo(f"name:      {t['name']}")
    click.echo(f"notes:     {t['notes'] or '-'}")
    click.echo(f"created:   {t['created_at']}")
    click.echo(f"updated:   {t['updated_at']}")
    if t["answer_key"]:
        click.echo("answer key:")
        for section, qs in t["answer_key"].items():
            click.echo(f"  {section}: {len(qs)} answers")
    else:
        click.echo("answer key: (not loaded)")
    if t["scaler"]:
        click.echo("scaler:")
        for section, table in t["scaler"].items():
            click.echo(f"  {section}: {len(table)} raw→scaled entries")
    else:
        click.echo("scaler: (not loaded)")


@test_group.command("delete")
@click.argument("test_id")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def cmd_test_delete(test_id: str, yes: bool) -> None:
    """Delete a test and ALL its submissions."""
    if not yes:
        click.confirm(
            f"Delete test '{test_id}' and all its submissions? This can't be undone.",
            abort=True,
        )
    ok = dbmod.delete_test(test_id)
    click.echo("Deleted." if ok else f"No test '{test_id}'.")


@test_group.command("set-key")
@click.argument("test_id")
@click.option("--from", "from_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="JSON file holding {test_form, answers: {section: {q: opt}}} "
                   "or just the inner {section: {q: opt}} dict.")
def cmd_test_set_key(test_id: str, from_path: str) -> None:
    """Load the answer key for a test from a JSON file."""
    data = json.loads(Path(from_path).read_text())
    inner = data.get("answers", data)
    if not isinstance(inner, dict):
        raise click.ClickException("JSON must contain an 'answers' object or be a {section: {...}} dict.")
    dbmod.set_test_answer_key(test_id, inner)
    counts = {section: len(qs) for section, qs in inner.items()}
    click.echo(f"Loaded answer key for '{test_id}': {counts}")


@test_group.command("set-scaler")
@click.argument("test_id")
@click.option("--from", "from_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="JSON file holding {test_form, scaler: {section: {raw: scaled}}} "
                   "or just the inner dict.")
def cmd_test_set_scaler(test_id: str, from_path: str) -> None:
    """Load the raw→scaled conversion table for a test from a JSON file."""
    data = json.loads(Path(from_path).read_text())
    inner = data.get("scaler", data)
    if not isinstance(inner, dict):
        raise click.ClickException("JSON must contain a 'scaler' object or be a {section: {...}} dict.")
    dbmod.set_test_scaler(test_id, inner)
    counts = {section: len(table) for section, table in inner.items()}
    click.echo(f"Loaded scaler for '{test_id}': {counts}")


### grade + submissions -------------------------------------------------------

@cli.command("grade")
@click.argument("answers_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--test", "test_id", required=True, help="Test id (from `test list`).")
@click.option("--template", "template_path",
              default="data/sheets/act_sheet.template.json", show_default=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--save/--no-save", default=True, show_default=True,
              help="Persist the result as a submission row.")
@click.option("--student-id", default=None)
@click.option("--student-name", default=None)
@click.option("--student-email", default=None)
def cmd_grade(
    answers_path: str,
    test_id: str,
    template_path: str,
    save: bool,
    student_id: str | None,
    student_name: str | None,
    student_email: str | None,
) -> None:
    """Grade an answers JSON file against a saved test."""
    t = dbmod.get_test(test_id)
    if t is None:
        raise click.ClickException(f"No test '{test_id}'. Add it first with `test add`.")
    if not t["answer_key"]:
        raise click.ClickException(f"Test '{test_id}' has no answer key; use `test set-key`.")

    raw_answers = json.loads(Path(answers_path).read_text())
    # `read-fm --json-out` writes {"answers": {...}, "fills": ..., ...}; accept that or a bare dict.
    answers_dict = raw_answers.get("answers", raw_answers)
    answers = {int(k): v for k, v in answers_dict.items()}

    template = json.loads(Path(template_path).read_text())
    report = full_grade(answers, template, t["answer_key"], t["scaler"])

    # Pretty-print
    click.echo(f"Test: {t['name']} ({test_id})")
    click.echo("-" * 40)
    for section, info in report["sections"].items():
        raw = info["raw_score"]
        scaled = info.get("scaled_score", "-")
        click.echo(
            f"  {section:8s} raw={raw}/{info['n_questions']}  scaled={scaled}  "
            f"(correct={info['n_correct']} wrong={info['n_incorrect']} "
            f"blank={info['n_blank']} multi={info['n_multi']})"
        )
    click.echo(f"\nComposite: {report.get('composite', '-')}")

    if save:
        sub_id = dbmod.add_submission(
            test_id=test_id,
            answers=answers,
            score=report,
            student_id=student_id,
            student_name=student_name,
            student_email=student_email,
        )
        click.echo(f"\nSaved as submission #{sub_id}.")


@cli.command("grade-classroom")
@click.argument("email")
@click.argument("course_id")
@click.argument("cw_id")
@click.option("--test", "test_id", required=True, help="Saved test id (from `test list`).")
@click.option("--template", "template_path",
              default="data/sheets/act_sheet.template.json", show_default=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--all", "fetch_all", is_flag=True,
              help="Also include submissions still in NEW/CREATED states (default: "
                   "TURNED_IN + RETURNED only).")
@click.option("--no-refetch", is_flag=True,
              help="Use the cached manifest instead of re-downloading from Classroom.")
def cmd_grade_classroom(
    email: str,
    course_id: str,
    cw_id: str,
    test_id: str,
    template_path: str,
    fetch_all: bool,
    no_refetch: bool,
) -> None:
    """Fetch every submission for a Classroom assignment, read each scan, grade
    against `--test`, and save a submission row per student."""
    out = grade_classroom_assignment(
        email,
        course_id,
        cw_id,
        test_id=test_id,
        template_path=template_path,
        only_turned_in=not fetch_all,
        refetch=not no_refetch,
    )

    results = out["results"]
    if not results:
        click.echo("(no students found for this assignment)")
        return

    # Summary first, then per-student detail.
    graded = [r for r in results if r["status"] == "graded"]
    failed = [r for r in results if r["status"] != "graded"]
    click.echo(f"Test:       {test_id}")
    click.echo(f"Assignment: {course_id} / {cw_id}")
    click.echo(f"Graded:     {len(graded)} / {len(results)} students")
    if graded:
        avg = sum(r["composite"] for r in graded if r["composite"] is not None) / len(graded)
        comps = [r["composite"] for r in graded if r["composite"] is not None]
        click.echo(f"Composites: min={min(comps)} mean={avg:.1f} max={max(comps)}")
    click.echo()

    click.echo(f"{'Composite':<10} {'Student':<30} {'Email':<30} {'Status':<14}")
    click.echo("-" * 90)
    for r in results:
        comp = r.get("composite")
        comp_s = str(comp) if comp is not None else "-"
        name = (r.get("name") or "-")[:29]
        emailv = (r.get("email") or "-")[:29]
        status = r["status"]
        click.echo(f"{comp_s:<10} {name:<30} {emailv:<30} {status:<14}")
        if r["status"] != "graded" and r.get("error"):
            click.echo(f"           └─ {r['error']}")


@cli.group("assignment")
def assignment_group() -> None:
    """Create / inspect Classroom assignments owned by this app.

    Important: Classroom restricts grade writes to the OAuth project that
    *created* the coursework. To grade via API, the assignment must be
    created with `assignment create` (not in the Classroom web UI).
    """


@assignment_group.command("create")
@click.argument("email")
@click.argument("course_id")
@click.option("--title", required=True, help="Visible assignment title.")
@click.option("--description", default=None, help="Optional description shown to students.")
@click.option("--max-points", type=float, default=36, show_default=True,
              help="Set to 36 for natural composite display.")
@click.option("--draft", is_flag=True,
              help="Create as DRAFT (teacher-only) instead of PUBLISHED.")
def cmd_assignment_create(
    email: str, course_id: str, title: str,
    description: str | None, max_points: float, draft: bool,
) -> None:
    """Create a new assignment in a Classroom course."""
    cw = create_coursework(
        email, course_id, title,
        description=description, max_points=max_points, draft=draft,
    )
    click.echo(f"Created coursework: {cw['id']}")
    click.echo(f"  title:      {cw.get('title')}")
    click.echo(f"  state:      {cw.get('state')}")
    click.echo(f"  maxPoints:  {cw.get('maxPoints')}")
    click.echo(f"  url:        {cw.get('alternateLink')}")
    click.echo()
    click.echo("Share the assignment URL with students. After they turn in, grade with:")
    click.echo(f"  bubble-grader grade-classroom {email} {course_id} {cw['id']} --test <test_id>")


@cli.command("send-feedback")
@click.argument("email")
@click.argument("course_id")
@click.argument("cw_id")
@click.option("--only", "only_students", multiple=True,
              help="Limit to specific students (Classroom userId or email). Repeatable.")
@click.option("--teacher-name", default=None, help="Signs the email (e.g. 'Ms. Smith').")
@click.option("--test-name", default=None,
              help="Override the test label that appears in the subject + body.")
@click.option("--no-overlay", is_flag=True,
              help="Don't attach the green-circle / red-X overlay PDF.")
@click.option("--dry-run", is_flag=True, help="Print what would be sent instead of sending.")
def cmd_send_feedback(
    email: str, course_id: str, cw_id: str,
    only_students: tuple[str, ...], teacher_name: str | None,
    test_name: str | None, no_overlay: bool, dry_run: bool,
) -> None:
    """Email each graded student a per-section + missed-questions report from your Gmail."""
    out = send_feedback_for_assignment(
        email, course_id, cw_id,
        test_name=test_name,
        only_students=list(only_students) or None,
        teacher_name=teacher_name,
        include_overlay=not no_overlay,
        dry_run=dry_run,
    )
    rows = out["results"]
    if not rows:
        click.echo("(no graded students match)")
        return
    click.echo(f"{'Status':<14} {'Attach':<7} {'Student':<26} {'Email':<32}")
    click.echo("-" * 85)
    for r in rows:
        attach = "yes" if r.get("attached_overlay") else "no"
        click.echo(f"{r['status']:<14} {attach:<7} {(r.get('name') or '-')[:25]:<26} {(r.get('email') or '-')[:31]:<32}")
        if r.get("error"):
            click.echo(f"       └─ {r['error']}")
        if r.get("overlay_error"):
            click.echo(f"       └─ overlay skipped: {r['overlay_error']}")
        if dry_run and r.get("body"):
            click.echo("       --- BODY ---")
            for line in r["body"].splitlines():
                click.echo(f"       {line}")
            click.echo("       ------------")


@cli.command("release-grades")
@click.argument("email")
@click.argument("course_id")
@click.argument("cw_id")
@click.option("--scale-to", type=float, default=None,
              help="If set, scale composite (1-36) to this max-points value.")
@click.option("--return", "return_to_student", is_flag=True,
              help="Also 'return' each graded submission so the student sees it.")
@click.option("--draft-only", is_flag=True,
              help="Set only the draft grade (teacher visibility) — do not assign.")
@click.option("--only", "only_students", multiple=True,
              help="Limit release to specific students (Classroom userId or email). "
                   "Pass --only repeatedly for multiple students.")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
def cmd_release_grades(
    email: str,
    course_id: str,
    cw_id: str,
    scale_to: float | None,
    return_to_student: bool,
    draft_only: bool,
    only_students: tuple[str, ...],
    yes: bool,
) -> None:
    """Push every saved grade for an assignment back into Google Classroom."""
    if return_to_student and draft_only:
        raise click.ClickException("--return and --draft-only are mutually exclusive.")

    if not yes:
        action = "draft-set"
        if not draft_only and not return_to_student:
            action = "assigned (teacher-visible)"
        elif return_to_student:
            action = "RETURNED to students (they'll see scores)"
        target = f" for {len(only_students)} selected student(s)" if only_students else " for everyone with a graded submission"
        click.confirm(
            f"This will update grades in Classroom — action: {action}{target}. Proceed?",
            abort=True,
        )

    out = release_grades(
        email, course_id, cw_id,
        scale_to=scale_to,
        return_to_student=return_to_student,
        draft_only=draft_only,
        only_students=list(only_students) or None,
    )

    if out.get("max_points") is not None:
        click.echo(f"Classroom assignment maxPoints = {out['max_points']}")
    if scale_to is not None:
        click.echo(f"Composite (0..36) scaled to (0..{scale_to}).")
    else:
        click.echo("Sending composite as-is (set Classroom maxPoints=36 for natural display).")
    click.echo()

    rows = out["results"]
    click.echo(f"{'Comp':<5} {'Sent':<8} {'Student':<28} {'Status':<20}")
    click.echo("-" * 70)
    for r in rows:
        comp = str(r.get("composite") or "-")
        sent = f"{r['sent']:.1f}" if r.get("sent") is not None else "-"
        name = (r.get("name") or "-")[:27]
        click.echo(f"{comp:<5} {sent:<8} {name:<28} {r['status']:<20}")
        if r.get("error"):
            click.echo(f"       └─ {r['error']}")


@cli.group("submissions")
def submissions_group() -> None:
    """List and inspect graded submissions."""


@submissions_group.command("list")
@click.option("--test", "test_id", default=None)
@click.option("--student", "student_id", default=None)
def cmd_submissions_list(test_id: str | None, student_id: str | None) -> None:
    rows = dbmod.list_submissions(test_id=test_id, student_id=student_id)
    if not rows:
        click.echo("(no submissions match)")
        return
    click.echo(f"{'ID':<5} {'Test':<25} {'Student':<25} {'Comp':<5} {'When':<19}")
    click.echo("-" * 80)
    for r in rows:
        ident = r["student_name"] or r["student_email"] or r["student_id"] or "-"
        comp = str(r["composite"]) if r["composite"] is not None else "-"
        click.echo(f"{r['id']:<5} {r['test_id']:<25} {ident[:24]:<25} {comp:<5} {r['created_at']}")


@submissions_group.command("show")
@click.argument("submission_id", type=int)
def cmd_submissions_show(submission_id: int) -> None:
    sub = dbmod.get_submission(submission_id)
    if sub is None:
        raise click.ClickException(f"No submission #{submission_id}.")
    click.echo(json.dumps(sub, indent=2))


if __name__ == "__main__":
    cli()
