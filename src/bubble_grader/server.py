"""FastAPI app: OAuth + JSON API + server-rendered web UI for teachers."""

import base64
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.middleware.sessions import SessionMiddleware

from . import db as dbmod
from .classroom import (
    create_coursework,
    delete_coursework,
    list_courses,
    list_coursework,
    list_roster,
    list_submissions,
)
from .config import (
    ALLOWED_TEACHERS,
    AUTO_GRADE_ON_TURNIN,
    AUTO_GRADE_TOKEN,
    FERNET_KEY,
    IS_VERCEL,
    SERVER_PORT,
)
from .google_auth import (
    authorization_url,
    credentials_to_dict,
    email_from_credentials,
    exchange_code,
)
from .feedback import send_feedback_for_assignment
from .scoring import full_grade
from .submissions import (
    fetch_assignment,
    grade_classroom_assignment,
    release_grades,
)
from .version_check import snapshot as update_snapshot, start_background_poller


@asynccontextmanager
async def lifespan(_app: FastAPI):
    dbmod.init_db()
    if not IS_VERCEL:
        # Background threads make no sense on serverless — requests are the
        # only execution there. Locally they behave exactly as before.
        start_background_poller()
        if AUTO_GRADE_ON_TURNIN:
            from .auto_grade import start_auto_grade_poller
            start_auto_grade_poller()
    yield


app = FastAPI(title="Grader Form v2", lifespan=lifespan)

# Cookie session secret derived from FERNET_KEY (already stored in .env).
# Sessions hold {"email": "...", "flash": {...}} between requests.
if not FERNET_KEY:
    raise RuntimeError(
        "FERNET_KEY missing. Run `uv run bubble-grader setup` first."
    )
_session_secret = base64.urlsafe_b64encode(FERNET_KEY).decode()
app.add_middleware(SessionMiddleware, secret_key=_session_secret, session_cookie="bg_session")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico():
    # Browsers request this path unprompted; serve the multi-size ICO.
    return FileResponse(STATIC_DIR / "favicon.ico", media_type="image/x-icon")

# Build the Jinja Environment ourselves so we can disable the LRUCache.
# On Python 3.14, Jinja 3.1.x's LRUCache hits a "tuple is not hashable" path;
# we don't need template caching at this scale and skipping it sidesteps it.
_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    cache_size=0,
)
templates = Jinja2Templates(env=_jinja_env)


# ----- session helpers ------------------------------------------------------

def _email(request: Request) -> str | None:
    return request.session.get("email")


def _flash(request: Request, kind: str, msg: str) -> None:
    request.session["flash"] = {"kind": kind, "msg": msg}


def _pop_flash(request: Request) -> dict | None:
    return request.session.pop("flash", None)


def _is_allowed(email: str | None) -> bool:
    """Sign-in allowlist. An empty ALLOWED_TEACHERS means open (local use)."""
    return bool(email) and (not ALLOWED_TEACHERS or email.lower() in ALLOWED_TEACHERS)


def _require(request: Request) -> str | RedirectResponse:
    """Return the signed-in email, or a redirect to '/' if not signed in."""
    email = _email(request)
    if not email:
        return RedirectResponse("/", status_code=303)
    if not _is_allowed(email):
        # Covers sessions minted before the allowlist existed (or was edited).
        request.session.clear()
        return RedirectResponse("/", status_code=303)
    return email


def _render(request: Request, name: str, **ctx: Any):
    # Modern Starlette signature: (request, name, context). The Request is also
    # auto-injected into the template context as `request` so templates can
    # access it directly.
    update = update_snapshot()
    return templates.TemplateResponse(
        request,
        name,
        {
            "signed_in_as": _email(request),
            "flash": _pop_flash(request),
            "update_available": update.available and not IS_VERCEL,
            "update_command": "uv run bubble-grader update",
            **ctx,
        },
    )


# ----- OAuth + session lifecycle --------------------------------------------

@app.get("/oauth/start")
def oauth_start():
    url, state, code_verifier = authorization_url()
    dbmod.save_state(state, code_verifier)
    return RedirectResponse(url)


@app.get("/oauth/callback")
def oauth_callback(request: Request):
    params = dict(request.query_params)
    if "error" in params:
        raise HTTPException(400, f"Google returned error: {params['error']}")
    code = params.get("code")
    state = params.get("state")
    if not code or not state:
        raise HTTPException(400, "missing code or state")
    code_verifier = dbmod.consume_state(state)
    if not code_verifier:
        raise HTTPException(400, "unknown or replayed state")

    creds = exchange_code(code=code, state=state, code_verifier=code_verifier)
    if not creds.refresh_token:
        raise HTTPException(
            500,
            "No refresh_token returned. Revoke app access in your Google account and try again.",
        )
    email = email_from_credentials(creds)
    if not _is_allowed(email):
        # Turn strangers away at the door; never store their credentials.
        _flash(request, "bad",
               f"{email} isn't authorized to use this grader. "
               "Ask the admin to add you to the teacher list.")
        return RedirectResponse("/", status_code=303)
    dbmod.store_credentials(email, credentials_to_dict(creds))
    from .google_api import forget_credentials
    forget_credentials(email)
    request.session["email"] = email
    _flash(request, "ok", f"Signed in as {email}.")
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/signout")
def signout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


# ----- public landing -------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _render(request, "index.html")


# ----- dashboard ------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    courses = list_courses(email)
    return _render(request, "dashboard.html", courses=courses)


# ----- course view (list assignments) ---------------------------------------

@app.get("/courses/{course_id}", response_class=HTMLResponse)
def course_view(request: Request, course_id: str):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    courses = list_courses(email)
    course = next((c for c in courses if c["id"] == course_id), None)
    if course is None:
        raise HTTPException(404, "Course not found in your active courses.")
    coursework = list_coursework(email, course_id)
    owned_ids = {a["coursework_id"] for a in dbmod.list_app_assignments(course_id)}
    for cw in coursework:
        cw["is_app_owned"] = cw["id"] in owned_ids
    return _render(request, "course.html", course=course, coursework=coursework)


# ----- roster view ----------------------------------------------------------

def _resolve_course(email: str, course_id: str) -> dict:
    """Lookup-or-404 helper used by every course-scoped route."""
    course = next((c for c in list_courses(email) if c["id"] == course_id), None)
    if course is None:
        raise HTTPException(404, "Course not found in your active courses.")
    return course


def _normalize_roster(roster_raw: list[dict]) -> list[dict]:
    """Flatten the Classroom roster API shape into {id, name, email}."""
    out = []
    for r in roster_raw:
        prof = r.get("profile") or {}
        sid = prof.get("id") or r.get("userId")
        name = (prof.get("name") or {}).get("fullName") or prof.get("emailAddress") or "?"
        out.append({
            "id": sid,
            "name": name,
            "email": prof.get("emailAddress"),
        })
    out.sort(key=lambda s: s["name"].lower())
    return out


@app.get("/courses/{course_id}/roster", response_class=HTMLResponse)
def course_roster(request: Request, course_id: str):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    course = _resolve_course(email, course_id)
    roster = _normalize_roster(list_roster(email, course_id))
    subs = dbmod.list_submissions(course_id=course_id)
    # Map student → most recent submission (list is already ORDER BY created_at DESC).
    latest_by_student: dict[str, dict] = {}
    for s in subs:
        sid = s.get("student_id")
        if sid and sid not in latest_by_student:
            latest_by_student[sid] = s
    # Map test_id → familiar test name so the roster shows "Alex Atala
    # (New Format)" rather than the raw id/slug.
    test_names = {t["id"]: t.get("name") for t in dbmod.list_tests()}
    for student in roster:
        latest = latest_by_student.get(student["id"])
        if latest is not None:
            latest = {
                **latest,
                "test_name": test_names.get(latest.get("test_id")) or latest.get("test_id"),
            }
        student["latest"] = latest
    return _render(
        request, "roster.html",
        course=course, roster=roster,
    )


def _as_dt(ts: Any) -> datetime | None:
    """Coerce a submission timestamp (datetime or ISO string) to a datetime."""
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None


def _due_datetime(due_date: dict | None, due_time: dict | None) -> datetime | None:
    """Build a UTC datetime from Classroom's split dueDate/dueTime fields.

    ``dueDate`` is ``{year, month, day}``; ``dueTime`` is ``{hours, minutes,...}``
    in UTC and may be absent (then we anchor to the start of that day).
    """
    if not due_date:
        return None
    try:
        t = due_time or {}
        return datetime(
            int(due_date["year"]), int(due_date["month"]), int(due_date["day"]),
            int(t.get("hours", 0)), int(t.get("minutes", 0)),
            tzinfo=timezone.utc,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _coursework_due_map(email: str, course_id: str) -> dict[str, datetime]:
    """coursework_id → due datetime (UTC), best-effort. Empty on any API error."""
    try:
        cws = list_coursework(email, course_id)
    except Exception:  # noqa: BLE001 — degrade gracefully to "no due dates"
        return {}
    out: dict[str, datetime] = {}
    for cw in cws:
        due = _due_datetime(cw.get("dueDate"), cw.get("dueTime"))
        if due is not None:
            out[cw["id"]] = due
    return out


def _pick_cluster_winner(cluster: list[dict], due_by_cw: dict[str, datetime]) -> dict:
    """Choose which submission represents a cluster of re-grades.

    ``cluster`` is newest-first. Selection rules:

    1. A submission carrying **manual overrides** is sticky — keep the newest
       such one, so a later automatic re-grade can't silently wipe a teacher's
       correction. (Overrides patch a row in place, so they live on whichever
       submission has a non-empty ``score['overrides']``.)
    2. Otherwise keep the grade whose timestamp is **closest to the due date**.
    3. If no due date is known, fall back to the newest grade.
    """
    overrides = [s for s in cluster if (s.get("score") or {}).get("overrides")]
    if overrides:
        return overrides[0]  # newest-first → newest override

    best, best_dist = None, None
    for s in cluster:
        due = due_by_cw.get(s.get("coursework_id"))
        when = _as_dt(s.get("created_at"))
        if due is None or when is None:
            continue
        dist = abs((when - due).total_seconds())
        if best_dist is None or dist < best_dist:
            best, best_dist = s, dist
    return best if best is not None else cluster[0]


def _dedupe_regrades(
    subs: list[dict], due_by_cw: dict[str, datetime], window_days: int = 7
) -> list[dict]:
    """Collapse re-grades of the same assignment that happen close together.

    Submissions for the same (coursework, test) within ``window_days`` of each
    other are one cluster, represented by a single row chosen via
    :func:`_pick_cluster_winner`. Re-grades more than a week apart are treated
    as separate attempts and each is kept. Filtering is display-only — no rows
    are deleted from the database.

    ``subs`` must be newest-first (as ``list_submissions`` returns).
    """
    window = timedelta(days=window_days)
    groups: dict[tuple, list[dict]] = {}
    for s in subs:  # newest-first
        key = (s.get("coursework_id"), s.get("test_id"))
        groups.setdefault(key, []).append(s)

    keep_ids: set = set()
    for group in groups.values():
        cluster: list[dict] = []
        anchor: datetime | None = None
        for s in group:  # newest-first within the group
            when = _as_dt(s.get("created_at"))
            if anchor is not None and when is not None and (anchor - when) <= window:
                cluster.append(s)
            else:
                if cluster:
                    keep_ids.add(_pick_cluster_winner(cluster, due_by_cw)["id"])
                cluster, anchor = [s], when
        if cluster:
            keep_ids.add(_pick_cluster_winner(cluster, due_by_cw)["id"])
    return [s for s in subs if s["id"] in keep_ids]


@app.get(
    "/courses/{course_id}/students/{student_id}",
    response_class=HTMLResponse,
)
def course_student_detail(request: Request, course_id: str, student_id: str):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    course = _resolve_course(email, course_id)
    # Find the student in the roster (for the name / email at the top).
    roster = _normalize_roster(list_roster(email, course_id))
    student = next((s for s in roster if s["id"] == student_id), None)
    if student is None:
        raise HTTPException(404, "Student not found in this course's roster.")

    subs = dbmod.list_submissions(
        course_id=course_id, student_id=student_id, include_score=True,
    )
    # A test re-graded several times in the same week shows once — the grade
    # closest to the due date, unless a re-grade carries a manual override
    # (which stays sticky). Keeps the table and averages from being skewed.
    due_by_cw = _coursework_due_map(email, course_id)
    subs = _dedupe_regrades(subs, due_by_cw)

    # For each submission: pull per-section raw + scaled out of `score`.
    # The full_grade output is { "sections": {"Test 1": {raw_score, scaled, ...}, ...},
    #                            "composite": int }.
    section_names = ["Test 1", "Test 2", "Test 3", "Test 4"]
    test_names = {t["id"]: t.get("name") for t in dbmod.list_tests()}
    rows = []
    for s in subs:
        sc = s.get("score") or {}
        secs = sc.get("sections") or {}
        per_section = {}
        for sec in section_names:
            d = secs.get(sec) or {}
            per_section[sec] = {
                "raw": d.get("raw_score"),
                "scaled": d.get("scaled"),
            }
        rows.append({
            "id": s["id"],
            "test_id": s["test_id"],
            "test_name": test_names.get(s["test_id"]) or s["test_id"],
            "created_at": s["created_at"],
            "composite": s.get("composite"),
            "coursework_id": s.get("coursework_id"),
            "sections": per_section,
        })

    # Averages — across submissions, per section, for raw and scaled.
    def _avg(values: list[float | int | None]) -> float | None:
        vals = [v for v in values if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 1) if vals else None

    averages = {"composite": _avg([r["composite"] for r in rows])}
    for sec in section_names:
        averages[sec] = {
            "raw": _avg([r["sections"][sec]["raw"] for r in rows]),
            "scaled": _avg([r["sections"][sec]["scaled"] for r in rows]),
        }

    return _render(
        request, "student_course_detail.html",
        course=course, student=student, rows=rows,
        averages=averages, section_names=section_names,
    )


# ----- create assignment ----------------------------------------------------

@app.get("/courses/{course_id}/new", response_class=HTMLResponse)
def new_assignment_form(request: Request, course_id: str):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    courses = list_courses(email)
    course = next((c for c in courses if c["id"] == course_id), None)
    if course is None:
        raise HTTPException(404, "Course not found.")
    tests = dbmod.list_tests()
    # Pre-compute the passage map: { format: { section: [{q_start, q_end, label}, ...] } }
    # so the new-assignment form can render the Passage dropdown without
    # extra round-trips when the teacher switches tests.
    from .passages import PASSAGES_BY_FORMAT_AND_SECTION
    passage_map = {
        fmt: {
            sec: [{"q_start": s, "q_end": e, "label": label} for (s, e, label) in psgs]
            for sec, psgs in by_section.items()
        }
        for fmt, by_section in PASSAGES_BY_FORMAT_AND_SECTION.items()
    }
    return _render(
        request, "new_assignment.html",
        course=course, tests=tests, passage_map_json=json.dumps(passage_map),
    )


def _parse_scope(
    scope_type: str,
    section: str,
    q_start: str,
    q_end: str,
    label: str,
) -> dict | None:
    """Translate the new-assignment form's scope fields into a scope dict.

    Returns ``None`` (= full scope) when scope_type is missing or "full".
    Returns ``{"type": "partial", ...}`` when the partial fields validate.
    Raises HTTPException(400) when the partial config is malformed so the
    teacher gets a clear error instead of a silently-stored bad row.
    """
    if scope_type != "partial":
        return None
    try:
        qs = int(q_start)
        qe = int(q_end)
    except ValueError:
        raise HTTPException(400, "Question range must be numeric (e.g. 11 to 20).")
    if qs < 1 or qe < qs:
        raise HTTPException(400, "Question range is invalid: end must be >= start, both >= 1.")
    if section not in {"Test 1", "Test 2", "Test 3", "Test 4"}:
        raise HTTPException(400, "Section must be one of Test 1–4.")
    return {
        "type": "partial",
        "section": section,
        "q_start": qs,
        "q_end": qe,
        "label": (label or "").strip() or None,
    }


@app.post("/courses/{course_id}/new")
def new_assignment_submit(
    request: Request,
    course_id: str,
    title: str = Form(...),
    description: str = Form(""),
    test_id: str = Form(""),
    max_points: float = Form(36),
    draft: str = Form(""),
    scope_type: str = Form("full"),
    scope_section: str = Form(""),
    scope_q_start: str = Form(""),
    scope_q_end: str = Form(""),
    scope_label: str = Form(""),
):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    scope = _parse_scope(scope_type, scope_section, scope_q_start, scope_q_end, scope_label)
    cw = create_coursework(
        email, course_id, title,
        description=description or None,
        max_points=max_points,
        draft=bool(draft),
    )
    dbmod.record_app_assignment(
        course_id, cw["id"],
        test_id=test_id or None,
        title=title,
        created_by=email,
        scope=scope,
    )
    _flash(request, "ok", f"Created assignment '{title}'.")
    return RedirectResponse(
        f"/courses/{course_id}/coursework/{cw['id']}", status_code=303
    )


# ----- assignment detail ----------------------------------------------------

def _section_scaled_summary(score: dict | None) -> str:
    if not score:
        return ""
    sps = score.get("scaled_per_section") or {}
    if not sps:
        return ""
    order = ["Test 1", "Test 2", "Test 3", "Test 4"]
    return " / ".join(str(sps.get(s, "—")) for s in order)


@app.get("/courses/{course_id}/coursework/{cw_id}", response_class=HTMLResponse)
def assignment_view(request: Request, course_id: str, cw_id: str, test: str | None = None):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    # Four independent Google calls — run them concurrently so the page pays
    # one round trip (~0.4 s) instead of four in a row.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_courses = pool.submit(list_courses, email)
        f_coursework = pool.submit(list_coursework, email, course_id)
        f_cls_subs = pool.submit(list_submissions, email, course_id, cw_id)
        f_roster = pool.submit(list_roster, email, course_id)
        courses = f_courses.result()
        coursework = f_coursework.result()
        try:
            cls_subs = f_cls_subs.result()
        except Exception:  # noqa: BLE001 — state column is best-effort
            cls_subs = []
        roster = f_roster.result()

    course = next((c for c in courses if c["id"] == course_id), None)
    if course is None:
        raise HTTPException(404, "Course not found.")
    cw = next((x for x in coursework if x["id"] == cw_id), None)
    if cw is None:
        raise HTTPException(404, "Assignment not found.")
    owned = dbmod.get_app_assignment(course_id, cw_id)
    cw["is_app_owned"] = owned is not None

    # Pick which test to grade against: query param > stored > none.
    assigned_test_id = test or (owned.get("test_id") if owned else None)

    # Classroom-side submission state, keyed by student userId.
    cls_state: dict[str, str] = {
        s["userId"]: s.get("state", "?") for s in cls_subs if s.get("userId")
    }

    # Roster + locally graded submissions.
    # `list_submissions` returns rows ORDER BY created_at DESC, so the
    # newest row for each student appears FIRST. We want to keep that
    # first occurrence; a dict comprehension would silently overwrite
    # with the oldest. `setdefault` is the right tool here.
    graded: dict[str, dict] = {}
    for r in dbmod.list_submissions(
        course_id=course_id, coursework_id=cw_id, include_score=True,
    ):
        sid = r.get("student_id")
        if sid:
            graded.setdefault(sid, r)

    grade_errs = dbmod.list_grade_errors(course_id, cw_id)

    rows = []
    for student in roster:
        sid = student["userId"]
        profile = student.get("profile") or {}
        name = (profile.get("name") or {}).get("fullName")
        emailv = profile.get("emailAddress")
        g = graded.get(sid)
        score = g.get("score") if g else None
        partial = (score or {}).get("partial") if score else None
        rows.append({
            "student_id": sid,
            "name": name,
            "email": emailv,
            "classroom_state": cls_state.get(sid, "—"),
            "grade_error": grade_errs.get(sid),
            "partial": partial,
            "composite": g.get("composite") if g else None,
            "section_scaled": _section_scaled_summary(score),
            "graded_at": g.get("created_at") if g else None,
        })

    tests = dbmod.list_tests()
    return _render(
        request, "assignment.html",
        course=course, cw=cw,
        tests=tests, assigned_test_id=assigned_test_id,
        rows=rows,
        scope=(owned.get("scope") if owned else None),
    )


# ----- grade + release actions ----------------------------------------------

@app.post("/courses/{course_id}/coursework/{cw_id}/grade")
def assignment_grade(
    request: Request,
    course_id: str,
    cw_id: str,
    test_id: str = Form(""),
):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    # Fall back to stored test_id if the form didn't carry one (e.g. someone
    # submitted via curl). Normal browser flow always sends test_id from the
    # required <select>.
    if not test_id:
        owned = dbmod.get_app_assignment(course_id, cw_id)
        test_id = (owned or {}).get("test_id") or ""
    if not test_id:
        _flash(request, "bad", "No test selected to grade against. Pick one with the dropdown.")
        return RedirectResponse(f"/courses/{course_id}/coursework/{cw_id}", status_code=303)

    # Remember the choice on the assignment so it defaults next visit.
    owned = dbmod.get_app_assignment(course_id, cw_id)
    if owned is not None:
        dbmod.record_app_assignment(
            course_id, cw_id,
            test_id=test_id,
            title=owned.get("title"),
        )

    # Pick the OMR template matching the test's format. Legacy 75/60/40/40
    # tests use act_sheet.*; new-format 50/45/36/40 tests use act_sheet_new.*.
    from .submissions import template_for_test
    tpl_path, _ref_path = template_for_test(test_id)
    try:
        result = grade_classroom_assignment(
            email, course_id, cw_id,
            test_id=test_id,
            template_path=str(tpl_path),
        )
    except Exception as e:  # noqa: BLE001
        _flash(request, "bad", f"Grading failed: {type(e).__name__}: {e}")
        return RedirectResponse(f"/courses/{course_id}/coursework/{cw_id}", status_code=303)

    graded = sum(1 for r in result["results"] if r["status"] == "graded")
    total = len(result["results"])
    failed = total - graded
    msg = f"Graded {graded}/{total} students."
    if failed:
        msg += f" {failed} failed — see submission rows."
    _flash(request, "ok" if not failed else "warn", msg)
    return RedirectResponse(
        f"/courses/{course_id}/coursework/{cw_id}", status_code=303
    )


# ----- chunked grade + feedback (one student per request) -------------------
#
# Serverless hosting caps how long one request may run, so the browser drives
# the loop instead: /grade/start returns the student list, then the page calls
# /grade/student once per student with a progress bar. Feedback emails work the
# same way, with /feedback/complete ending the cycle (scan cleanup).

def _require_json(request: Request):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return JSONResponse({"error": "not signed in"}, status_code=401)
    return email


@app.post("/courses/{course_id}/coursework/{cw_id}/grade/start")
def assignment_grade_start(
    request: Request, course_id: str, cw_id: str, test_id: str = Form(""),
):
    email = _require_json(request)
    if isinstance(email, JSONResponse):
        return email
    if not test_id:
        owned = dbmod.get_app_assignment(course_id, cw_id)
        test_id = (owned or {}).get("test_id") or ""
    if not test_id:
        return JSONResponse({"error": "No test selected to grade against."}, status_code=400)
    if dbmod.get_test(test_id) is None:
        return JSONResponse({"error": f"Unknown test {test_id!r}."}, status_code=400)

    # Remember the choice on the assignment so it defaults next visit.
    owned = dbmod.get_app_assignment(course_id, cw_id)
    if owned is not None:
        dbmod.record_app_assignment(
            course_id, cw_id, test_id=test_id, title=owned.get("title"),
        )

    try:
        roster = {s["userId"]: s for s in list_roster(email, course_id)}
        subs = list_submissions(email, course_id, cw_id)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)

    students = []
    for s in subs:
        if s.get("state") not in ("TURNED_IN", "RETURNED"):
            continue
        sid = s.get("userId")
        profile = (roster.get(sid) or {}).get("profile") or {}
        students.append({
            "student_id": sid,
            "name": (profile.get("name") or {}).get("fullName") or sid,
        })
    return {"test_id": test_id, "students": students}


@app.post("/courses/{course_id}/coursework/{cw_id}/grade/student")
def assignment_grade_student(
    request: Request, course_id: str, cw_id: str,
    student_id: str = Form(...), test_id: str = Form(""),
):
    email = _require_json(request)
    if isinstance(email, JSONResponse):
        return email
    if not test_id:
        owned = dbmod.get_app_assignment(course_id, cw_id)
        test_id = (owned or {}).get("test_id") or ""
    if not test_id:
        return JSONResponse({"error": "No test selected."}, status_code=400)

    from .submissions import template_for_test
    tpl_path, _ref = template_for_test(test_id)
    try:
        result = grade_classroom_assignment(
            email, course_id, cw_id,
            test_id=test_id, template_path=str(tpl_path),
            only_students=[student_id],
        )
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    r = next(iter(result["results"]), None) or {"status": "no_classroom_submission"}
    if r.get("status") != "graded":
        # Surfaces in the hosting platform's function logs.
        print(f"[grade-failed] cw={cw_id} student={student_id}: {r.get('status')} — {r.get('error')}")
    return {
        "student_id": student_id,
        "status": r.get("status"),
        "composite": r.get("composite"),
        "error": r.get("error"),
    }


@app.post("/courses/{course_id}/coursework/{cw_id}/feedback/student")
def assignment_feedback_student(
    request: Request, course_id: str, cw_id: str,
    student_id: str = Form(...), teacher_name: str = Form(""),
):
    email = _require_json(request)
    if isinstance(email, JSONResponse):
        return email
    owned = dbmod.get_app_assignment(course_id, cw_id) or {}
    test_obj = dbmod.get_test(owned.get("test_id")) if owned.get("test_id") else None
    try:
        result = send_feedback_for_assignment(
            email, course_id, cw_id,
            test_name=(test_obj or {}).get("name"),
            only_students=[student_id],
            teacher_name=teacher_name.strip() or None,
            delete_scans=False,  # the cycle ends via /feedback/complete
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    r = next(iter(result["results"]), None) or {"status": "no_graded_submission"}
    return {
        "student_id": student_id,
        "status": r.get("status"),
        "attached_overlay": bool(r.get("attached_overlay")),
        "error": r.get("error") or r.get("overlay_error"),
    }


@app.post("/courses/{course_id}/coursework/{cw_id}/feedback/complete")
def assignment_feedback_complete(request: Request, course_id: str, cw_id: str):
    """End of a grading cycle: drop the assignment's stored scans."""
    email = _require_json(request)
    if isinstance(email, JSONResponse):
        return email
    return {"scans_deleted": dbmod.delete_assignment_scans(course_id, cw_id)}


@app.get("/healthz")
def healthz():
    """Ultra-cheap liveness endpoint for keep-warm pingers. No auth, no DB."""
    return {"ok": True}


@app.get("/cron/auto-grade")
def cron_auto_grade(token: str = ""):
    """Grade everything newly turned in — trigger for an external pinger.

    Serverless hosts can't run background threads, so an external pinger
    (e.g. UptimeRobot every 5 minutes) hits this instead; it also keeps the
    function warm. Bounded so one run always fits the request time limit;
    leftovers are picked up by the next ping. Requires AUTO_GRADE_TOKEN.
    """
    if not AUTO_GRADE_TOKEN or token != AUTO_GRADE_TOKEN:
        raise HTTPException(404, "Not found.")
    from .auto_grade import run_auto_grade_once
    summary = run_auto_grade_once(max_students=10, time_budget=210)
    if summary["students_graded"] or summary["students_failed"]:
        print(f"[auto-grade] {summary}")
    return summary


@app.post("/courses/{course_id}/coursework/{cw_id}/release")
async def assignment_release(
    request: Request,
    course_id: str,
    cw_id: str,
):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    # Read action + selected student_ids from the multi-valued form body.
    form = await request.form()
    action = form.get("action", "draft")
    selected = [v for v in form.getlist("student_ids") if isinstance(v, str) and v]

    if not selected:
        _flash(request, "warn", "No students selected — tick at least one row to release.")
        return RedirectResponse(f"/courses/{course_id}/coursework/{cw_id}", status_code=303)

    draft_only = action == "draft"
    return_to_student = action == "release"
    try:
        result = release_grades(
            email, course_id, cw_id,
            return_to_student=return_to_student,
            draft_only=draft_only,
            only_students=selected,
        )
    except Exception as e:  # noqa: BLE001
        _flash(request, "bad", f"Release failed: {type(e).__name__}: {e}")
        return RedirectResponse(f"/courses/{course_id}/coursework/{cw_id}", status_code=303)

    ok = sum(1 for r in result["results"] if r["status"] in ("draft_set", "grade_assigned", "returned"))
    bad = [r for r in result["results"] if r["status"] not in ("draft_set", "grade_assigned", "returned")]
    if bad:
        first_err = next((r.get("error") for r in bad if r.get("error")), None)
        _flash(request, "warn",
               f"Released {ok}/{len(selected)} selected; {len(bad)} failed. First error: {first_err}")
    else:
        verb = "released" if return_to_student else "set to draft"
        _flash(request, "ok", f"{ok} grade(s) {verb} for selected students.")
    return RedirectResponse(f"/courses/{course_id}/coursework/{cw_id}", status_code=303)


@app.post("/courses/{course_id}/coursework/{cw_id}/delete")
def assignment_delete(request: Request, course_id: str, cw_id: str):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email

    # Only assignments our app created can be deleted via the Classroom API.
    if dbmod.get_app_assignment(course_id, cw_id) is None:
        _flash(
            request, "bad",
            "This assignment wasn't created by Grader Form, so Classroom's API "
            "won't let us delete it. Delete it from the Classroom web UI directly.",
        )
        return RedirectResponse(f"/courses/{course_id}/coursework/{cw_id}", status_code=303)

    try:
        delete_coursework(email, course_id, cw_id)
    except Exception as e:  # noqa: BLE001
        _flash(request, "bad", f"Classroom delete failed: {type(e).__name__}: {e}")
        return RedirectResponse(f"/courses/{course_id}/coursework/{cw_id}", status_code=303)

    n = dbmod.delete_app_assignment(course_id, cw_id)
    _flash(request, "ok", f"Assignment deleted (and {n} local row(s) cleaned up).")
    return RedirectResponse(f"/courses/{course_id}", status_code=303)


@app.post("/courses/{course_id}/coursework/{cw_id}/feedback")
async def assignment_feedback(
    request: Request,
    course_id: str,
    cw_id: str,
):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    form = await request.form()
    selected = [v for v in form.getlist("student_ids") if isinstance(v, str) and v]
    teacher_name = (form.get("teacher_name") or "").strip() or None

    if not selected:
        _flash(request, "warn", "No students selected — tick at least one row to send feedback.")
        return RedirectResponse(f"/courses/{course_id}/coursework/{cw_id}", status_code=303)

    # Use the test we last graded with as the email subject's "test name"
    owned = dbmod.get_app_assignment(course_id, cw_id) or {}
    test_id = owned.get("test_id")
    test_obj = dbmod.get_test(test_id) if test_id else None
    test_label = (test_obj or {}).get("name") if test_obj else None

    try:
        result = send_feedback_for_assignment(
            email, course_id, cw_id,
            test_name=test_label,
            only_students=selected,
            teacher_name=teacher_name,
        )
    except Exception as e:  # noqa: BLE001
        _flash(request, "bad", f"Sending feedback failed: {type(e).__name__}: {e}")
        return RedirectResponse(f"/courses/{course_id}/coursework/{cw_id}", status_code=303)

    sent = sum(1 for r in result["results"] if r["status"] == "sent")
    with_overlay = sum(1 for r in result["results"] if r.get("attached_overlay"))
    bad = [r for r in result["results"] if r["status"] != "sent"]
    if bad:
        first_err = next((r.get("error") for r in bad if r.get("error")), None) or bad[0].get("status")
        _flash(request, "warn",
               f"Sent {sent}/{len(selected)} emails (overlay attached on {with_overlay}); "
               f"{len(bad)} failed. First issue: {first_err}")
    else:
        _flash(request, "ok",
               f"Sent {sent} feedback email(s) — overlay PDF attached on {with_overlay}.")
    return RedirectResponse(f"/courses/{course_id}/coursework/{cw_id}", status_code=303)


# ----- per-student detail + manual overrides --------------------------------

_SECTION_LABELS = {
    "Test 1": "English", "Test 2": "Math",
    "Test 3": "Reading", "Test 4": "Science",
}
_SECTION_OPTIONS = [
    {"id": k, "label": v} for k, v in _SECTION_LABELS.items()
]


def _latest_submission_for_student(course_id: str, cw_id: str, student_id: str) -> dict | None:
    rows = dbmod.list_submissions(
        course_id=course_id, coursework_id=cw_id, student_id=student_id
    )
    if not rows:
        return None
    return dbmod.get_submission(rows[0]["id"])


@app.get(
    "/courses/{course_id}/coursework/{cw_id}/students/{student_id}",
    response_class=HTMLResponse,
)
def student_detail_view(request: Request, course_id: str, cw_id: str, student_id: str):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    sub = _latest_submission_for_student(course_id, cw_id, student_id)
    if sub is None:
        _flash(
            request, "warn",
            "No graded submission for this student yet. Pick a test and click "
            "“Fetch + grade all turned-in” on the assignment page first.",
        )
        return RedirectResponse(
            f"/courses/{course_id}/coursework/{cw_id}", status_code=303
        )

    courses = list_courses(email)
    course = next((c for c in courses if c["id"] == course_id), None)
    cw_list = list_coursework(email, course_id)
    cw = next((x for x in cw_list if x["id"] == cw_id), None)
    if course is None or cw is None:
        raise HTTPException(404, "Course or assignment not found.")

    score = sub.get("score") or {}
    sections_info: dict = score.get("sections") or {}

    # Field-test ("skipped") questions are kept out of the answer key, so the
    # stored score never sees them. Re-grade them here (scored key + field-test
    # answers) just to show how the student did on the not-scored questions:
    # how many they got right (next to Correct) vs wrong (next to Wrong).
    from .scoring import grade_answers as _grade, merge_field_test as _merge
    from .submissions import template_for_test
    ft_by_section: dict[str, list] = {}
    try:
        _t = dbmod.get_test(sub.get("test_id")) or {}
        _fta = _t.get("field_test_answers")
        if _fta:
            _tp, _ = template_for_test(sub.get("test_id"))
            _tpl = json.loads(Path(_tp).read_text())
            _ans = {int(k): v for k, v in (sub.get("answers") or {}).items()}
            _re = _grade(_ans, _tpl, _merge(_t.get("answer_key"), _fta), not_scored=_fta)
            ft_by_section = {sec: info.get("details", []) for sec, info in _re.items()}
    except Exception:  # noqa: BLE001 — no field-test breakdown if anything fails
        ft_by_section = {}

    sections = []
    for k in ["Test 1", "Test 2", "Test 3", "Test 4"]:
        if k not in sections_info:
            continue
        info = sections_info[k]
        ft = [d for d in ft_by_section.get(k, []) if not d.get("scored", True)]
        skipped_correct = sum(1 for d in ft if d.get("status") == "correct")
        skipped_wrong = sum(1 for d in ft if d.get("status") in ("incorrect", "blank", "multi"))
        sections.append({
            "display": _SECTION_LABELS.get(k, k), "info": info,
            "skipped_correct": skipped_correct,
            "skipped_wrong": skipped_wrong,
            "skipped": skipped_correct + skipped_wrong,
        })

    # Pre-fill the review/override table. New grades carry score["review"]:
    # every borderline reader call (faint blanks, multis, shaky singles) with
    # a cropped image of the row so the teacher adjudicates from a picture.
    # Older grades fall back to the plain BLANK/MULTI list.
    detail_by_q: dict[int, dict] = {}
    for info in sections_info.values():
        for d in info.get("details", []):
            if d.get("q") is not None:
                detail_by_q[d["q"]] = d

    flagged: list[dict] = []
    review = score.get("review")
    if review is not None:
        for r0 in review:
            d = detail_by_q.get(r0.get("q"), {})
            flagged.append({
                "section": r0.get("section"),
                "q_in_test": r0.get("q_in_test"),
                "q": r0.get("q"),
                "given": r0.get("given"),
                "correct": d.get("correct"),
                "reason": r0.get("reason"),
                "crop_b64": r0.get("crop_b64"),
            })
    else:
        for section_key, info in sections_info.items():
            for d in info.get("details", []):
                if d.get("status") in ("blank", "multi"):
                    flagged.append({
                        "section": section_key,
                        "q_in_test": d.get("q_in_test"),
                        "q": d.get("q"),
                        "given": d.get("given"),
                        "correct": d.get("correct"),
                    })
        flagged.sort(key=lambda r: (list(_SECTION_LABELS).index(r["section"]) if r["section"] in _SECTION_LABELS else 999,
                                    r["q_in_test"] or 0))

    return _render(
        request, "student_detail.html",
        course=course, cw=cw,
        student_id=student_id,
        student_name=sub.get("student_name") or student_id,
        student_email=sub.get("student_email"),
        composite=score.get("composite"),
        sections=sections,
        flagged=flagged,
        section_options=_SECTION_OPTIONS,
        overrides_log=score.get("overrides") or [],
    )


@app.get("/courses/{course_id}/coursework/{cw_id}/students/{student_id}/overlay.pdf")
def student_overlay_pdf(request: Request, course_id: str, cw_id: str, student_id: str):
    """The marked-up sheet (green = correct, red X = missed), viewable in-browser
    — same rendering the feedback email attaches, without having to send one."""
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    from .feedback import build_overlay_for_student
    try:
        pdf, err = build_overlay_for_student(email, course_id, cw_id, student_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Overlay failed: {type(e).__name__}: {e}")
    if pdf is None:
        raise HTTPException(404, err or "No overlay available.")
    return Response(
        content=pdf, media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=marked_up_sheet.pdf"},
    )


@app.post("/courses/{course_id}/coursework/{cw_id}/students/{student_id}/overrides")
async def student_apply_overrides(
    request: Request, course_id: str, cw_id: str, student_id: str
):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    form = await request.form()
    sections_in = form.getlist("section")
    qs_in = form.getlist("q_in_test")
    answers_in = form.getlist("answer")

    detail_url = f"/courses/{course_id}/coursework/{cw_id}/students/{student_id}"

    if not (len(sections_in) == len(qs_in) == len(answers_in)):
        _flash(request, "bad", "Form rows out of alignment — please retry.")
        return RedirectResponse(detail_url, status_code=303)

    sub = _latest_submission_for_student(course_id, cw_id, student_id)
    if sub is None:
        _flash(request, "warn", "No graded submission for this student yet.")
        return RedirectResponse(
            f"/courses/{course_id}/coursework/{cw_id}", status_code=303
        )

    test = dbmod.get_test(sub["test_id"]) if sub.get("test_id") else None
    if not test or not test.get("answer_key"):
        _flash(request, "bad", "Can't recompute — test or answer key missing.")
        return RedirectResponse(detail_url, status_code=303)

    # Pick the OMR template matching the test's format. Without this the
    # legacy 215-question template gets used for new-format submissions,
    # which corrupts the (section, q_in_test) → global q mapping below
    # and wipes Test 4 out of the regraded score entirely.
    from .submissions import template_for_test
    template_path, _ = template_for_test(sub["test_id"])
    template = json.loads(template_path.read_text())

    # (section, q_in_test) → global q. Derived from the chosen template's
    # bubble list so we don't have to hard-code section offsets here.
    sq_to_q: dict[tuple[str, int], int] = {}
    for b in template.get("bubbles", []):
        key = (b.get("section"), b.get("q_in_test"))
        sq_to_q.setdefault(key, b.get("q"))

    cur_answers: dict[int, str] = {int(k): v for k, v in (sub.get("answers") or {}).items()}
    changes: list[dict] = []
    new_overrides: dict[int, str] = {}

    for sec, q_str, ans in zip(sections_in, qs_in, answers_in):
        ans_clean = (ans or "").strip().upper()
        if not ans_clean:
            continue  # blank New-answer cell → ignore that row
        try:
            q_in_test = int(q_str)
        except (TypeError, ValueError):
            continue
        global_q = sq_to_q.get((sec, q_in_test))
        if global_q is None:
            continue  # unknown section/Q
        if ans_clean not in {"A", "B", "C", "D", "E", "F", "G", "H", "J", "K"}:
            continue  # not a valid ACT option letter
        old = cur_answers.get(global_q)
        if old == ans_clean:
            continue  # no-op
        new_overrides[global_q] = ans_clean
        changes.append({
            "q": global_q, "section": sec, "q_in_test": q_in_test,
            "old": old, "new": ans_clean,
        })

    if not new_overrides:
        _flash(request, "warn", "Nothing to apply — all rows were empty or unchanged.")
        return RedirectResponse(detail_url, status_code=303)

    # Apply, recompute, persist, preserving prior override history.
    cur_answers.update(new_overrides)
    new_score = full_grade(cur_answers, template, test["answer_key"], test["scaler"])
    prior_log = ((sub.get("score") or {}).get("overrides")) or []
    new_score["overrides"] = prior_log + changes

    dbmod.update_submission(sub["id"], answers=cur_answers, score=new_score)

    # Optionally re-push to Classroom so a previously-released grade gets updated.
    push = bool((form.get("push_to_classroom") or "").strip())
    push_msg = ""
    if push:
        owned = dbmod.get_app_assignment(course_id, cw_id)
        if not owned:
            push_msg = " — but the assignment isn't app-owned, so Classroom push was skipped."
        elif not sub.get("classroom_submission_id"):
            push_msg = " — but no Classroom submission id on file, so the push was skipped."
        else:
            try:
                from .submissions import release_grades
                push_result = release_grades(
                    email, course_id, cw_id,
                    only_students=[student_id],
                    return_to_student=True,
                )
                ok = any(r["status"] in ("returned", "grade_assigned") for r in push_result["results"])
                if ok:
                    push_msg = " New score pushed to Classroom."
                else:
                    err = next((r.get("error") for r in push_result["results"] if r.get("error")), "unknown")
                    push_msg = f" Override saved, but Classroom push failed: {err}"
            except Exception as e:  # noqa: BLE001
                push_msg = f" Override saved, but Classroom push failed: {type(e).__name__}: {e}"

    _flash(request, "ok",
           f"Applied {len(new_overrides)} override(s). New composite: {new_score.get('composite')}.{push_msg}")
    return RedirectResponse(detail_url, status_code=303)


# ----- tests management -----------------------------------------------------

@app.get("/tests", response_class=HTMLResponse)
def tests_view(request: Request):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    # `list_tests()` already returns has_key, has_scaler, key_count,
    # scaler_count, and the new-format flag — no per-test follow-up queries.
    return _render(request, "tests.html", tests=dbmod.list_tests())


@app.post("/tests/import")
async def tests_import(request: Request, key_pdf: UploadFile = File(...)):
    """Upload an official answer-key PDF; Claude extracts, our validator
    enforces the scored/not-scored split, letter parity, and scaler coverage
    before anything can be saved. Renders a preview for the teacher to confirm."""
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    from .config import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        _flash(request, "bad",
               "Key import isn't configured — the server needs an ANTHROPIC_API_KEY "
               "environment variable (a console.anthropic.com API key).")
        return RedirectResponse("/tests", status_code=303)
    data = await key_pdf.read()
    if len(data) > 25 * 1024 * 1024:
        _flash(request, "bad", "PDF is over 25 MB — that's not an answer-key booklet.")
        return RedirectResponse("/tests", status_code=303)

    from .key_import import claude_extract, extract_pdf_text, validate_and_normalize
    try:
        text = extract_pdf_text(data)
        raw = claude_extract(text, ANTHROPIC_API_KEY)
        norm = validate_and_normalize(raw)
    except ValueError as e:
        _flash(request, "bad", f"Import failed: {e}")
        return RedirectResponse("/tests", status_code=303)
    except Exception as e:  # noqa: BLE001
        _flash(request, "bad", f"Import failed: {type(e).__name__}: {e}")
        return RedirectResponse("/tests", status_code=303)

    sections = []
    for sec in ["Test 1", "Test 2", "Test 3", "Test 4"]:
        if sec in norm["answers"]:
            ft = (norm.get("field_test_answers") or {}).get(sec) or {}
            sections.append({
                "label": {"Test 1": "English", "Test 2": "Math",
                          "Test 3": "Reading", "Test 4": "Science"}[sec],
                "scored": len(norm["answers"][sec]),
                "field_test": len(ft),
                "scaler_max": max(int(k) for k in norm["scaler"][sec]),
            })
    payload = base64.urlsafe_b64encode(json.dumps(norm).encode()).decode()
    return _render(
        request, "tests_import_preview.html",
        norm=norm, sections=sections, payload=payload,
        filename=key_pdf.filename,
    )


@app.post("/tests/import/confirm")
async def tests_import_confirm(
    request: Request,
    payload: str = Form(...),
    test_id: str = Form(""),
    test_name: str = Form(""),
):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    from .key_import import sanity_check
    try:
        norm = json.loads(base64.urlsafe_b64decode(payload.encode()).decode())
        sanity_check(norm)
    except Exception as e:  # noqa: BLE001
        _flash(request, "bad", f"Import payload failed re-validation: {e}")
        return RedirectResponse("/tests", status_code=303)

    tid = (test_id or norm.get("test_id_hint") or "").strip()
    name = (test_name or norm.get("name") or tid).strip()
    if not tid:
        _flash(request, "bad", "A test id is required.")
        return RedirectResponse("/tests", status_code=303)
    if dbmod.get_test(tid) is not None:
        _flash(request, "bad", f"A test with id '{tid}' already exists — pick another id.")
        return RedirectResponse("/tests", status_code=303)

    notes = "Imported from answer-key PDF."
    if norm.get("test_form"):
        notes += f" Form {norm['test_form']}."
    if norm.get("conversions"):
        notes += (f" {len(norm['conversions'])} letters converted to sheet positions "
                  "(renumbered booklet).")
    dbmod.upsert_test(tid, name=name, notes=notes)
    dbmod.set_test_answer_key(tid, norm["answers"])
    dbmod.set_test_scaler(tid, norm["scaler"])
    dbmod.set_test_field_test_answers(tid, norm.get("field_test_answers"))
    n_scored = sum(len(v) for v in norm["answers"].values())
    n_ft = sum(len(v) for v in (norm.get("field_test_answers") or {}).values())
    _flash(request, "ok",
           f"Imported '{name}' — {n_scored} scored answers"
           + (f" + {n_ft} field-test (not scored)" if n_ft else "")
           + ". It's ready in the grade dropdown.")
    return RedirectResponse("/tests", status_code=303)


@app.post("/tests/new")
async def tests_new(
    request: Request,
    test_id: str = Form(...),
    name: str = Form(...),
    notes: str = Form(""),
    key_file: UploadFile | None = File(None),
    scaler_file: UploadFile | None = File(None),
):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email

    dbmod.upsert_test(test_id, name=name, notes=notes or None)
    if key_file is not None and key_file.filename:
        try:
            payload = json.loads((await key_file.read()).decode())
            inner = payload.get("answers", payload)
            dbmod.set_test_answer_key(test_id, inner)
        except Exception as e:  # noqa: BLE001
            _flash(request, "warn", f"Saved test but couldn't load key: {e}")
            return RedirectResponse("/tests", status_code=303)
    if scaler_file is not None and scaler_file.filename:
        try:
            payload = json.loads((await scaler_file.read()).decode())
            inner = payload.get("scaler", payload)
            dbmod.set_test_scaler(test_id, inner)
        except Exception as e:  # noqa: BLE001
            _flash(request, "warn", f"Saved test but couldn't load scaler: {e}")
            return RedirectResponse("/tests", status_code=303)
    _flash(request, "ok", f"Saved test '{test_id}'.")
    return RedirectResponse("/tests", status_code=303)


@app.post("/tests/{test_id}/delete")
def tests_delete(request: Request, test_id: str):
    email = _require(request)
    if isinstance(email, RedirectResponse):
        return email
    dbmod.delete_test(test_id)
    _flash(request, "ok", f"Deleted test '{test_id}'.")
    return RedirectResponse("/tests", status_code=303)


# ----- JSON API kept for the CLI --------------------------------------------

@app.get("/teachers")
def teachers_api():
    return {"teachers": dbmod.list_teachers()}


@app.get("/teachers/{email}/courses")
def teacher_courses_api(email: str):
    try:
        courses = list_courses(email)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {
        "email": email,
        "count": len(courses),
        "courses": [{"id": c["id"], "name": c["name"], "section": c.get("section")} for c in courses],
    }


@app.get("/teachers/{email}/courses/{course_id}/coursework")
def teacher_coursework_api(email: str, course_id: str):
    items = list_coursework(email, course_id)
    return {
        "count": len(items),
        "items": [
            {"id": cw["id"], "title": cw.get("title"), "type": cw.get("workType"),
             "state": cw.get("state"), "due": cw.get("dueDate")}
            for cw in items
        ],
    }


@app.get("/teachers/{email}/courses/{course_id}/students")
def teacher_students_api(email: str, course_id: str):
    return {
        "students": [
            {"id": s["userId"],
             "name": (s.get("profile") or {}).get("name", {}).get("fullName"),
             "email": (s.get("profile") or {}).get("emailAddress")}
            for s in list_roster(email, course_id)
        ]
    }


@app.get("/teachers/{email}/courses/{course_id}/coursework/{cw_id}/submissions")
def teacher_submissions_api(email: str, course_id: str, cw_id: str):
    subs = list_submissions(email, course_id, cw_id)
    return {
        "count": len(subs),
        "submissions": [
            {"id": s["id"], "user_id": s["userId"], "state": s.get("state"),
             "attachments": len((s.get("assignmentSubmission") or {}).get("attachments", []) or [])}
            for s in subs
        ],
    }


@app.post("/teachers/{email}/courses/{course_id}/coursework/{cw_id}/fetch")
def teacher_fetch_api(email: str, course_id: str, cw_id: str, only_turned_in: bool = True):
    return fetch_assignment(email, course_id, cw_id, only_turned_in=only_turned_in)


def main() -> None:
    import uvicorn
    uvicorn.run(
        "bubble_grader.server:app",
        host="127.0.0.1",
        port=SERVER_PORT,
        reload=False,
    )
