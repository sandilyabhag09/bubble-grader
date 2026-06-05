"""Background update-availability check.

Periodically asks GitHub's commits API for the latest SHA on `main` and
compares it to the SHA the local install knows about. The UI reads
``UPDATE_STATE`` (a thread-safe snapshot) and shows a banner when the
remote is ahead. Failures are silent — no banner just means "no signal."
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

from .config import PROJECT_ROOT


_GITHUB_OWNER_REPO = "sandilyabhag09/bubble-grader"
_COMMITS_API = f"https://api.github.com/repos/{_GITHUB_OWNER_REPO}/commits/main"
_INSTALLED_SHA_FILE = PROJECT_ROOT / "data" / ".installed_sha"
_POLL_SECONDS = 30 * 60  # every 30 minutes


@dataclass(frozen=True)
class UpdateState:
    available: bool = False
    local_sha: str | None = None
    remote_sha: str | None = None
    last_checked: float = 0.0  # unix time of last successful remote fetch


UPDATE_STATE = UpdateState()
_lock = threading.Lock()


def _git_head_sha(root: Path) -> str | None:
    """Return the local git HEAD SHA, or None if this isn't a git checkout."""
    if not (root / ".git").exists() or shutil.which("git") is None:
        return None
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def local_sha() -> str | None:
    """SHA of the currently-installed code, from git or the tarball-stamp file."""
    sha = _git_head_sha(PROJECT_ROOT)
    if sha:
        return sha
    if _INSTALLED_SHA_FILE.exists():
        s = _INSTALLED_SHA_FILE.read_text().strip()
        return s or None
    return None


def record_installed_sha(sha: str) -> None:
    """Tarball installs call this after a successful update to stamp the version."""
    _INSTALLED_SHA_FILE.parent.mkdir(parents=True, exist_ok=True)
    _INSTALLED_SHA_FILE.write_text(sha.strip() + "\n")


def fetch_remote_sha(timeout: float = 5.0) -> str | None:
    """Latest commit SHA on main per GitHub, or None on any failure."""
    req = urllib.request.Request(
        _COMMITS_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "bubble-grader"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        sha = payload.get("sha")
        return sha if isinstance(sha, str) and sha else None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _update_once() -> None:
    """One check cycle. Updates UPDATE_STATE in place under the lock."""
    global UPDATE_STATE
    local = local_sha()
    remote = fetch_remote_sha()
    if remote is None:
        # Couldn't reach GitHub — keep the previous state so an existing
        # banner doesn't flicker off on transient network blips.
        return
    available = bool(local) and bool(remote) and local != remote
    with _lock:
        UPDATE_STATE = UpdateState(
            available=available,
            local_sha=local,
            remote_sha=remote,
            last_checked=time.time(),
        )


def start_background_poller() -> threading.Thread:
    """Kick off a daemon thread that runs ``_update_once`` periodically."""
    def loop() -> None:
        # First check after a short delay so the server can come up cleanly
        # even if GitHub is slow.
        time.sleep(5)
        while True:
            try:
                _update_once()
            except Exception:  # noqa: BLE001 — never let the poller crash the app
                pass
            time.sleep(_POLL_SECONDS)

    t = threading.Thread(target=loop, name="version-check", daemon=True)
    t.start()
    return t


def snapshot() -> UpdateState:
    """Cheap read of the current state for the request handler."""
    with _lock:
        return replace(UPDATE_STATE)
