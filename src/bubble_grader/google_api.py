"""Build authenticated Google API clients from a teacher's stored credentials.

Performance note: the expensive part of every Google call used to be fetching
and decrypting the teacher's credentials from the database (~300 ms, on every
single call). Decrypted credentials are now cached in-process for a few
minutes; the discovery client itself is cheap (~25 ms) and is rebuilt per call
because googleapiclient service objects aren't safe to share across threads —
and pages fan their Google calls out in parallel.
"""

from __future__ import annotations

import threading
import time

from googleapiclient.discovery import build

from .db import load_credentials, store_credentials
from .google_auth import (
    credentials_from_dict,
    credentials_to_dict,
    ensure_fresh,
)

# Long enough that a teacher's credentials are read from the database about
# once a day per instance, not on every burst of Classroom notifications.
# A revoked grant drops the entry immediately (see _credentials_for).
_CREDS_TTL_SECONDS = 12 * 60 * 60
_creds_cache: dict[str, tuple[object, dict, float]] = {}  # email -> (creds, raw, expires_at)
_lock = threading.Lock()


def forget_credentials(email: str | None = None) -> None:
    """Drop cached credentials (all, or one teacher) — e.g. after a re-sign-in."""
    with _lock:
        if email is None:
            _creds_cache.clear()
        else:
            _creds_cache.pop(email, None)


def _credentials_for(email: str):
    now = time.monotonic()
    with _lock:
        hit = _creds_cache.get(email)
    if hit and hit[2] > now:
        creds, raw = hit[0], hit[1]
    else:
        raw = load_credentials(email)
        if not raw:
            raise ValueError(
                f"No stored credentials for {email}. Sign in via /oauth/start first."
            )
        creds = credentials_from_dict(raw)
        with _lock:
            _creds_cache[email] = (creds, raw, now + _CREDS_TTL_SECONDS)

    from google.auth.exceptions import RefreshError
    try:
        creds = ensure_fresh(creds)
    except RefreshError:
        forget_credentials(email)  # revoked/expired grant: re-read after a re-sign-in
        raise
    if creds.token != raw.get("token"):
        new = credentials_to_dict(creds)
        new["id_token"] = raw.get("id_token")  # id_token isn't reissued on refresh
        # Only a new refresh token is worth a database write; a fresh access
        # token is cheap to mint again and saving it would wake the database.
        if creds.refresh_token != raw.get("refresh_token"):
            store_credentials(email, new)
        with _lock:
            _creds_cache[email] = (creds, new, now + _CREDS_TTL_SECONDS)
    return creds


def service_for(email: str, service: str, version: str):
    """Return a discovery client (e.g. classroom v1, drive v3) for the given teacher."""
    creds = _credentials_for(email)
    return build(service, version, credentials=creds, cache_discovery=False)
