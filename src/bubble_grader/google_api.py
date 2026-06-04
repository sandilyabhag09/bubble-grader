"""Build authenticated Google API clients from a teacher's stored credentials."""

from googleapiclient.discovery import build

from .db import load_credentials, store_credentials
from .google_auth import (
    credentials_from_dict,
    credentials_to_dict,
    ensure_fresh,
)


def service_for(email: str, service: str, version: str):
    """Return a discovery client (e.g. classroom v1, drive v3) for the given teacher."""
    raw = load_credentials(email)
    if not raw:
        raise ValueError(
            f"No stored credentials for {email}. Sign in via /oauth/start first."
        )
    creds = credentials_from_dict(raw)
    creds = ensure_fresh(creds)

    # If refresh issued a new access token, persist it (refresh_token is unchanged).
    if creds.token != raw.get("token"):
        new = credentials_to_dict(creds)
        new["id_token"] = raw.get("id_token")  # id_token isn't reissued on refresh
        store_credentials(email, new)

    return build(service, version, credentials=creds, cache_discovery=False)
