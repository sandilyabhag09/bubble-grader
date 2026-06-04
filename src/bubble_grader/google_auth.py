"""Google OAuth helpers: authorization URL, code exchange, credential (de)serialization."""

import os

# Google sometimes adds extra granted scopes (e.g. profile when you request email).
# This stops google-auth-oauthlib from raising on the scope mismatch.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

import secrets as secrets_mod

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token as google_id_token
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from .config import CLIENT_SECRET_PATH, OAUTH_REDIRECT_URI, SCOPES


def build_flow(state: str | None = None) -> Flow:
    return Flow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        scopes=SCOPES,
        redirect_uri=OAUTH_REDIRECT_URI,
        state=state,
    )


def authorization_url() -> tuple[str, str, str]:
    """Return (url, state, code_verifier). Caller must persist both for the callback."""
    state = secrets_mod.token_urlsafe(32)
    # PKCE: 43-128 char URL-safe string. token_urlsafe(64) → ~86 chars.
    code_verifier = secrets_mod.token_urlsafe(64)
    flow = build_flow(state=state)
    flow.code_verifier = code_verifier  # Flow derives the S256 code_challenge from this
    url, _ = flow.authorization_url(
        access_type="offline",          # required for a refresh_token
        prompt="consent",               # forces refresh_token even on re-login
        include_granted_scopes="true",
    )
    return url, state, code_verifier


def exchange_code(code: str, state: str, code_verifier: str) -> Credentials:
    flow = build_flow(state=state)
    flow.code_verifier = code_verifier  # sent as `code_verifier` on the token request
    flow.fetch_token(code=code)
    return flow.credentials


def credentials_to_dict(creds: Credentials) -> dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else [],
        "id_token": creds.id_token,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }


def credentials_from_dict(d: dict) -> Credentials:
    return Credentials(
        token=d.get("token"),
        refresh_token=d.get("refresh_token"),
        token_uri=d.get("token_uri"),
        client_id=d.get("client_id"),
        client_secret=d.get("client_secret"),
        scopes=d.get("scopes"),
    )


def ensure_fresh(creds: Credentials) -> Credentials:
    if not creds.token or creds.expired:
        creds.refresh(GoogleRequest())
    return creds


def email_from_credentials(creds: Credentials) -> str:
    """Verify the id_token we just got from Google and pull the email claim."""
    idinfo = google_id_token.verify_oauth2_token(
        creds.id_token, GoogleRequest(), creds.client_id
    )
    return idinfo["email"]
