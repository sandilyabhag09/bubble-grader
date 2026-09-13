"""Download Drive file attachments using a teacher's stored credentials."""

from pathlib import Path

from .config import DATA_DIR
from .google_api import service_for

# Map of mimeType → file extension we save as. Anything else falls back to .bin.
EXT_BY_MIME = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/heic": "heic",
    "image/heif": "heif",
    "image/webp": "webp",
    "application/pdf": "pdf",
}


def _drive(email: str):
    return service_for(email, "drive", "v3")


def file_metadata(email: str, file_id: str) -> dict:
    return (
        _drive(email)
        .files()
        .get(fileId=file_id, fields="id,name,mimeType,size", supportsAllDrives=True)
        .execute()
    )


def download_file_bytes(email: str, file_id: str) -> tuple[bytes, dict]:
    """Download a Drive file into memory; return (content, metadata).

    Used by the DB-backed scan store — nothing touches local disk, so this
    works identically on a laptop and on serverless hosting.
    """
    meta = file_metadata(email, file_id)
    data = _drive(email).files().get_media(fileId=file_id).execute()
    return data, meta


def download_file(email: str, file_id: str, dest_dir: Path) -> tuple[Path, dict]:
    """Download a Drive file to dest_dir keyed by file_id; return (path, metadata).

    No-ops if the file is already cached at the expected path."""
    meta = file_metadata(email, file_id)
    ext = EXT_BY_MIME.get(meta.get("mimeType", ""), "bin")
    dest = Path(dest_dir) / f"{file_id}.{ext}"
    if not dest.exists():
        data = _drive(email).files().get_media(fileId=file_id).execute()
        dest.write_bytes(data)
    return dest, meta


def submission_cache_dir(course_id: str, coursework_id: str, student_id: str) -> Path:
    p = DATA_DIR / "submissions" / course_id / coursework_id / student_id
    p.mkdir(parents=True, exist_ok=True)
    return p


RESULTS_FOLDER = "Grader Form results"


def _results_folder_id(svc, folder_name: str = RESULTS_FOLDER) -> str:
    """Find (or create) the teacher's results folder in their own Drive."""
    safe = folder_name.replace("'", "\\'")
    res = svc.files().list(
        q=(f"name = '{safe}' and mimeType = 'application/vnd.google-apps.folder' "
           "and trashed = false and 'me' in owners"),
        fields="files(id)", pageSize=1,
    ).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    created = svc.files().create(
        body={"name": folder_name, "mimeType": "application/vnd.google-apps.folder"},
        fields="id",
    ).execute()
    return created["id"]


def upload_pdf(email: str, name: str, data: bytes) -> dict:
    """Upload PDF bytes to the teacher's Drive results folder. Returns {id, webViewLink}.

    Needs the drive.file scope (files created by this app only)."""
    import io
    from googleapiclient.http import MediaIoBaseUpload

    svc = _drive(email)
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/pdf", resumable=False)
    return svc.files().create(
        body={"name": name, "parents": [_results_folder_id(svc)]},
        media_body=media,
        fields="id,webViewLink",
    ).execute()
