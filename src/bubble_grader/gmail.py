"""Send email from a teacher's own Gmail account using their stored OAuth credentials.

The Classroom API doesn't expose private comments to third-party apps, so this
is our closest substitute: the teacher's Gmail relays personalized feedback to
each student. The 'From' address is the teacher's own email, so replies route
back to them.
"""

import base64
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .google_api import service_for

# Each attachment is (filename, bytes, mimetype). e.g.
#   ("Lalitha_results.pdf", b"%PDF-1.4...", "application/pdf")
Attachment = tuple[str, bytes, str]


def send_email(
    teacher_email: str,
    to_email: str,
    subject: str,
    body: str,
    attachments: list[Attachment] | None = None,
) -> dict:
    """Send a plain-text email (optionally with attachments) from the teacher's Gmail."""
    svc = service_for(teacher_email, "gmail", "v1")

    if attachments:
        msg: MIMEMultipart | MIMEText = MIMEMultipart()
        msg.attach(MIMEText(body, "plain", "utf-8"))
        for filename, data, mimetype in attachments:
            _main, _sub = mimetype.split("/", 1)
            part = MIMEApplication(data, _subtype=_sub)
            part.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(part)
    else:
        msg = MIMEText(body, "plain", "utf-8")

    msg["to"] = to_email
    msg["from"] = teacher_email
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return svc.users().messages().send(userId="me", body={"raw": raw}).execute()
