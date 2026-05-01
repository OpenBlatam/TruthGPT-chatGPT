"""
OpenClaw Messaging -- Email Adapter (SMTP / IMAP).

Sends and receives emails, forwarding them to the AgentClient.

Environment variables:
    EMAIL_SMTP_HOST      -- SMTP server host (default smtp.gmail.com).
    EMAIL_SMTP_PORT      -- SMTP server port (default 587).
    EMAIL_IMAP_HOST      -- IMAP server host (default imap.gmail.com).
    EMAIL_ADDRESS        -- Sender email address.
    EMAIL_PASSWORD       -- App password or account password.
"""

import email
import logging
import os
import smtplib
from email.mime.text import MIMEText
from typing import Any, Dict, Optional, Union

from ..models import AgentResponse
from .base import BaseMessagingAdapter

logger = logging.getLogger(__name__)


class EmailAdapter(BaseMessagingAdapter):
    """
    Email messaging adapter (SMTP outbound, IMAP inbound).

    Modes:
    1. **Webhook / polling** -- A cron job or background task calls
       :meth:`check_inbox` and forwards new emails to the agent.
    2. **Direct** -- Call :meth:`process_email` with parsed email data.
    """

    def __init__(self, agent_client: Any) -> None:
        super().__init__(agent_client)
        self.smtp_host = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self.imap_host = os.getenv("EMAIL_IMAP_HOST", "imap.gmail.com")
        self.address = os.getenv("EMAIL_ADDRESS", "")
        self.password = os.getenv("EMAIL_PASSWORD", "")

        if not self.address or not self.password:
            logger.warning(
                "EMAIL_ADDRESS / EMAIL_PASSWORD not set. "
                "Email integration will not work until credentials are provided."
            )

    # ------------------------------------------------------------------
    # BaseMessagingAdapter implementation
    # ------------------------------------------------------------------

    async def on_message(
        self,
        platform_user_id: str,
        text: str,
        metadata: Optional[dict] = None,
    ) -> AgentResponse:
        return await self.agent_client.run(
            user_id=f"email_{platform_user_id}",
            prompt=text,
            return_response=True
        )

    async def send_response(
        self,
        platform_user_id: str,
        response: AgentResponse,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Send an email reply via SMTP."""
        subject = (metadata or {}).get("subject", "OpenClaw Agent Reply")
        to_address = platform_user_id  # the sender's email

        response_text = response.content
        if response.action_type == "approval_required":
            response_text = (
                f"{response_text}\n\n"
                "--- ACTION REQUIRED ---\n"
                "The agent requires manual approval to proceed with this sensitive tool. "
                "Please approve via the API or respond with your authorization."
            )

        msg = MIMEText(response_text)
        msg["Subject"] = f"Re: {subject}"
        msg["From"] = self.address
        msg["To"] = to_address

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.address, self.password)
                server.send_message(msg)
            logger.info("Email sent to %s", to_address)
            return True
        except Exception:
            logger.exception("Failed to send email to %s", to_address)
            return False

    # ------------------------------------------------------------------
    # Inbound helpers
    # ------------------------------------------------------------------

    async def process_email(self, email_data: Dict[str, str]) -> Optional[AgentResponse]:
        """
        Process an incoming email.

        Args:
            email_data: dict with keys ``from``, ``subject``, ``body``.

        Returns:
            The agent response, or *None* if the body is empty.
        """
        sender = email_data.get("from", "")
        subject = email_data.get("subject", "")
        body = email_data.get("body", "").strip()

        if not body:
            return None

        return await self.handle(
            platform_user_id=sender,
            text=body,
            metadata={"subject": subject},
        )

    def check_inbox(self, folder: str = "INBOX", limit: int = 5) -> list:
        """
        Synchronously fetch the latest unread emails via IMAP.

        Returns a list of dicts: ``[{"from": ..., "subject": ..., "body": ...}]``
        """
        import imaplib

        results = []
        try:
            mail = imaplib.IMAP4_SSL(self.imap_host)
            mail.login(self.address, self.password)
            mail.select(folder)

            _, data = mail.search(None, "UNSEEN")
            ids = data[0].split()[-limit:] if data[0] else []

            for eid in ids:
                _, msg_data = mail.fetch(eid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)

                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            body = payload.decode(errors="replace") if payload else str(part)
                            break
                else:
                    payload = msg.get_payload(decode=True)
                    body = payload.decode(errors="replace") if payload else str(msg)

                results.append({
                    "from": msg.get("From", ""),
                    "subject": msg.get("Subject", ""),
                    "body": body,
                })

            mail.logout()
        except Exception:
            logger.exception("Failed to check inbox")

        return results

