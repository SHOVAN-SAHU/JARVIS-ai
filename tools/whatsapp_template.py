"""
tools/whatsapp_template.py
─────────────────────────────────────────────────────────────────────────────
TEMPLATE — Future WhatsApp Tool
─────────────────────────────────────────────────────────────────────────────

This is a blueprint showing HOW to build new tools for JARVIS.
Copy this file, rename it, and fill in your logic.

To activate:
  1. Implement _run() below
  2. In tools/__init__.py, uncomment the WhatsApp import lines
─────────────────────────────────────────────────────────────────────────────
"""
from typing import Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class WhatsAppInput(BaseModel):
    action: str = Field(
        description="Action to perform: 'check_messages', 'send_message', 'get_contacts'"
    )
    contact: str = Field(default="", description="Contact name (for sending)")
    message: str = Field(default="", description="Message to send")


class WhatsAppTool(BaseTool):
    name: str = "whatsapp"
    description: str = (
        "Interact with WhatsApp: check new messages, send messages to contacts. "
        "Actions: 'check_messages', 'send_message'. "
        "For sending, provide contact name and message."
    )
    args_schema: Type[BaseModel] = WhatsAppInput
    before_action_phrase: str = "Alright, checking your WhatsApp messages now."

    def _run(self, action: str, contact: str = "", message: str = "") -> str:
        """
        Implement actual WhatsApp integration here.
        Options:
          - whatsapp-web.py (browser automation)
          - Twilio WhatsApp API
          - Meta Business API
        """
        if action == "check_messages":
            # TODO: Connect to WhatsApp API / web automation
            return "WhatsApp integration not yet implemented. Connect a WhatsApp API here."

        elif action == "send_message":
            if not contact or not message:
                return "Please provide both a contact name and a message to send."
            # TODO: Send message logic
            return f"Message to {contact} queued (not yet implemented)."

        return "Unknown WhatsApp action."

    async def _arun(self, action: str, contact: str = "", message: str = "") -> str:
        return self._run(action, contact, message)