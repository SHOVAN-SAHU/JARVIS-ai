"""
tools/__init__.py
Central registry — import all tools here and expose get_all_tools().
Adding a new tool = create the file + add it to this list. That's it.
"""
from .datetime_tool import DateTimeTool
from .websearch_tool import WebSearchTool
from .system_tool import SystemInfoTool


def get_all_tools() -> list:
    """Returns initialized instances of all available JARVIS tools."""
    return [
        DateTimeTool(),
        WebSearchTool(),
        SystemInfoTool(),
        # ── Add new tools here ──────────────────────────────────────
        # from .whatsapp_tool import WhatsAppTool
        # WhatsAppTool(),
        # ───────────────────────────────────────────────────────────
    ]