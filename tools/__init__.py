"""
tools/__init__.py
Central registry — import all tools here and expose get_all_tools().
Adding a new tool = create the file + add it to this list. That's it.
"""
from .datetime_tool import DateTimeTool
from .websearch_tool import WebSearchTool
from .system_tool import SystemInfoTool

# Singleton — tools are instantiated once at import time
_TOOLS = [
    DateTimeTool(),
    WebSearchTool(),
    SystemInfoTool(),
]

def get_all_tools() -> list:
    """Returns the shared tool instances."""
    return _TOOLS