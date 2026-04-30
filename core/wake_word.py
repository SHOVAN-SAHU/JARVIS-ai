"""
core/wake_word.py
Wake word and sleep command detection.
Works on the transcribed text from STT — no separate model needed.
"""
import logging
from enum import Enum, auto

from config import settings

logger = logging.getLogger(__name__)


class CommandType(Enum):
    WAKE = auto()      # Wake JARVIS up
    SLEEP = auto()     # Put JARVIS to sleep
    QUERY = auto()     # Normal query/command
    UNKNOWN = auto()   # Nothing useful


def detect_command(text: str) -> CommandType:
    """
    Analyze transcribed text and classify the command type.
    Returns CommandType enum value.
    """
    if not text:
        return CommandType.UNKNOWN

    lower = text.lower().strip()

    # Check wake words
    for wake in settings.WAKE_WORDS:
        if wake in lower:
            logger.debug(f"Wake word detected: '{wake}'")
            return CommandType.WAKE

    # Check sleep words
    for sleep_cmd in settings.SLEEP_WORDS:
        if sleep_cmd in lower:
            logger.debug(f"Sleep command detected: '{sleep_cmd}'")
            return CommandType.SLEEP

    return CommandType.QUERY


def strip_wake_word(text: str) -> str:
    """
    Remove the wake word prefix from a query.
    e.g. "Hey JARVIS what's the weather" → "what's the weather"
    """
    lower = text.lower()
    for wake in settings.WAKE_WORDS:
        if lower.startswith(wake):
            return text[len(wake):].strip(" ,.")
    return text