"""
config/settings.py
Centralized configuration — all values pulled from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get(key: str, default=None, required=False):
    val = os.getenv(key, default)
    if required and not val:
        raise EnvironmentError(
            f"Missing required env variable: {key}\n"
            f"Please copy .env.example → .env and fill it in."
        )
    return val


# ── Groq ─────────────────────────────────────────────────────
GROQ_API_KEY: str = _get("GROQ_API_KEY", required=True)
GROQ_LLM_MODEL: str = _get("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_WHISPER_MODEL: str = _get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

# ── TTS ──────────────────────────────────────────────────────
TTS_VOICE: str = _get("TTS_VOICE", "en-US-GuyNeural")
TTS_RATE: str = _get("TTS_RATE", "+10%")
TTS_VOLUME: str = _get("TTS_VOLUME", "+0%")

# ── Wake / Sleep Words ───────────────────────────────────────
WAKE_WORDS: list[str] = [
    w.strip().lower()
    for w in _get("WAKE_WORDS", "hey jarvis,wake up,jarvis").split(",")
]
SLEEP_WORDS: list[str] = [
    w.strip().lower()
    for w in _get("SLEEP_WORDS", "sleep,go to sleep,goodbye jarvis").split(",")
]

# ── Audio Recording ──────────────────────────────────────────
RECORD_SAMPLE_RATE: int = int(_get("RECORD_SAMPLE_RATE", 16000))
RECORD_CHANNELS: int = int(_get("RECORD_CHANNELS", 1))
SILENCE_THRESHOLD: float = float(_get("SILENCE_THRESHOLD", 0.03))
SILENCE_DURATION: float = float(_get("SILENCE_DURATION", 1.5))
MAX_RECORD_SECONDS: int = int(_get("MAX_RECORD_SECONDS", 15))

# ── Agent ────────────────────────────────────────────────────
AGENT_MAX_ITERATIONS: int = int(_get("AGENT_MAX_ITERATIONS", 5))
AGENT_VERBOSE: bool = _get("AGENT_VERBOSE", "false").lower() == "true"

# ── App ──────────────────────────────────────────────────────
LOG_LEVEL: str = _get("LOG_LEVEL", "INFO").upper()
LOGS_DIR: Path = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
AUTONOMOUS_INTERVAL: int = int(_get("AUTONOMOUS_INTERVAL", 0))