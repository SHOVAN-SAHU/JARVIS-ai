"""
main.py
JARVIS entry point.
Sets up logging, validates config, and starts the assistant.
"""
import logging
import sys
from pathlib import Path

import colorlog

# ── Logging Setup ─────────────────────────────────────────────────────────────
def setup_logging(log_level: str = "INFO"):
    """Configure colorized console logging + rotating file logging."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )

    # File handler (plain text)
    file_handler = logging.FileHandler(log_dir / "jarvis.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Silence noisy libraries
    for noisy in ("httpx", "httpcore", "urllib3", "langchain", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
  Just A Rather Very Intelligent System
  ─────────────────────────────────────────
"""


def main():
    print(BANNER)

    # Load config (will raise if GROQ_API_KEY is missing)
    try:
        from config import settings
        setup_logging(settings.LOG_LEVEL)
    except EnvironmentError as e:
        print(f"\n❌  Configuration Error:\n{e}\n")
        sys.exit(1)

    logger = logging.getLogger(__name__)
    logger.info("Starting JARVIS...")

    # Import here so logging is set up first
    from core.assistant import JarvisAssistant

    assistant = JarvisAssistant()

    try:
        assistant.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


main()