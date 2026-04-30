"""
core/assistant.py
The heart of JARVIS — state machine that orchestrates:
  STT → Wake/Sleep detection → Agent → TTS
  
States:
  IDLE    — Listening for wake words only
  ACTIVE  — Fully listening, processing all commands
"""
import logging
import threading
import time
from enum import Enum, auto

from core.agent import JarvisAgent
from core.stt import SpeechToText
from core.tts import TextToSpeech
from core.wake_word import CommandType, detect_command, strip_wake_word
from config import settings

logger = logging.getLogger(__name__)


class JarvisState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    SHUTTING_DOWN = auto()


# ── Personality Phrases ───────────────────────────────────────────────────────
WAKE_RESPONSES = [
    "Online and ready, sir.",
    "Yes, I'm here. What do you need?",
    "At your service.",
    "Good to hear from you. What can I do for you?",
    "JARVIS online. How can I help?",
]

SLEEP_RESPONSES = [
    "Going to standby mode. Call me anytime.",
    "Understood. I'll be here if you need me.",
    "Entering idle mode. Just say my name to wake me.",
    "Standby activated. Rest well, sir.",
]

IDLE_HEARING = [
    "I'm listening...",
    "Go ahead, sir.",
    "What would you like to know?",
]

import random


class JarvisAssistant:
    """
    Main JARVIS controller.
    Runs the listen → detect → respond loop in a background thread.
    """

    def __init__(self):
        logger.info("⚡  Initializing JARVIS systems...")

        self.tts = TextToSpeech()
        self.stt = SpeechToText()
        self.agent = JarvisAgent(speak_callback=self.tts.speak)

        self.state = JarvisState.IDLE
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()

        # Autonomous mode settings
        self._autonomous_interval = settings.AUTONOMOUS_INTERVAL
        self._last_autonomous_check = time.time()

        logger.info("✅  All systems initialized.")

    # ── State Management ─────────────────────────────────────────────────────

    def _set_state(self, new_state: JarvisState):
        with self._state_lock:
            if self.state == new_state:
                return
            old = self.state.name
            self.state = new_state
            logger.info(f"🔄  State: {old} → {new_state.name}")

    def _get_state(self) -> JarvisState:
        with self._state_lock:
            return self.state

    # ── Wake / Sleep ─────────────────────────────────────────────────────────

    def _wake_up(self):
        self._set_state(JarvisState.ACTIVE)
        self.agent.clear_memory()  # Fresh context on wake
        response = random.choice(WAKE_RESPONSES)
        logger.info(f"🟢  JARVIS ACTIVE — \"{response}\"")
        self.tts.speak(response)

    def _go_to_sleep(self):
        self._set_state(JarvisState.IDLE)
        response = random.choice(SLEEP_RESPONSES)
        logger.info(f"🌙  JARVIS IDLE — \"{response}\"")
        self.tts.speak(response)

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def _print_status(self):
        """Print listening status to terminal."""
        state = self._get_state()
        if state == JarvisState.IDLE:
            print("\r🌙  [IDLE]   Listening for wake word...   ", end="", flush=True)
        else:
            print("\r🟢  [ACTIVE] Listening...                 ", end="", flush=True)

    def _handle_text(self, text: str):
        """Core logic: route transcribed text to appropriate action."""
        state = self._get_state()
        cmd = detect_command(text)

        # ── IDLE MODE ────────────────────────────────────────────────────────
        if state == JarvisState.IDLE:
            if cmd == CommandType.WAKE:
                # Wake word detected — also check if there's a query after it
                query = strip_wake_word(text)
                self._wake_up()
                if query and len(query) > 2:
                    # e.g. "Hey JARVIS, what time is it?" — handle immediately
                    self._process_query(query)
            # In idle mode, ignore everything else
            return

        # ── ACTIVE MODE ──────────────────────────────────────────────────────
        if cmd == CommandType.SLEEP:
            self._go_to_sleep()

        elif cmd == CommandType.WAKE:
            # Already awake — acknowledge
            self.tts.speak(random.choice(IDLE_HEARING))

        elif cmd == CommandType.QUERY:
            self._process_query(text)

    def _process_query(self, query: str):
        """Send query to agent and speak the response."""
        print()  # Newline after status line
        logger.info(f"📨  Processing query: \"{query}\"")

        response = self.agent.think(query)
        self.tts.speak(response)
        self._print_status()

    def _autonomous_tick(self):
        """
        Called periodically — JARVIS can take initiative here.
        Override or extend this to add proactive behavior.
        Currently: just a hook for future autonomous tasks.
        """
        if self._autonomous_interval <= 0:
            return

        now = time.time()
        if now - self._last_autonomous_check >= self._autonomous_interval:
            self._last_autonomous_check = now
            # Future: check calendar, emails, news, etc.
            logger.debug("Autonomous tick — no tasks configured.")

    # ── Public Interface ──────────────────────────────────────────────────────

    def startup(self):
        """Speak the startup announcement."""
        self.tts.speak(
            "JARVIS systems online. All modules initialized. "
            "Say 'Hey JARVIS' whenever you need me, sir."
        )
        self._print_status()

    def run(self):
        """
        Main blocking loop.
        Continuously listens → processes → responds.
        Run this on the main thread or a dedicated thread.
        """
        logger.info("🚀  JARVIS is running. Press Ctrl+C to stop.")
        self.startup()

        try:
            while not self._stop_event.is_set():
                self._print_status()

                # Listen for speech
                text = self.stt.listen()

                if text:
                    self._handle_text(text)

                # Autonomous behavior hook
                self._autonomous_tick()

                # Small sleep to prevent busy-waiting
                time.sleep(0.05)

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown."""
        self._set_state(JarvisState.SHUTTING_DOWN)
        self._stop_event.set()
        print()
        logger.info("🛑  Shutting down JARVIS...")
        self.tts.speak("JARVIS shutting down. Goodbye, sir.")