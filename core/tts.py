# """
# core/tts.py
# Text-to-Speech using Microsoft Edge TTS (edge-tts).
# High quality neural voices, free, no API key needed.
# Async under the hood but exposed synchronously for simplicity.
# """
# import asyncio
# import logging
# import os
# import tempfile
# import threading

# import edge_tts
# import pygame

# from config import settings

# logger = logging.getLogger(__name__)


# class TextToSpeech:
#     """Converts text to speech using Edge TTS and plays via pygame."""

#     def __init__(self):
#         self.voice = settings.TTS_VOICE
#         self.rate = settings.TTS_RATE
#         self.volume = settings.TTS_VOLUME
#         self._lock = threading.Lock()

#         # Initialize pygame mixer for audio playback
#         pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

#     async def _synthesize(self, text: str, output_path: str):
#         """Async synthesis via edge-tts."""
#         communicate = edge_tts.Communicate(
#             text=text,
#             voice=self.voice,
#             rate=self.rate,
#             volume=self.volume,
#         )
#         await communicate.save(output_path)

#     def speak(self, text: str):
#         """
#         Synthesize text to speech and play it.
#         Blocks until playback is complete.
#         """
#         if not text or not text.strip():
#             return

#         logger.info(f"🔊  Speaking: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

#         with self._lock:
#             # Write to a temp file
#             with tempfile.NamedTemporaryFile(
#                 suffix=".mp3", delete=False
#             ) as tmp:
#                 tmp_path = tmp.name

#             try:
#                 # Run async synthesis synchronously
#                 asyncio.run(self._synthesize(text, tmp_path))

#                 # Play the audio
#                 pygame.mixer.music.load(tmp_path)
#                 pygame.mixer.music.play()

#                 # Wait for playback to finish
#                 while pygame.mixer.music.get_busy():
#                     pygame.time.Clock().tick(10)

#             except Exception as e:
#                 logger.error(f"TTS error: {e}")
#             finally:
#                 # Clean up temp file
#                 pygame.mixer.music.unload()
#                 try:
#                     os.unlink(tmp_path)
#                 except OSError:
#                     pass

#     def speak_async(self, text: str):
#         """Speak in a background thread (non-blocking)."""
#         t = threading.Thread(target=self.speak, args=(text,), daemon=True)
#         t.start()
#         return t


"""
core/tts.py
Text-to-Speech using pyttsx3 — fully offline, no network required.
Works on Windows (SAPI5), Linux (espeak), macOS (nsss) automatically.
No API key, no firewall issues.
"""
import logging
import threading

import pyttsx3

from config import settings

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Offline TTS via pyttsx3.
    Uses Windows SAPI5 voices on Windows — sounds decent out of the box.
    Thread-safe: we serialize all speak() calls through a lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        """Initialize pyttsx3 engine and configure voice/rate/volume."""
        try:
            self._engine = pyttsx3.init()

            # Rate (words per minute) — parse TTS_RATE like "+10%" or "190"
            rate_offset = 0
            try:
                raw = settings.TTS_RATE.replace("%", "").replace("+", "").strip()
                rate_offset = int(raw)
            except (ValueError, AttributeError):
                pass
            current_rate = self._engine.getProperty("rate")
            self._engine.setProperty("rate", max(100, current_rate + rate_offset))

            # Volume
            self._engine.setProperty("volume", 1.0)

            # Voice: prefer a male English voice (closest to JARVIS)
            voices = self._engine.getProperty("voices")
            selected = None

            for v in voices:
                vname = (v.name or "").lower()
                if any(n in vname for n in ("david", "mark", "guy", "zira")):
                    selected = v.id
                    break

            if not selected:
                for v in voices:
                    lang = "".join(str(l) for l in (v.languages or []))
                    if "en" in lang.lower() or "english" in (v.name or "").lower():
                        selected = v.id
                        break

            if selected:
                self._engine.setProperty("voice", selected)
                logger.info(f"TTS engine ready | Voice: {selected}")
            else:
                logger.info("TTS engine ready | Voice: system default")

        except Exception as e:
            logger.error(f"TTS init error: {e}")
            self._engine = None

    def speak(self, text: str):
        """Speak text synchronously (blocks until done). Safe from any thread."""
        if not text or not text.strip():
            return

        logger.info(f"Speaking: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

        if self._engine is None:
            logger.warning("TTS engine not available — skipping speech.")
            return

        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except RuntimeError:
                # Engine was busy — stop, reinit, retry once
                try:
                    self._engine.stop()
                    self._init_engine()
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    logger.error(f"TTS retry error: {e}")
            except Exception as e:
                logger.error(f"TTS error: {e}")

    def speak_async(self, text: str):
        """Speak in a background thread (non-blocking)."""
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t