# """
# core/stt.py
# Speech-to-Text using Groq's Whisper API.
# Records audio via sounddevice, streams it to Groq for transcription.
# No GPU required — Groq handles everything in the cloud.
# """
# import io
# import logging
# import time
# import numpy as np
# import sounddevice as sd
# from scipy.io import wavfile
# from groq import Groq

# from config import settings

# logger = logging.getLogger(__name__)


# class SpeechToText:
#     """Records microphone audio and transcribes via Groq Whisper."""

#     def __init__(self):
#         self.client = Groq(api_key=settings.GROQ_API_KEY)
#         self.sample_rate = settings.RECORD_SAMPLE_RATE
#         self.channels = settings.RECORD_CHANNELS
#         self.silence_threshold = settings.SILENCE_THRESHOLD
#         self.silence_duration = settings.SILENCE_DURATION
#         self.max_seconds = settings.MAX_RECORD_SECONDS

#     def _record_until_silence(self) -> np.ndarray:
#         """
#         Records audio until the user stops speaking (silence detected),
#         or until MAX_RECORD_SECONDS is hit.
#         Returns raw numpy audio array.
#         """
#         logger.debug("🎙️  Recording started — listening for speech...")
#         frames = []
#         silent_chunks = 0
#         speaking_started = False

#         # chunk size: 100ms worth of samples
#         chunk_size = int(self.sample_rate * 0.1)
#         max_chunks = int(self.max_seconds / 0.1)
#         silence_chunks_needed = int(self.silence_duration / 0.1)

#         with sd.InputStream(
#             samplerate=self.sample_rate,
#             channels=self.channels,
#             dtype="float32",
#         ) as stream:
#             for _ in range(max_chunks):
#                 chunk, _ = stream.read(chunk_size)
#                 frames.append(chunk.copy())

#                 amplitude = np.abs(chunk).mean()

#                 if amplitude > self.silence_threshold:
#                     speaking_started = True
#                     silent_chunks = 0
#                 else:
#                     if speaking_started:
#                         silent_chunks += 1
#                         if silent_chunks >= silence_chunks_needed:
#                             logger.debug("🔇  Silence detected — stopping recording.")
#                             break

#         if not frames:
#             return np.array([], dtype=np.float32)

#         audio = np.concatenate(frames, axis=0)
#         return audio

#     def _numpy_to_wav_bytes(self, audio: np.ndarray) -> bytes:
#         """Convert float32 numpy array to WAV bytes in memory."""
#         # Convert to int16 for WAV format
#         audio_int16 = (audio * 32767).astype(np.int16)
#         buf = io.BytesIO()
#         wavfile.write(buf, self.sample_rate, audio_int16)
#         buf.seek(0)
#         return buf.read()

#     def listen(self) -> str | None:
#         """
#         Full pipeline: record → convert → transcribe.
#         Returns transcribed text or None if nothing was captured.
#         """
#         audio = self._record_until_silence()

#         if audio.size < self.sample_rate * 0.3:  # Less than 0.3s of audio
#             logger.debug("Audio too short, skipping transcription.")
#             return None

#         wav_bytes = self._numpy_to_wav_bytes(audio)

#         try:
#             logger.debug("📡  Sending audio to Groq Whisper...")
#             transcription = self.client.audio.transcriptions.create(
#                 file=("audio.wav", wav_bytes, "audio/wav"),
#                 model=settings.GROQ_WHISPER_MODEL,
#                 response_format="text",
#                 language="en",
#             )
#             text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
#             if text:
#                 logger.info(f"🗣️  Heard: \"{text}\"")
#             return text or None

#         except Exception as e:
#             logger.error(f"STT error: {e}")
#             return None


"""
core/stt.py
Speech-to-Text via Groq Whisper API.

Speed improvements over v1:
  - Pre-speech buffer: captures audio BEFORE you finish your first syllable
  - Adaptive silence detection: stops recording faster after speech ends
  - Minimum audio gate: skips Groq API call for very short clips (noise)
  - Sends smaller WAV directly — no temp file overhead
"""
import io
import logging
import time
from collections import deque

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from groq import Groq

from config import settings

logger = logging.getLogger(__name__)

# Chunk duration in seconds — smaller = more responsive VAD
CHUNK_DURATION   = 0.06    # 60ms chunks (was 100ms)
# How many chunks to keep before speech starts (pre-roll buffer)
PRE_ROLL_CHUNKS  = 5       # 300ms of audio before trigger
# RMS silence threshold — audio below this is silence
# (overridden by settings.SILENCE_THRESHOLD)
DEFAULT_THRESHOLD = 0.025


class SpeechToText:
    def __init__(self):
        self.client        = Groq(api_key=settings.GROQ_API_KEY)
        self.sample_rate   = settings.RECORD_SAMPLE_RATE
        self.channels      = settings.RECORD_CHANNELS
        self.threshold     = settings.SILENCE_THRESHOLD
        self.max_seconds   = settings.MAX_RECORD_SECONDS

        # Adaptive silence: shorter wait after longer speech
        self._base_silence = settings.SILENCE_DURATION   # e.g. 1.5s
        self._min_silence  = 0.6                          # never less than 0.6s

        self._chunk_size   = int(self.sample_rate * CHUNK_DURATION)

    # ── Audio helpers ─────────────────────────────────────────────────────────

    def _rms(self, chunk: np.ndarray) -> float:
        return float(np.sqrt(np.mean(chunk ** 2)))

    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, self.sample_rate, audio_int16)
        return buf.getvalue()

    # ── Recording with VAD ────────────────────────────────────────────────────

    def _record(self) -> np.ndarray | None:
        """
        Record until silence using a rolling pre-roll buffer.
        Returns numpy audio array or None if nothing was captured.
        """
        pre_roll     = deque(maxlen=PRE_ROLL_CHUNKS)   # ring buffer before speech
        speech_chunks = []
        speaking      = False
        silent_time   = 0.0
        total_time    = 0.0
        speech_time   = 0.0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self._chunk_size,
            latency="low",                              # request low-latency device mode
        ) as stream:
            while total_time < self.max_seconds:
                chunk, _ = stream.read(self._chunk_size)
                chunk = chunk.copy()
                total_time  += CHUNK_DURATION
                level = self._rms(chunk)

                if not speaking:
                    pre_roll.append(chunk)
                    if level > self.threshold:
                        # Speech started — include pre-roll so we don't lose first syllable
                        speaking = True
                        speech_chunks.extend(list(pre_roll))
                        pre_roll.clear()
                        logger.debug("🎙️  Speech detected")
                else:
                    speech_chunks.append(chunk)
                    speech_time += CHUNK_DURATION

                    if level < self.threshold:
                        silent_time += CHUNK_DURATION
                        # Adaptive silence cutoff — shorter for longer utterances
                        dynamic_silence = max(
                            self._min_silence,
                            self._base_silence - (speech_time * 0.1)
                        )
                        if silent_time >= dynamic_silence:
                            logger.debug(f"🔇  Silence {silent_time:.2f}s — stopping")
                            break
                    else:
                        silent_time = 0.0   # reset on any sound

        if not speech_chunks:
            return None

        audio = np.concatenate(speech_chunks, axis=0).flatten()

        # Gate: ignore very short clips (< 0.4s) — likely noise/lip smacks
        if len(audio) < self.sample_rate * 0.4:
            logger.debug("Audio too short — skipping")
            return None

        return audio

    # ── Transcription ─────────────────────────────────────────────────────────

    def listen(self) -> str | None:
        """
        Record → transcribe via Groq Whisper.
        Returns text or None.
        """
        t0    = time.perf_counter()
        audio = self._record()

        if audio is None:
            return None

        record_ms = (time.perf_counter() - t0) * 1000
        wav_bytes = self._to_wav_bytes(audio)

        try:
            t1 = time.perf_counter()
            result = self.client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes, "audio/wav"),
                model=settings.GROQ_WHISPER_MODEL,
                response_format="text",
                language="en",
            )
            api_ms = (time.perf_counter() - t1) * 1000

            text = result.strip() if isinstance(result, str) else result.text.strip()

            if text:
                logger.info(
                    f"🗣️  Heard: \"{text}\" "
                    f"[record={record_ms:.0f}ms api={api_ms:.0f}ms]"
                )
            return text or None

        except Exception as e:
            logger.error(f"STT error: {e}")
            return None