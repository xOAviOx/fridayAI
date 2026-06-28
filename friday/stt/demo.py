"""End-to-end mic → Groq Whisper → Kokoro echo demo.

Run with::

    python -m friday.stt.demo

Hold the push-to-talk hotkey (default ``ctrl+space``), say something,
release. FRIDAY records the audio, ships it to Groq Whisper, and reads
the transcription back to you through Kokoro. Press ``ctrl+shift+esc``
to quit — pressing it during playback cuts the audio immediately.

What this proves:

* PTT hotkey → mic capture (already validated in ``audio.demo``).
* Float32 mic buffer → PCM16 → in-memory WAV → Groq Whisper round-trip.
* Real transcript text → Kokoro → speakers.
* Panic key still aborts in-flight TTS.

This is the "voice in, your own words out" milestone for Phase 1 chunk 2.
The LLM brain (chunk 5) replaces the echo with an actual response, but
the spine is the same.
"""

from __future__ import annotations

import logging
import sys
import threading

from friday.audio.capture import MicRecorder
from friday.audio.encoding import float32_to_pcm16_bytes
from friday.audio.hotkey import HotkeyController
from friday.audio.playback import Speaker
from friday.config import ConfigError, load_config
from friday.stt.groq_whisper import GroqWhisperSTT
from friday.tts.kokoro import KokoroTTS
from friday.utils.logging import setup_logging

log = logging.getLogger("friday.stt.demo")


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    setup_logging(level=config.logging.level, audit_path=None)

    if config.providers.stt != "groq":
        print(
            f"[FATAL] demo expects providers.stt = groq, got {config.providers.stt!r}",
            file=sys.stderr,
        )
        return 2
    if config.providers.tts != "kokoro":
        print(
            f"[FATAL] demo expects providers.tts = kokoro, got {config.providers.tts!r}",
            file=sys.stderr,
        )
        return 2

    # load_config() already enforced this for providers.stt = groq, but
    # narrow the type for the call below.
    api_key = config.secrets.groq_api_key
    assert api_key is not None

    stt = GroqWhisperSTT(api_key=api_key, model=config.stt.groq.model)
    tts = KokoroTTS(
        voice=config.tts.kokoro.voice,
        lang_code=config.tts.kokoro.lang_code,
        speed=config.tts.kokoro.speed,
    )
    speaker = Speaker()
    recorder = MicRecorder(
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
    )

    quit_event = threading.Event()

    def on_ptt_press() -> None:
        log.info("listening… (release %s to stop)", config.hotkeys.push_to_talk)
        recorder.start()

    def on_ptt_release() -> None:
        audio = recorder.stop()
        seconds = len(audio) / recorder.sample_rate if len(audio) else 0.0
        log.info("captured %.2fs of audio", seconds)
        if seconds < 0.2:
            log.info("nothing meaningful captured — try again")
            return

        pcm = float32_to_pcm16_bytes(audio)
        log.info("transcribing %d bytes via Groq Whisper…", len(pcm))
        try:
            transcript = stt.transcribe(pcm, sample_rate=recorder.sample_rate)
        except Exception as exc:  # noqa: BLE001 — demo guard
            log.exception("transcription failed: %s", exc)
            return

        text = (transcript.text or "").strip()
        if not text:
            log.info("transcript was empty — try again, a little louder.")
            return

        log.info(
            "you said: %r  (%.2fs audio, %d ms RTT)",
            text,
            transcript.duration_s or 0.0,
            transcript.latency_ms or 0,
        )
        speaker.play(tts.synthesize(f"You said: {text}"))

    def on_panic() -> None:
        log.warning("panic — aborting playback and quitting")
        speaker.stop()
        quit_event.set()

    hotkeys = HotkeyController(
        ptt=config.hotkeys.push_to_talk,
        panic=config.hotkeys.panic,
        on_ptt_press=on_ptt_press,
        on_ptt_release=on_ptt_release,
        on_panic=on_panic,
    )
    hotkeys.start()

    log.info(
        "stt demo ready. hold %s to talk, %s to quit.",
        config.hotkeys.push_to_talk,
        config.hotkeys.panic,
    )

    try:
        quit_event.wait()
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        speaker.stop()
        recorder.stop()
        hotkeys.stop()
        tts.close()
        stt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
