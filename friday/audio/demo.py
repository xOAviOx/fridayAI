"""End-to-end audio I/O smoke test — no STT, no LLM yet.

Run with::

    python -m friday.audio.demo

Hold the push-to-talk hotkey (default ``ctrl+space``) while you say
something; release to stop recording. The demo then synthesises a
short confirmation line through Kokoro and plays it back on the
default output device. Press the panic combo (default ``ctrl+shift+esc``)
at any time to quit; pressing it during playback cuts the audio
immediately.

What this proves:

* PTT hotkey → mic capture → buffer length we can measure.
* AudioChunk iterator from Kokoro → speaker playback.
* Panic key actually aborts in-flight TTS.

Everything Phase 1 needs upstream of STT.
"""

from __future__ import annotations

import logging
import sys
import threading

from friday.audio.capture import MicRecorder
from friday.audio.hotkey import HotkeyController
from friday.audio.playback import Speaker
from friday.config import ConfigError, load_config
from friday.tts.kokoro import KokoroTTS
from friday.utils.logging import setup_logging

log = logging.getLogger("friday.audio.demo")


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    setup_logging(level=config.logging.level, audit_path=None)

    if config.providers.tts != "kokoro":
        print(
            f"[FATAL] demo expects providers.tts = kokoro, got {config.providers.tts!r}",
            file=sys.stderr,
        )
        return 2

    kokoro_cfg = config.tts.kokoro
    tts = KokoroTTS(
        voice=kokoro_cfg.voice,
        lang_code=kokoro_cfg.lang_code,
        speed=kokoro_cfg.speed,
    )
    speaker = Speaker()
    recorder = MicRecorder(
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
    )

    # Single event drives the main-thread wait loop; panic sets it.
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
        line = f"Got it. I captured {seconds:.1f} seconds of audio."
        # Synthesise + play synchronously: Phase 1 is record-then-think-then-speak.
        speaker.play(tts.synthesize(line))

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
        "audio demo ready. hold %s to talk, %s to quit.",
        config.hotkeys.push_to_talk,
        config.hotkeys.panic,
    )

    try:
        # Wait forever until panic or Ctrl+C. The hotkey listener runs
        # on its own thread, so the main thread just parks here.
        quit_event.wait()
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        speaker.stop()
        recorder.stop()
        hotkeys.stop()
        tts.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
