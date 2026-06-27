"""Mic capture, push-to-talk hotkey, and TTS playback.

Lazy-import friendly: importing this package does *not* pull in
``sounddevice`` or ``pynput``. The provider classes themselves
import the audio extras inside their constructors, so
``python -m friday.main`` still boots cleanly when the extras
aren't installed.
"""

from friday.audio.capture import MicRecorder
from friday.audio.hotkey import HotkeyController
from friday.audio.playback import Speaker

__all__ = ["HotkeyController", "MicRecorder", "Speaker"]
