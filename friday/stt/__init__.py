"""Speech-to-text providers (the "ears").

``GroqWhisperSTT`` is re-exported lazily — the module body imports the
``openai`` client at construction time, so referencing the class name
without the ``stt-groq`` extras installed is fine; only actually
instantiating it requires the extras.
"""

from friday.stt.base import STTProvider, Transcript
from friday.stt.groq_whisper import GroqWhisperSTT

__all__ = ["GroqWhisperSTT", "STTProvider", "Transcript"]
