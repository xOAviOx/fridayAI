"""The Phase 1 agent loop — push-to-talk, transcribe, think, act, speak.

Run with::

    python -m friday.agent.loop

Hold ``ctrl+space``, say something, release. FRIDAY transcribes via
Groq Whisper, hands the text to the Groq LLM with the skills registry
as tool schemas, dispatches each tool call through the safety gate +
executor, and speaks the model's final reply through Kokoro. Press
``ctrl+shift+esc`` at any time to quit; pressing it during playback
hard-cuts the audio (the Phase 2 barge-in contract).

What this proves
----------------
The whole Phase 1 spine: mic → STT → LLM (with tool calling) →
safety + skills → TTS, all driven from ``config.yaml``. Switching
brain / STT / TTS is a yaml edit, never a code change.

Acceptance gate from the brief
------------------------------
"Open Spotify and play music" runs end-to-end with verbal
confirmation. Under ``safety.dry_run: true`` (the Phase 1 default)
nothing actually executes — the audit log records the intent and the
LLM produces a confirmation as if the actions had succeeded.

Phase 1 is synchronous on purpose: record fully, then transcribe,
then think, then speak. Streaming / barge-in / VAD belong to Phase 2.
"""

from __future__ import annotations

import logging
import sys
import threading

from friday.agent.executor import Executor
from friday.agent.router import Router
from friday.agent.safety import SafetyGate
from friday.audio.capture import MicRecorder
from friday.audio.encoding import float32_to_pcm16_bytes
from friday.audio.hotkey import HotkeyController
from friday.audio.playback import Speaker
from friday.config import Config, ConfigError, load_config
from friday.llm.base import ChatMessage
from friday.llm.groq_provider import GroqLLM
from friday.llm.history import SlidingWindowHistory
from friday.skills import default_registry
from friday.stt.groq_whisper import GroqWhisperSTT
from friday.tts.kokoro import KokoroTTS
from friday.utils.logging import setup_logging
from friday.utils.tokens import BudgetTracker

log = logging.getLogger("friday.agent.loop")

# Lean system prompt. Every word here costs TPM on every turn — keep
# the instructions short and let the tool descriptions carry the
# capability surface.
SYSTEM_PROMPT = (
    "You are FRIDAY, a voice-controlled PC assistant. Your replies are "
    "spoken aloud, so keep them under 30 words unless the user asks for "
    "detail. You can call tools to act on the user's machine; do so "
    "without asking for confirmation. After a tool call returns, "
    "confirm what you did in one short sentence. If a tool result "
    "starts with [dry_run], treat the action as having succeeded and "
    "tell the user; if it starts with [needs_confirmation] or [denied], "
    "tell the user what blocked it."
)

# Bound the number of LLM ↔ tool round-trips per user turn. The model
# can chain `open_app` → `media_control` → final reply in three hops;
# six is generous and stops a buggy model from running away with the
# RPM budget.
_MAX_TOOL_HOPS = 6

# Discard recordings shorter than this — usually a stuck-key bounce or
# the user changing their mind mid-press.
_MIN_AUDIO_S = 0.2


class AgentLoop:
    """Wires every Phase 1 component into one push-to-talk conversation loop."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._quit = threading.Event()

        # ----- providers ----------------------------------------------------
        # Each provider lazy-imports its heavy deps inside __init__, so
        # any missing extras surface here (clean traceback) rather than
        # mid-conversation.
        self._stt = GroqWhisperSTT(
            api_key=_require(config.secrets.groq_api_key, "GROQ_API_KEY"),
            model=config.stt.groq.model,
        )
        self._tts = KokoroTTS(
            voice=config.tts.kokoro.voice,
            lang_code=config.tts.kokoro.lang_code,
            speed=config.tts.kokoro.speed,
        )
        budget = BudgetTracker(
            config.rate_limits.get("groq")
            or _empty_budget(),
            name="groq",
        )
        self._llm = GroqLLM(
            api_key=_require(config.secrets.groq_api_key, "GROQ_API_KEY"),
            model=config.llm.groq.model,
            base_url=config.llm.groq.base_url,
            budget=budget,
        )

        # ----- audio I/O ----------------------------------------------------
        self._recorder = MicRecorder(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
        )
        self._speaker = Speaker()

        # ----- agent layers -------------------------------------------------
        self._registry = default_registry
        self._gate = SafetyGate(config.safety, self._registry)
        self._executor = Executor(self._registry, self._gate)
        self._router = Router()
        self._history = SlidingWindowHistory(system=SYSTEM_PROMPT)
        # Snapshotting once is fine — the schema set is fixed at boot.
        self._tools = self._registry.tool_schemas()

        # ----- hotkeys ------------------------------------------------------
        self._hotkeys = HotkeyController(
            ptt=config.hotkeys.push_to_talk,
            panic=config.hotkeys.panic,
            on_ptt_press=self._on_ptt_press,
            on_ptt_release=self._on_ptt_release,
            on_panic=self._on_panic,
        )

    # ----- lifecycle --------------------------------------------------------

    def run(self) -> int:
        self._hotkeys.start()
        log.info(
            "agent ready. hold %s to talk, %s to quit. "
            "(brain=%s, ears=%s, mouth=%s, dry_run=%s)",
            self._cfg.hotkeys.push_to_talk,
            self._cfg.hotkeys.panic,
            self._cfg.providers.llm,
            self._cfg.providers.stt,
            self._cfg.providers.tts,
            self._cfg.safety.dry_run,
        )
        try:
            self._quit.wait()
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self._teardown()
        return 0

    def _teardown(self) -> None:
        # Order matters: hotkeys first so no callback fires mid-tear,
        # then cut any in-flight playback, then close clients.
        self._hotkeys.stop()
        self._speaker.stop()
        self._recorder.stop()
        self._tts.close()
        self._stt.close()
        self._llm.close()

    # ----- hotkey callbacks -------------------------------------------------

    def _on_ptt_press(self) -> None:
        log.info("listening… (release %s to stop)", self._cfg.hotkeys.push_to_talk)
        # Stop any TTS still playing so we don't try to talk over the user.
        self._speaker.stop()
        self._recorder.start()

    def _on_ptt_release(self) -> None:
        audio = self._recorder.stop()
        seconds = len(audio) / self._recorder.sample_rate if len(audio) else 0.0
        if seconds < _MIN_AUDIO_S:
            log.info("captured %.2fs — too short, ignoring", seconds)
            return
        log.info("captured %.2fs of audio", seconds)

        # ----- STT ----------------------------------------------------------
        try:
            transcript = self._stt.transcribe(
                float32_to_pcm16_bytes(audio),
                sample_rate=self._recorder.sample_rate,
            )
        except Exception as exc:  # noqa: BLE001 — protect the loop
            log.exception("transcription failed: %s", exc)
            self._speak("Sorry, I couldn't hear that.")
            return

        user_text = (transcript.text or "").strip()
        if not user_text:
            log.info("empty transcript — ignoring")
            return
        log.info("you said: %r", user_text)
        self._history.add(ChatMessage(role="user", content=user_text))

        # ----- LLM + tools turn --------------------------------------------
        try:
            self._run_turn()
        except Exception as exc:  # noqa: BLE001 — protect the loop
            log.exception("turn failed: %s", exc)
            self._speak("Something went wrong on my end.")

    def _on_panic(self) -> None:
        log.warning("panic — aborting playback and quitting")
        self._speaker.stop()
        self._quit.set()

    # ----- turn runner ------------------------------------------------------

    def _run_turn(self) -> None:
        """LLM → router → executor → LLM until the model is done with tools."""
        for hop in range(_MAX_TOOL_HOPS):
            response = self._llm.chat(
                self._history.messages(),
                tools=self._tools,
                max_tokens=512,
            )
            self._history.add(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=list(response.tool_calls),
                )
            )

            route = self._router.route(response)
            if route.has_speech:
                self._speak(route.speech)

            if route.is_terminal:
                if response.usage is not None:
                    log.info(
                        "turn done after %d hop(s); usage prompt=%d completion=%d",
                        hop + 1,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                    )
                return

            for call in route.tool_calls:
                log.info("tool: %s(%r)", call.name, call.arguments)
                tool_msg = self._executor.run(call)
                log.info("tool result: %s", (tool_msg.content or "")[:120])
                self._history.add(tool_msg)

        log.warning(
            "max tool hops (%d) reached without a final reply; ending turn",
            _MAX_TOOL_HOPS,
        )
        self._speak("I'm getting stuck — let me try again.")

    # ----- helpers ----------------------------------------------------------

    def _speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        try:
            self._speaker.play(self._tts.synthesize(text))
        except Exception as exc:  # noqa: BLE001 — log but don't crash
            log.exception("playback failed: %s", exc)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    setup_logging(level=config.logging.level, audit_path=config.logging.audit_path)

    # Hard-fail early on provider mismatches so the user sees a clear
    # error instead of an obscure constructor RuntimeError.
    if config.providers.stt != "groq":
        print(
            f"[FATAL] loop expects providers.stt = groq, got {config.providers.stt!r}",
            file=sys.stderr,
        )
        return 2
    if config.providers.llm != "groq":
        print(
            f"[FATAL] loop expects providers.llm = groq, got {config.providers.llm!r}",
            file=sys.stderr,
        )
        return 2
    if config.providers.tts != "kokoro":
        print(
            f"[FATAL] loop expects providers.tts = kokoro, got {config.providers.tts!r}",
            file=sys.stderr,
        )
        return 2

    return AgentLoop(config).run()


def _require(value: str | None, env_name: str) -> str:
    if value is None or not value.strip():
        raise RuntimeError(f"{env_name} must be set in .env")
    return value


def _empty_budget():
    """Fall-back rate-limit config — no caps, no warnings.

    Used when ``rate_limits.groq`` isn't present in ``config.yaml``.
    The shipped default config has the block, so this is belt-and-
    braces for edited configs.
    """
    from friday.config import RateLimitConfig

    return RateLimitConfig()


if __name__ == "__main__":
    sys.exit(main())
