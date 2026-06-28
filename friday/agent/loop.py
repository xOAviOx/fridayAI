"""Phase 2 agent loop — streaming TTS, barge-in, non-blocking turns.

Run with::

    python -m friday.agent.loop

Hold ``ctrl+space`` to talk; release when done.  FRIDAY transcribes via
Groq Whisper, streams the reply from the Groq LLM, and begins speaking
through Kokoro as soon as the *first sentence* is ready — not after the
full response is generated.

Press ``ctrl+space`` *again while FRIDAY is speaking* to barge in:
playback cuts immediately, the running turn is cancelled, and a fresh
recording starts right away.

Press ``ctrl+shift+esc`` at any time to quit.

What changed from Phase 1
--------------------------
* **Non-blocking listener thread** — PTT callbacks return immediately;
  all STT / LLM / TTS work happens on a background *turn thread* so the
  hotkey listener is always responsive.
* **Streaming LLM → sentence TTS** — ``GroqLLM.stream_chat()`` feeds
  tokens into a :class:`~friday.utils.sentence.SentenceSplitter`; each
  complete sentence goes to Kokoro before the model has finished
  generating.  First-word latency drops from *full-response time* to
  *first-sentence time*.
* **Barge-in** — PTT press during playback sets ``_turn_cancel`` and
  calls ``speaker.stop()``, which aborts the in-flight ``sd.OutputStream``
  and unblocks the TTS pipeline thread so it exits cleanly.
* **TTS pipeline** — a ``_TTSPipeline`` helper queues synthesised
  sentences for sequential playback without re-opening the stream per
  sentence.

Phase 2 constraints (still synchronous at the turn level)
----------------------------------------------------------
* One turn runs at a time.  A PTT press during an active turn cancels it;
  the *release* then starts a fresh turn on the new audio.
* VAD is plumbed in config but not wired — ``ctrl+space`` PTT is still
  the input trigger.
* Streaming is token-level on the LLM side; Kokoro synthesis of each
  sentence is still CPU-synchronous (one segment per sentence, ~1 s/s).
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from pathlib import Path
from typing import Iterator

from friday.agent.executor import Executor
from friday.agent.monitor import ProactiveMonitor
from friday.agent.router import Router
from friday.agent.safety import SafetyGate
from friday.audio.capture import MicRecorder
from friday.audio.encoding import float32_to_pcm16_bytes
from friday.audio.hotkey import HotkeyController
from friday.audio.playback import Speaker
from friday.config import Config, ConfigError, load_config
from friday.llm.base import ChatMessage
from friday.llm.groq_provider import GroqLLM, ToolCallError
from friday.llm.history import SlidingWindowHistory
from friday.skills import default_registry
from friday.skills import timers as timers_module
from friday.stt.groq_whisper import GroqWhisperSTT
from friday.tts.base import AudioChunk
from friday.tts.kokoro import KokoroTTS
from friday.utils.logging import setup_logging
from friday.utils.sentence import SentenceSplitter
from friday.utils.tokens import BudgetTracker

log = logging.getLogger("friday.agent.loop")

# Inject the real home directory so the LLM picks correct file paths.
_HOME = Path.home()

import datetime as _dt
import random as _random

_NOW = _dt.datetime.now()

SYSTEM_PROMPT = (
    "You are FRIDAY — a voice assistant who sounds like a real person, not a robot. "
    "Talk exactly like a sharp, witty friend texting you back — casual, quick, human. "

    # Filler words & natural rhythm
    "Use natural filler words and speech rhythms: start replies with 'So...', 'Okay so...', "
    "'Yeah,', 'Alright,', 'Oh,', 'Hmm,', 'Well,', 'Right so,' — vary it every time. "
    "Throw in 'like', 'you know', 'basically', 'honestly', 'actually', 'I mean' mid-sentence where it fits naturally. "
    "Use 'um' or 'uh' occasionally when transitioning — not every sentence, just sometimes. "

    # Length & style
    "Keep replies under 20 words unless asked for detail. Use contractions always (don't, it's, I've, you're). "
    "Never say 'Certainly', 'Sure!', 'Of course!', 'Absolutely!', 'Great question', or any robotic opener. "
    "Occasionally call the user 'boss' — naturally, maybe once every few replies. "

    # After tool calls — speak results like a human, never read raw output
    "After every tool result, reply in ONE casual spoken sentence — like you're telling a friend. "
    "CRITICAL — never read raw numbers, code output, or formatted text verbatim. Always convert: "
    "large numbers → natural form ('4.3 billion', 'about a gig', '2 to the 32 is roughly 4 billion'); "
    "decimals → say 'point' ('3 point 14'); "
    "file paths → just the filename or folder ('your Desktop folder'); "
    "stack traces / errors → just the error type and message, nothing else ('got a ZeroDivisionError'); "
    "long lists → summarise ('found 12 files', 'you have 8 packages installed'); "
    "code output → interpret it, don't recite it ('that came out to about 4 billion'). "
    "Trust every tool result — never verify by calling another tool. "
    "If a result starts with [dry_run] treat it as succeeded. "
    "If [needs_confirmation] or [denied], just tell the user plainly. "

    # Capabilities reminder
    "You can remember things, set timers, check weather, search the web, and control the computer. "
    "Just do it — don't ask for permission. "

    # Context
    f"Home: {_HOME}. Desktop: {_HOME}/Desktop. Downloads: {_HOME}/Downloads. "
    "Always use full absolute paths for files. "
    "Current time: " + _NOW.strftime("%I:%M %p, %A %B %d %Y") + "."
)

# Short spoken fillers played right after STT — while the LLM is thinking.
# Sounds like FRIDAY is acknowledging before responding, very human.
_THINKING_FILLERS = [
    "Hmm.",
    "Okay.",
    "Let me check.",
    "Yeah, one sec.",
    "On it.",
    "Alright.",
    "Sure, give me a sec.",
    "Mm-hmm.",
    "Got it.",
    "Right, let me see.",
]

# Fillers for tool-heavy queries (when words like "search", "find", "open" detected).
_ACTION_FILLERS = [
    "On it, boss.",
    "Yeah, pulling that up.",
    "Let me grab that.",
    "Alright, checking now.",
    "One sec.",
    "Got it, looking that up.",
]

_MAX_TOOL_HOPS = 4
_MIN_AUDIO_S = 0.2


# --------------------------------------------------------------------------- #
# TTS pipeline — sentence queue → Kokoro → Speaker (background thread)       #
# --------------------------------------------------------------------------- #


class _TTSPipeline:
    """Queue sentences for TTS synthesis and sequential playback.

    A single background thread drains the sentence queue, calls
    ``tts.synthesize()`` per sentence (producing ``AudioChunk`` chunks),
    and pipes them into ``speaker.play()``.

    ``cancel()`` is safe to call from any thread.  It sets a flag,
    stops the speaker (dropping buffered audio), and inserts a sentinel
    so the background thread exits cleanly.
    """

    def __init__(self, tts: KokoroTTS, speaker: Speaker) -> None:
        self._tts = tts
        self._speaker = speaker
        self._q: queue.Queue[str | None] = queue.Queue()
        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="friday-tts")
        self._thread.start()

    # ------------------------------------------------------------------ #
    # Producer interface (turn thread)                                    #
    # ------------------------------------------------------------------ #

    def push(self, sentence: str) -> None:
        """Add a sentence to be synthesized and played."""
        if not self._cancelled.is_set() and sentence.strip():
            self._q.put(sentence)

    def finish(self) -> None:
        """Signal that no more sentences will be added (this turn)."""
        self._q.put(None)

    def wait(self, timeout: float = 60.0) -> None:
        """Block until the pipeline has played everything queued."""
        self._done.wait(timeout=timeout)

    # ------------------------------------------------------------------ #
    # Cancellation (any thread)                                           #
    # ------------------------------------------------------------------ #

    def cancel(self) -> None:
        """Cut playback immediately and stop the pipeline thread."""
        self._cancelled.set()
        self._speaker.stop()
        # Unblock the queue.get() so the thread can see the cancel flag.
        self._q.put(None)

    # ------------------------------------------------------------------ #
    # Background thread                                                   #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        try:
            self._speaker.play(self._gen_chunks())
        except Exception:
            log.exception("TTS pipeline error")
        finally:
            self._done.set()

    def _gen_chunks(self) -> Iterator[AudioChunk]:
        """Drain the sentence queue, synthesising each sentence on demand."""
        while True:
            sentence = self._q.get()
            if sentence is None or self._cancelled.is_set():
                return
            try:
                yield from self._tts.synthesize(sentence)
            except Exception:
                log.exception("Kokoro synthesis failed for %r", sentence[:40])


# --------------------------------------------------------------------------- #
# Agent loop                                                                  #
# --------------------------------------------------------------------------- #


class AgentLoop:
    """Phase 2: streaming LLM → sentence TTS, barge-in, non-blocking turns."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._quit = threading.Event()

        # ----- providers ------------------------------------------------
        self._stt = GroqWhisperSTT(
            api_key=_require(config.secrets.groq_api_key, "GROQ_API_KEY"),
            model=config.stt.groq.model,
        )
        self._tts = KokoroTTS(
            voice=config.tts.kokoro.voice,
            lang_code=config.tts.kokoro.lang_code,
            speed=config.tts.kokoro.speed,
        )
        if config.providers.llm == "openai_compat":
            budget = BudgetTracker(
                config.rate_limits.get("openai_compat") or _empty_budget(),
                name="openai_compat",
            )
            self._llm = GroqLLM(
                api_key=_require(
                    config.secrets.openai_compat_api_key, "OPENAI_COMPAT_API_KEY"
                ),
                model=config.llm.openai_compat.model,
                base_url=config.llm.openai_compat.base_url,
                budget=budget,
                prompt_tools=True,   # inject schemas as text for non-native endpoints
                no_stream=True,      # fallback for providers without SSE
            )
        else:
            budget = BudgetTracker(
                config.rate_limits.get("groq") or _empty_budget(),
                name="groq",
            )
            self._llm = GroqLLM(
                api_key=_require(config.secrets.groq_api_key, "GROQ_API_KEY"),
                model=config.llm.groq.model,
                base_url=config.llm.groq.base_url,
                budget=budget,
            )

        # ----- audio I/O ------------------------------------------------
        self._recorder = MicRecorder(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
        )
        self._speaker = Speaker()

        # ----- agent layers ---------------------------------------------
        self._registry = default_registry
        self._gate = SafetyGate(config.safety, self._registry)
        self._executor = Executor(self._registry, self._gate)
        self._router = Router()
        self._history = SlidingWindowHistory(system=SYSTEM_PROMPT)
        self._tools = self._registry.tool_schemas()

        # ----- hotkeys --------------------------------------------------
        self._hotkeys = HotkeyController(
            ptt=config.hotkeys.push_to_talk,
            panic=config.hotkeys.panic,
            on_ptt_press=self._on_ptt_press,
            on_ptt_release=self._on_ptt_release,
            on_panic=self._on_panic,
        )

        # ----- proactive monitor (Phase 4) ------------------------------
        self._monitor = ProactiveMonitor(self._speak_sync, config.monitor)

        # ----- Phase 2: turn threading ----------------------------------
        # The listener thread only sets flags / queues audio; the heavy
        # work (STT → LLM → TTS) runs on a daemon turn thread so the
        # listener is always free to process barge-in or panic.
        self._turn_cancel = threading.Event()
        self._turn_thread: threading.Thread | None = None
        # Holds the single pending audio recording if a new PTT release
        # fires while a turn is already running.  Latest wins.
        self._pending_audio: "list | None" = None
        self._pending_lock = threading.Lock()

    # ----- lifecycle --------------------------------------------------------

    def run(self) -> int:
        # Wire the timer / reminder system to this loop's TTS speaker.
        timers_module.set_speak_callback(self._speak_sync)

        self._hotkeys.start()
        self._monitor.start()
        log.info(
            "agent ready (phase 4). hold %s to talk, %s to quit. "
            "barge-in supported. proactive monitor %s. "
            "(brain=%s, ears=%s, mouth=%s, dry_run=%s, streaming=True)",
            self._cfg.hotkeys.push_to_talk,
            self._cfg.hotkeys.panic,
            "ON" if self._cfg.monitor.enabled else "OFF",
            self._cfg.providers.llm,
            self._cfg.providers.stt,
            self._cfg.providers.tts,
            self._cfg.safety.dry_run,
        )

        # Startup announcement — give the TTS stack half a second to
        # warm up, then speak. Non-fatal if it fails.
        import threading as _t
        def _announce() -> None:
            import time as _time
            _time.sleep(0.6)
            try:
                self._speak_sync("Systems online. Ready when you are, boss.")
            except Exception:
                log.debug("startup announcement failed — not critical", exc_info=True)
        _t.Thread(target=_announce, daemon=True, name="friday-announce").start()

        try:
            self._quit.wait()
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            self._teardown()
        return 0

    def _teardown(self) -> None:
        self._hotkeys.stop()
        self._monitor.stop()
        # Cancel any running turn so it exits cleanly.
        self._turn_cancel.set()
        self._speaker.stop()
        if self._turn_thread and self._turn_thread.is_alive():
            self._turn_thread.join(timeout=3.0)
        self._recorder.stop()
        self._tts.close()
        self._stt.close()
        self._llm.close()

    # ----- hotkey callbacks (listener thread — must return quickly) ---------

    def _on_ptt_press(self) -> None:
        log.info("listening… (release %s to stop)", self._cfg.hotkeys.push_to_talk)
        # Barge-in: if a turn is in flight, cancel it and cut audio NOW.
        self._turn_cancel.set()
        self._speaker.stop()
        self._recorder.start()

    def _on_ptt_release(self) -> None:
        audio = self._recorder.stop()
        seconds = len(audio) / self._recorder.sample_rate if len(audio) else 0.0
        if seconds < _MIN_AUDIO_S:
            log.info("captured %.2fs — too short, ignoring", seconds)
            return
        log.info("captured %.2fs of audio", seconds)

        # Store audio for the turn worker.  If the previous turn is
        # still finishing its teardown, the worker will pick this up.
        with self._pending_lock:
            self._pending_audio = audio

        self._maybe_spawn_turn()

    def _on_panic(self) -> None:
        log.warning("panic — aborting everything and quitting")
        self._turn_cancel.set()
        self._speaker.stop()
        self._quit.set()

    # ----- turn thread management -------------------------------------------

    def _maybe_spawn_turn(self) -> None:
        """Start a turn thread if none is running; otherwise let the running
        one pick up ``_pending_audio`` when it finishes."""
        if self._turn_thread is not None and self._turn_thread.is_alive():
            # Running turn will check _pending_audio after it wraps up.
            return
        self._turn_cancel.clear()
        self._turn_thread = threading.Thread(
            target=self._turn_worker, daemon=True, name="friday-turn"
        )
        self._turn_thread.start()

    def _turn_worker(self) -> None:
        """Drain _pending_audio: do STT → LLM → TTS for each utterance."""
        while True:
            with self._pending_lock:
                audio = self._pending_audio
                self._pending_audio = None

            if audio is None:
                return  # nothing to do

            self._turn_cancel.clear()
            try:
                self._process_audio(audio)
            except Exception:
                log.exception("turn worker crashed")
                self._speak_sync("Something went wrong on my end.")

    def _process_audio(self, audio: "Any") -> None:  # noqa: ANN401
        """STT + agent turn for one recording."""
        if self._turn_cancel.is_set():
            return

        # ----- STT -------------------------------------------------------
        try:
            transcript = self._stt.transcribe(
                float32_to_pcm16_bytes(audio),
                sample_rate=self._recorder.sample_rate,
            )
        except Exception as exc:
            log.exception("transcription failed: %s", exc)
            self._speak_sync("Sorry, I couldn't hear that.")
            return

        user_text = (transcript.text or "").strip()
        if not user_text:
            log.info("empty transcript — ignoring")
            return
        log.info("you said: %r", user_text)
        self._history.add(ChatMessage(role="user", content=user_text))

        # ----- Thinking filler (sounds human while LLM warms up) ----------
        # Pick action fillers for "do something" queries, generic for the rest.
        _action_kw = ("search", "find", "open", "play", "get", "check", "set",
                      "remind", "show", "look", "fetch", "weather", "who", "what")
        lower = user_text.lower()
        if any(kw in lower for kw in _action_kw):
            filler = _random.choice(_ACTION_FILLERS)
        else:
            filler = _random.choice(_THINKING_FILLERS)
        self._speak_sync(filler)

        # ----- LLM + tools turn -----------------------------------------
        try:
            self._run_turn()
        except Exception as exc:
            log.exception("turn failed: %s", exc)
            self._speak_sync("Something went wrong on my end.")

    # ----- streaming turn ---------------------------------------------------

    def _run_turn(self) -> None:
        """Stream the LLM response and pipe complete sentences to TTS.

        For each hop:
        1. Open a streaming chat request.
        2. Feed tokens into a :class:`~friday.utils.sentence.SentenceSplitter`.
        3. Each complete sentence goes to a :class:`_TTSPipeline` that
           synthesises and plays it in a background thread.
        4. If the response has tool calls, wait for the TTS to drain,
           execute the tools, then loop back for the next hop.
        5. If the response is terminal (no tool calls), signal the
           pipeline and return.
        """
        for hop in range(_MAX_TOOL_HOPS):
            if self._turn_cancel.is_set():
                log.info("turn cancelled before hop %d", hop)
                return

            # --- open stream ---
            streamed = self._llm.stream_chat(
                self._history.messages(),
                tools=self._tools,
                max_tokens=512,
            )

            # --- TTS pipeline for this hop ---
            pipeline = _TTSPipeline(self._tts, self._speaker)
            splitter = SentenceSplitter()

            # --- drain token stream ---
            try:
                for token in streamed:
                    if self._turn_cancel.is_set():
                        pipeline.cancel()
                        streamed.response  # drain HTTP stream cleanly
                        return
                    for sentence in splitter.push(token):
                        pipeline.push(sentence)
            except ToolCallError:
                # Model failed to produce a valid tool call (too many tools,
                # or a model-specific limitation). Retry this hop without
                # tools — the user gets a plain spoken answer instead of silence.
                log.warning(
                    "hop %d: tool call generation failed — retrying without tools", hop + 1
                )
                pipeline.cancel()
                streamed_plain = self._llm.stream_chat(
                    self._history.messages(),
                    tools=[],        # no tools — forces a plain text reply
                    max_tokens=512,
                )
                pipeline = _TTSPipeline(self._tts, self._speaker)
                splitter = SentenceSplitter()
                for token in streamed_plain:
                    if self._turn_cancel.is_set():
                        pipeline.cancel()
                        return
                    for sentence in splitter.push(token):
                        pipeline.push(sentence)
                for sentence in splitter.flush():
                    pipeline.push(sentence)
                pipeline.finish()
                pipeline.wait()
                self._history.add(ChatMessage(
                    role="assistant",
                    content=streamed_plain.response.content,
                    tool_calls=[],
                ))
                return

            # Flush any trailing fragment (e.g. "Got it!" without a period)
            for sentence in splitter.flush():
                pipeline.push(sentence)
            pipeline.finish()  # no more sentences this hop

            # --- examine the response ---
            response = streamed.response
            self._history.add(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=list(response.tool_calls),
                )
            )

            if response.usage is not None:
                log.debug(
                    "hop %d usage: prompt=%d completion=%d",
                    hop + 1,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            route = self._router.route(response)
            log.info(
                "hop %d → finish=%s tools=%s speech=%r",
                hop + 1,
                response.finish_reason,
                [tc.name for tc in response.tool_calls],
                (response.content or "")[:120],
            )

            if route.is_terminal:
                pipeline.wait()  # let TTS finish before next user turn
                log.info("turn done after %d hop(s)", hop + 1)
                return

            # Tool-calling hop: wait for any speech, then dispatch.
            pipeline.wait()
            if self._turn_cancel.is_set():
                return

            for call in route.tool_calls:
                log.info("tool: %s(%r)", call.name, call.arguments)
                tool_msg = self._executor.run(call)
                log.info("tool result: %s", (tool_msg.content or "")[:120])
                self._history.add(tool_msg)

        # Exhausted all hops without a terminal response.
        # If the last tool result looks like a success, say "Done" instead of
        # an alarm — the action completed, the model just over-thought it.
        log.warning("max tool hops (%d) reached without a final reply", _MAX_TOOL_HOPS)
        last = self._history.messages[-1] if self._history.messages else None
        last_content = (last.content or "") if last else ""
        if last and last.role == "tool" and not last_content.startswith("["):
            self._speak_sync("Done.")
        else:
            self._speak_sync("I'm getting stuck — let me try again.")

    # ----- helpers ----------------------------------------------------------

    def _speak_sync(self, text: str) -> None:
        """Synthesise and play text synchronously (used for error messages)."""
        text = (text or "").strip()
        if not text:
            return
        try:
            self._speaker.play(self._tts.synthesize(text))
        except Exception:
            log.exception("playback failed")


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

    if config.providers.stt != "groq":
        print(
            f"[FATAL] loop expects providers.stt = groq, got {config.providers.stt!r}",
            file=sys.stderr,
        )
        return 2
    if config.providers.llm not in ("groq", "openai_compat"):
        print(
            f"[FATAL] loop expects providers.llm = groq or openai_compat, "
            f"got {config.providers.llm!r}",
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
    from friday.config import RateLimitConfig
    return RateLimitConfig()


# Satisfy the type-checker in _process_audio without pulling numpy at
# module level — the actual type is numpy.ndarray.
from typing import Any  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
