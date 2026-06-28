# FRIDAY

Voice-controlled PC assistant. Push a hotkey, talk, it actually does the thing.

> **Status:** Phase 0 scaffold. Boots, loads config, prints "ready", exits.
> The voice loop lands in Phase 1.

## Architecture (one-paragraph version)

Every request is routed through three layers, fastest/safest first:

1. **Skill registry** — hand-written Python functions for the common stuff
   (open app, search, media, type, system info). ~80% of requests land here.
2. **Code execution (sandboxed)** — the LLM writes and runs Python/bash for
   open-ended tasks. Gated behind confirmation. Off by default.
3. **Computer use (vision + mouse/keyboard)** — last-resort fallback for GUI
   tasks with no API. Off by default.

The whole thing is one Python daemon for now. A HUD comes much later (Phase 6).

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) recommended (falls back to `venv` + `pip`)
- Windows / macOS / Linux

## Setup

```bash
# 1. Get a venv with the base deps (config + logging only)
uv sync                       # preferred
# or:
python -m venv .venv && .venv\Scripts\activate && pip install -e .

# 2. (Optional) Add the local TTS extras for the default $0 voice path
uv sync --extra tts-kokoro    # preferred
# or:
pip install -e ".[tts-kokoro]"
# First synthesis will download ~330 MB of Kokoro weights into the
# HuggingFace cache. Subsequent runs are offline.

# 3. Configure secrets
cp .env.example .env
# Out of the box you only need GROQ_API_KEY — the default stack is
# Groq (STT + LLM) + Kokoro (local TTS), and Kokoro needs no key.
# Missing keys for the *selected* providers fail loudly at boot.
```

The default config is `providers.tts: kokoro` (local, free). To switch to
cloud streaming TTS, set `providers.tts: elevenlabs` in `config.yaml` and
put `ELEVENLABS_API_KEY` in `.env`.

## Run

```bash
python -m friday.main
```

You should see something like:

```
12:34:56 | INFO    | friday | FRIDAY ready.
12:34:56 | INFO    | friday |   brain (LLM): groq
12:34:56 | INFO    | friday |   ears (STT):  groq
12:34:56 | INFO    | friday |   mouth (TTS): elevenlabs
12:34:56 | INFO    | friday |   dry-run:     True
12:34:56 | INFO    | friday |   code-exec:   False
12:34:56 | INFO    | friday |   computer-use:False
12:34:56 | INFO    | friday | Phase 0 scaffold: no event loop yet — exiting cleanly.
```

Process exits with code 0. Config errors exit with code 2 and a clear message.

## Smoke-test the TTS

Phase 0 doesn't wire TTS into the runtime loop yet, but you can drive the
provider directly to confirm Kokoro is working end-to-end. After
`uv sync --extra tts-kokoro`, drop this in `scratch_tts.py` and run it:

```python
import wave
from friday.config import load_config
from friday.tts import KokoroTTS

cfg = load_config().tts.kokoro
tts = KokoroTTS(voice=cfg.voice, lang_code=cfg.lang_code, speed=cfg.speed)

chunks = list(tts.synthesize("FRIDAY here. Local voice, zero dollars."))
print(f"got {len(chunks)} chunk(s), sr={chunks[0].sample_rate}")

with wave.open("out.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)               # 16-bit PCM
    w.setframerate(chunks[0].sample_rate)
    for c in chunks:
        w.writeframes(c.pcm)
print("wrote out.wav")
```

```bash
python scratch_tts.py
# → got N chunk(s), sr=24000
# → wrote out.wav
```

Open `out.wav` in any audio player; if you hear the line, the provider,
config, and PCM conversion are all healthy.

## Try the audio I/O loop

Once you've also installed the audio extras you can drive the full
hotkey + mic + Kokoro + speakers chain end-to-end, no STT or LLM yet:

```bash
# Adds sounddevice + pynput + numpy
uv sync --extra audio --extra tts-kokoro

python -m friday.audio.demo
```

Hold `ctrl+space` and say anything; release to stop. FRIDAY will
synthesise "Got it. I captured N seconds of audio." through Kokoro
and play it back through your default output device. Press
`ctrl+shift+esc` to quit — pressing it during playback cuts the
audio immediately, which is the panic-key behaviour Phase 1 needs.

Linux/macOS users: install PortAudio first (`brew install portaudio`
or `apt install libportaudio2`). The Windows `sounddevice` wheel
bundles it.

## Try the STT loop (voice in, your own words out)

Chunk 2 wires Groq Whisper in behind the same hotkey. Hold
`ctrl+space`, say something, release — FRIDAY transcribes your speech
on Groq and speaks the transcript back through Kokoro.

```bash
# Adds the openai SDK on top of the audio + tts-kokoro extras
uv sync --extra audio --extra tts-kokoro --extra stt-groq
# or:
pip install -e ".[audio,tts-kokoro,stt-groq]"

# Make sure GROQ_API_KEY is set in .env
python -m friday.stt.demo
```

Expected log lines per utterance:

```
listening… (release ctrl+space to stop)
captured 2.40s of audio
transcribing 76800 bytes via Groq Whisper…
you said: 'Hello FRIDAY, can you hear me?'  (2.40s audio, 412 ms RTT)
```

If the transcript looks right and the echo plays back, the full
mic → STT → TTS spine is healthy. The LLM brain plugs in next (chunk 5)
and replaces the echo with an actual response.

## Skills (the tools the LLM can call)

`friday/skills/registry.py` defines a `@skill` decorator that turns any
typed Python function into an LLM-callable tool. Schemas are derived
from the signature + docstring — there are zero hand-written JSON
schemas in this codebase.

```python
from friday.skills import skill

@skill
def open_app(name: str) -> str:
    """Open the named application on the user's machine.

    Parameters
    ----------
    name:
        Application name like "spotify" or "chrome".
    """
    ...
```

Five starter skills ship in `friday/skills/builtin.py`: `open_app`,
`web_search`, `media_control`, `system_info`, `type_text`. They're
auto-registered in `default_registry` when the package is imported.

`type_text` is the only one marked `destructive=True` — the safety
layer (chunk 4) will use that flag to require explicit confirmation
before it runs.

```bash
# Install the runtime deps only if you actually want to execute skills:
uv sync --extra skills    # adds pyautogui + psutil
# or:
pip install -e ".[skills]"
```

The schema generator is covered by `tests/test_skill_schema.py`:

```bash
python -m pytest tests/ -v
```

## Safety gate

Every tool call FRIDAY proposes routes through `friday/agent/safety.py`
before anything actually runs. `SafetyGate.evaluate(skill_name, args)`
returns one of four decisions:

- `allow` — execute the skill.
- `dry_run` — `safety.dry_run` is on; log the intent, do nothing.
- `needs_confirmation` — destructive skill outside dry-run, or an
  off-allowlist `open_app`/shell command. The executor (chunk 6) is
  responsible for prompting the user.
- `deny` — unknown skill, empty shell command, or open_app with no
  name. Hard refuse.

Every decision is appended as one JSON line to `friday_audit.log`:

```json
{"ts": "2026-06-28T06:07:16+00:00", "event": "skill_invoke", "skill": "open_app",
 "arguments": {"name": "notion"}, "decision": "needs_confirmation",
 "reason": "app 'notion' is not in app_allowlist",
 "app": "notion", "allowlist": ["chrome", "code", "firefox", "notepad", "spotify"]}
```

`destructive=True` is the marker the gate reads off each skill — only
`type_text` carries it among the five builtins. Off-allowlist apps
get the same `needs_confirmation` treatment.

Defaults in `config.yaml` are paranoid: `dry_run: true`,
`enable_code_exec: false`, `enable_computer_use: false`. Keep them on
throughout Phase 1.

## LLM brain (Groq chat with tool calling)

`friday/llm/groq_provider.py` implements the `LLMProvider` interface
against Groq's OpenAI-SDK-compatible chat-completions endpoint. Tool
schemas come from `default_registry.tool_schemas()`; the provider
translates `ChatMessage` ↔ OpenAI shape in both directions, so the
agent loop never touches a raw OpenAI dict and never JSON-decodes
tool-call arguments by hand.

Free-tier discipline is built in via `friday/utils/tokens.py`. The
`BudgetTracker` you pass to `GroqLLM(budget=...)` enforces the limits
declared in `config.yaml`'s `rate_limits.groq` block:

```yaml
rate_limits:
  groq:
    requests_per_min: 30
    tokens_per_min: 6000        # the real bottleneck on free tier
    requests_per_day: 14400
    warn_at_pct: 80
```

What the tracker does on each call:

- Pre-flight: blocks (sleeps) when RPM or TPM is exhausted; raises
  `BudgetExceededError` only on the daily-requests cap (sleeping
  doesn't help there).
- Warns at `warn_at_pct` for every meter that's projected over the
  threshold after this call.
- On 429: honors the server's `retry-after` header verbatim
  (capped at 30 s), then retries — bounded to three attempts.

Conversation state lives in `friday/llm/history.py`. The
`SlidingWindowHistory` keeps a sticky system message plus the last
N user turns (default 8), preserving tool-call/tool-result pairs as
units so the window never splits one. The agent loop owns the
history instance; the provider is stateless.

```bash
# Install the LLM extras (same openai SDK as stt-groq)
uv sync --extra llm-groq
# or:
pip install -e ".[llm-groq]"
```

## Layout

```
friday/
  main.py            # entry point (Phase 0: boot + log + exit)
  config.py          # .env + config.yaml loader, validated with pydantic
  utils/logging.py   # structured logging setup + audit logger
  stt/base.py        # STTProvider interface
  stt/groq_whisper.py # Groq Whisper STT (default — OpenAI-SDK compatible)
  stt/demo.py        # `python -m friday.stt.demo` — mic → Groq Whisper → Kokoro echo
  llm/base.py        # LLMProvider interface (Phase 1: groq_provider.py)
  tts/base.py        # TTSProvider interface
  tts/kokoro.py      # local CPU TTS (default — no API key, no cost)
  audio/capture.py   # mic recorder driven by PTT start/stop
  audio/playback.py  # AudioChunk iterator → speakers (with stop()/barge-in)
  audio/hotkey.py    # global PTT + panic hotkeys via pynput
  audio/encoding.py  # float32 → 16-bit PCM bytes helper for STT
  audio/demo.py      # `python -m friday.audio.demo` — end-to-end I/O smoke test
  audio/             # mic capture + playback (Phase 1)
  skills/registry.py # @skill decorator + SkillRegistry + JSON-schema generator
  skills/builtin.py  # the five starter skills (open_app, web_search, …)
  agent/safety.py    # SafetyGate: dry-run + allowlists + destructive flag + audit
  llm/groq_provider.py # Groq chat with tool calling + budget-tracked retries
  llm/history.py     # SlidingWindowHistory: sticky system + last N user turns
  utils/tokens.py    # BudgetTracker: RPM/TPM/RPD gating + 429 retry-after
  agent/             # loop, router, executor, safety (Phase 1+)
  codeexec/          # sandboxed code execution (Phase 3)
  computeruse/       # screenshot + pyautogui driver (Phase 4)
```

## Phase plan

- **Phase 0** — scaffold (you are here)
- **Phase 1** — MVP loop: push-to-talk → Whisper → Groq + tools → ElevenLabs, 5 starter skills, dry-run on
- **Phase 2** — latency & feel: streaming STT/LLM/TTS, barge-in, panic key, optional VAD
- **Phase 3** — more skills + gated code-exec + full safety system
- **Phase 4** — computer-use fallback via Gemini vision
- **Phase 5** — wake word ("Hey FRIDAY")
- **Phase 6** — Electron HUD overlay

## Safety, in one line

`safety.dry_run: true` means nothing actually executes — every intended action
is logged to `friday_audit.log`. Keep it on while you're testing.
