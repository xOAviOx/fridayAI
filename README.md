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
# 1. Get a venv with deps
uv sync                       # preferred
# or:
python -m venv .venv && .venv\Scripts\activate && pip install -e .

# 2. Configure secrets
cp .env.example .env
# Fill in only the keys for the providers you've selected in config.yaml.
# Phase 0 doesn't actually call any API — it just validates config — but
# missing keys for the *selected* providers will fail loudly here.
```

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

## Layout

```
friday/
  main.py            # entry point (Phase 0: boot + log + exit)
  config.py          # .env + config.yaml loader, validated with pydantic
  utils/logging.py   # structured logging setup + audit logger
  stt/base.py        # STTProvider interface (Phase 1: groq_whisper.py)
  llm/base.py        # LLMProvider interface (Phase 1: groq_provider.py)
  tts/base.py        # TTSProvider interface (Phase 1: elevenlabs.py)
  audio/             # mic capture + playback (Phase 1)
  skills/            # @skill registry + builtins (Phase 1)
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
