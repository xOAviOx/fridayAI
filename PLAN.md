# FRIDAY — Build Plan

> Living document. Supersedes the original brief where they conflict.
> Specifically: **Kokoro is the TTS provider — ElevenLabs is out**.

---

## Project status

- **Phase 0 (scaffold):** done. `python -m friday.main` boots, loads
  config (`.env` + `config.yaml` via pydantic), logs readiness, exits clean.
- **Phase 1 (MVP loop):** partial. Only the TTS provider is implemented
  (`friday/tts/kokoro.py`). Everything else listed below is still to build.

---

## Approved deviation from the original brief: Kokoro replaces ElevenLabs

The original spec had ElevenLabs as the Phase 1 TTS and Kokoro as a
"Phase 2+" alternative. That ordering is reversed and the cloud option
is dropped from the plan. **Kokoro is the TTS provider — both the default
and the only one we're actively building toward.** Anywhere the original
brief says "ElevenLabs," read "Kokoro."

What this means concretely:

- `providers.tts: kokoro` is the default in `config.yaml`.
- `friday/tts/kokoro.py` implements `TTSProvider` against `kokoro>=0.9.4`,
  lazy-imported via the `tts-kokoro` optional dependency group.
- Default voice `af_heart`, `lang_code: a` (American English), 24 kHz mono
  int16 PCM output. One `AudioChunk` per Kokoro segment; `stop()`
  short-circuits the iterator between segments for future barge-in.
- Kokoro needs no API key — the secret check in `_REQUIRED_SECRETS`
  correctly omits it, so a missing `ELEVENLABS_API_KEY` no longer blocks
  boot.
- The ElevenLabs config block still exists in `config.yaml` as an inert
  opt-in for anyone who wants it later. **Do not build it out** unless
  this decision is explicitly reversed.

---

## Tech stack (fixed)

- Python 3.11+, managed with `uv` (fallback: venv + pip)
- **Audio:** `sounddevice` (mic + speaker), `pynput` (global push-to-talk
  + panic hotkey)
- **STT:** Groq Whisper `whisper-large-v3-turbo` via OpenAI-SDK-compatible
  client. Base URL `https://api.groq.com/openai/v1`.
- **LLM (brain):** Groq `llama-3.3-70b-versatile` default. OpenAI-SDK
  compatible. Gemini and Anthropic swap purely via `config.yaml` /
  env vars — no code rewrites.
- **TTS:** Kokoro (local, free, default). See deviation above.
- **PC control:** `pyautogui` / `subprocess` / OS APIs inside skill
  functions.
- **Config:** `.env` for secrets only, `config.yaml` for everything else.

---

## Free-tier discipline (built in from day one)

- Groq free tier: ~30 req/min, **~6000 TPM (the real bottleneck)**,
  ~14400 req/day.
- Lean system prompt + tool schemas — they cost TPM every turn.
- Prompt caching on the stable system prompt where the provider
  supports it.
- Sliding-window history (last N turns); summarize older context if
  needed.
- Per-call token tracking, warn at 80% TPM, exponential backoff that
  respects the `retry-after` header on 429s.

---

## Phase 1 build order

Six chunks, each commit-sized and individually testable. Stop after each
for user testing before moving on.

1. **`friday/audio/`** — `capture.py` (sounddevice mic + PTT via pynput),
   `playback.py` (drain `Iterator[AudioChunk]` to speakers), `hotkey.py`
   (PTT `ctrl+space` + panic `ctrl+shift+esc`). Demo:
   `python -m friday.audio.demo` records, then echoes back through Kokoro
   saying "you said something." New extras group:
   `audio = ["sounddevice>=0.4", "pynput>=1.7", "numpy>=1.26"]`.
   **THIS IS THE NEXT CHUNK.**
2. **`friday/stt/groq_whisper.py`** — Groq Whisper impl behind the
   existing `STTProvider` interface.
3. **`friday/skills/registry.py` + `builtin.py`** — `@skill` decorator
   with auto JSON-schema generation from signature + docstring. Five
   starter skills: `open_app`, `web_search` (browser results),
   `media_control` (play/pause/next/volume), `system_info`
   (battery/time/cpu), `type_text`. Write schema-generation tests
   alongside — the spec calls these out by name.
4. **`friday/agent/safety.py`** — dry-run gate, shell + app allowlists,
   `destructive: bool` flag enforcement, audit log to
   `friday_audit.log`. Built **before** the loop so safety wires in from
   the start (the spec is emphatic: safety is not bolted on at the end).
5. **`friday/llm/groq_provider.py`** — Groq chat with tool-calling,
   sliding-window history. `friday/utils/tokens.py` (usage tracking +
   429 retry-after backoff) lands with it.
6. **`friday/agent/loop.py` + `router.py` + `executor.py`** —
   orchestration. Done when "open Spotify and play music" runs the full
   loop end-to-end with verbal confirmation.

---

## Phase 1 constraints

- **Synchronous in Phase 1.** Record fully, then transcribe, then think,
  then speak. Streaming / barge-in / VAD belong in Phase 2. Kokoro
  already yields per-segment so the streaming-TTS work in Phase 2 is
  mostly on the playback side.
- **Dry-run by default** during all Phase 1 testing
  (`safety.dry_run: true`).
- **Tests required for two specific things**: the router and the
  `@skill` schema generator. Spec is specific; no other tests are
  required at this stage.
- **Safety is first-class.** Built before the agent loop, not after.
- **Providers genuinely swappable.** Switching brain / STT / TTS is a
  `config.yaml` change, never a code rewrite.

---

## Phase plan rolling forward

- **Phase 2** — streaming STT, streaming LLM tokens, streaming TTS,
  barge-in, panic key actually wired, optional VAD.
- **Phase 3** — more skills + gated sandboxed code-exec + full safety
  system (confirmation prompts, allowlist enforcement, audit-log review).
- **Phase 4** — computer-use fallback via Gemini 2.5 Flash vision +
  `pyautogui`. Last-resort routing only.
- **Phase 5** — wake word (openWakeWord or Porcupine), "Hey FRIDAY".
- **Phase 6** (optional, far later) — Electron HUD overlay, dark /
  premium, thin client over the Python daemon via local IPC / websocket.

---

## Working agreements

- One chunk at a time. Stop after each, tell the user how to test, wait
  for go-ahead.
- No co-author trailers on commits.
- Commit author: `xOAviOx <avishuklacode@gmail.com>`.
- Remote: `git@github.com:xOAviOx/fridayAI.git`, `main` branch.
- Secrets only in `.env`; never commit keys.
- Default to the cheapest / fastest model that works for each turn.
  Opus escalation path stays wired but disabled until
  `ANTHROPIC_API_KEY` is present **and** `llm.anthropic.enabled = true`.

---

## File-by-file status

```
friday/
  main.py              # DONE   — Phase 0 boot + log + exit
  config.py            # DONE   — pydantic-validated .env + config.yaml loader
  utils/logging.py     # DONE   — structured logger + audit helper
  utils/tokens.py      # TODO   — usage tracking + 429 backoff (lands with chunk 5)
  stt/base.py          # DONE   — interface only
  stt/groq_whisper.py  # TODO   — chunk 2
  llm/base.py          # DONE   — interface only
  llm/groq_provider.py # TODO   — chunk 5
  tts/base.py          # DONE
  tts/kokoro.py        # DONE   — default local TTS
  audio/               # TODO   — chunk 1 (NEXT)
  skills/              # TODO   — chunk 3
  agent/safety.py      # TODO   — chunk 4
  agent/loop.py        # TODO   — chunk 6
  agent/router.py      # TODO   — chunk 6
  agent/executor.py    # TODO   — chunk 6
  codeexec/            # Phase 3
  computeruse/         # Phase 4
```
