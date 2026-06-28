# FRIDAY — Build Plan

> Living document. Supersedes the original brief where they conflict.
> Specifically: **Kokoro is the TTS provider — ElevenLabs is out**.

---

## Project status

- **Phase 0 (scaffold):** done. `python -m friday.main` boots, loads
  config (`.env` + `config.yaml` via pydantic), logs readiness, exits clean.
- **Phase 1 (MVP loop):** **DONE.** All six chunks shipped: audio I/O,
  Groq Whisper STT, skills registry + 5 starter skills, safety gate,
  Groq LLM + budget tracker + sliding window, and the agent loop
  itself. `python -m friday.agent.loop` runs the brief's acceptance
  gate end-to-end ("open Spotify and play music" → tool-calls →
  dry-run audit → verbal confirmation).

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

1. **`friday/audio/`** — **DONE.** `capture.py` (sounddevice mic +
   PTT-driven start/stop), `playback.py` (`AudioChunk` iterator →
   speakers with `stop()` → `stream.abort()` for barge-in), `hotkey.py`
   (PTT `ctrl+space` + panic `ctrl+shift+esc` via `pynput.Listener`),
   `demo.py` (`python -m friday.audio.demo` — end-to-end I/O smoke test
   that records, then plays a Kokoro-synthesised duration line back).
   Extras: `audio = ["sounddevice>=0.4", "pynput>=1.7", "numpy>=1.26"]`.
2. **`friday/stt/groq_whisper.py`** — **DONE.** `GroqWhisperSTT`
   against the existing `STTProvider` interface, driven by the
   OpenAI-SDK with `base_url = https://api.groq.com/openai/v1`. Wraps
   the caller's 16-bit PCM bytes in an in-memory WAV (stdlib `wave`,
   no disk I/O), POSTs to `audio.transcriptions`, returns a
   `Transcript` with `duration_s` and `latency_ms` populated. Bounded
   `retry-after` honoring on 429s; full RPM/TPM budgets land with
   chunk 5. New helper `friday/audio/encoding.py` converts the
   `MicRecorder` float32 buffer to PCM16 bytes for the wire. End-to-end
   demo at `python -m friday.stt.demo` — record, transcribe, echo via
   Kokoro. Extras: `stt-groq = ["openai>=1.40"]`.
3. **`friday/skills/registry.py` + `builtin.py`** — **DONE.** `@skill`
   decorator + `SkillRegistry` with auto JSON-schema generation from
   signature + docstring. Supports primitives, `Literal[...]` enums,
   `list[T]`, `T | None`, and NumPy-style docstring parameter blocks.
   Five starter skills register on package import: `open_app`,
   `web_search`, `media_control`, `system_info`, `type_text`.
   `type_text` is flagged `destructive=True` for chunk 4 to enforce.
   Schema-generation tests live in `tests/test_skill_schema.py` (the
   spec calls these out by name); registry behavior + builtin metadata
   tests in `tests/test_skill_registry.py`. All 35 tests pass.
   Extras: `skills = ["pyautogui>=0.9", "psutil>=5.9"]` — lazy-imported
   inside skill bodies so the registry stays import-clean.
4. **`friday/agent/safety.py`** — **DONE.** `SafetyGate.evaluate` and
   `evaluate_shell` return one of four decisions (`allow`, `dry_run`,
   `needs_confirmation`, `deny`) in a fixed order: unknown skill →
   deny; off-allowlist `open_app` → needs_confirmation; destructive
   outside dry-run → needs_confirmation; dry-run gate last. Allowlist
   matching is case-insensitive for apps, exact for shell. Every
   decision is appended as a JSON line to `friday_audit.log` via the
   existing `audit()` helper. Separate `record_execution(...)` event
   for the executor (chunk 6) to log realized outcomes. 18 tests in
   `tests/test_safety.py` cover all four axes plus audit-log content.
5. **`friday/llm/groq_provider.py`** — **DONE.** `GroqLLM` implements
   `LLMProvider` against Groq's OpenAI-SDK-compatible chat-completions
   endpoint. Two-way translation between `ChatMessage`/`ToolCall` and
   the OpenAI wire format (tool-call arguments auto-decoded from JSON
   so the agent loop never sees a string-shaped dict). 3-retry 429
   handling routed through the budget tracker.
   `friday/utils/tokens.py` ships the `BudgetTracker`: rolling-60s
   windows for RPM/TPM, rolling-24h window for RPD, warn-at-pct
   logging, `BudgetExceededError` on daily-cap exhaustion, and
   server-honored `retry-after` with a 30 s cap. Clock + sleeper are
   injectable for deterministic tests. `friday/llm/history.py` ships
   `SlidingWindowHistory` — sticky system message + last N user turns
   anchored on user messages so tool-call/tool-result pairs are never
   split. Extras: `llm-groq = ["openai>=1.40"]`. 39 new tests across
   `tests/test_{tokens,history,groq_provider_translation}.py`; total
   suite now 92 passing in <0.2 s. Live smoke test against the real
   Groq API confirmed plain chat + tool-calling round-trip both work
   end-to-end (the model correctly picks `open_app` from the chunk 3
   registry when asked to open Spotify).
6. **`friday/agent/loop.py` + `router.py` + `executor.py`** —
   **DONE.** `Router` is a pure function (`ChatResponse → RouteResult`)
   splitting model output into speech + tool calls; `is_terminal` is
   true exactly when there are no tool calls left, which is the
   loop's stop condition. `Executor` runs one `ToolCall` through the
   `SafetyGate`, dispatches via the registry when allowed, captures
   exceptions as `[error] …` strings, and renders non-allow decisions
   as synthetic `[dry_run]` / `[needs_confirmation]` / `[denied]`
   tool results so the LLM can react. `AgentLoop` ties everything to
   the audio I/O: PTT press → mic capture → STT → LLM ↔ executor
   round-trips (bounded to 6 hops) → TTS, with the panic key cutting
   playback hard. `python -m friday.agent.loop` is the daemon entry
   point. 16 new tests across `tests/test_router.py` (spec-required)
   and `tests/test_executor.py`. Total suite: 108 passing in <0.2s.
   The brief's acceptance gate was verified live against the real
   Groq API — "Open Spotify and play music" produced `open_app` +
   `media_control` tool calls in one hop, both gated to `dry_run`,
   followed by the verbal confirmation "I opened Spotify and started
   playing music." The audit log captured both `skill_invoke`
   `dry_run` decisions cleanly.

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
  utils/tokens.py      # DONE   — chunk 5 (RPM/TPM/RPD tracker + 429 backoff)
  stt/base.py          # DONE   — interface only
  stt/groq_whisper.py  # DONE   — chunk 2 (Groq Whisper via openai SDK)
  stt/demo.py          # DONE   — mic → Groq Whisper → Kokoro echo demo
  llm/base.py          # DONE   — interface only
  llm/groq_provider.py # DONE   — chunk 5 (Groq chat + tool calling + budget)
  llm/history.py       # DONE   — chunk 5 (sliding-window conversation history)
  tts/base.py          # DONE
  tts/kokoro.py        # DONE   — default local TTS
  audio/               # DONE   — chunk 1 (capture + playback + hotkey + demo)
  audio/encoding.py    # DONE   — float32 → PCM16 helper for STT (lands with chunk 2)
  skills/registry.py   # DONE   — chunk 3 (@skill + SkillRegistry + JSON-schema gen)
  skills/builtin.py    # DONE   — chunk 3 (5 starter skills)
  agent/safety.py      # DONE   — chunk 4 (gate + allowlists + audit)
  agent/router.py      # DONE   — chunk 6 (ChatResponse → RouteResult)
  agent/executor.py    # DONE   — chunk 6 (one ToolCall through the gate)
  agent/loop.py        # DONE   — chunk 6 (full PTT conversation daemon)
  codeexec/            # Phase 3
  computeruse/         # Phase 4
```
