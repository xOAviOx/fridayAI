# FRIDAY — Build Plan

> Living document. Supersedes the original brief where they conflict.
> Specifically: **Kokoro is the TTS provider — ElevenLabs is out**.

---

## Project status

- **Phase 0 (scaffold):** DONE.
- **Phase 1 (MVP loop):** DONE. Full PTT → STT → LLM → TTS → tool-call pipeline.
- **Phase 2 (streaming + barge-in):** DONE. Streaming tokens, sentence-level TTS, barge-in.
- **Phase 3 (skills power-up):** 🔨 IN PROGRESS — see below.

---

## Approved deviation from the original brief: Kokoro replaces ElevenLabs

The original spec had ElevenLabs as the Phase 1 TTS and Kokoro as a
"Phase 2+" alternative. Kokoro is the TTS provider — both the default
and the only one we're actively building toward.

---

## Tech stack (fixed)

- Python 3.11+, managed with `uv` (fallback: venv + pip)
- **Audio:** `sounddevice` (mic + speaker), `pynput` (global push-to-talk + panic)
- **STT:** Groq Whisper `whisper-large-v3-turbo`
- **LLM (brain):** Groq `llama-3.1-8b-instant` default (or `llama-3.3-70b-versatile`)
- **TTS:** Kokoro (local, free, default).
- **PC control:** `pyautogui` / `subprocess` / OS APIs inside skill functions.
- **Config:** `.env` for secrets only, `config.yaml` for everything else.

---

## Phase 3 — Skills Power-Up  🔨 BUILDING NOW

Six chunks shipped in order. Each is independently testable.

### 3.1 — Startup voice + expanded system metrics  ✅
- FRIDAY speaks *"Systems online. Ready when you are, boss."* on launch.
- New skills: `get_ram_usage()`, `get_disk_usage()`, `get_ip_address()`.

### 3.2 — Weather skill  ✅
- `get_weather(location)` → hits wttr.in (no API key, always free).
- Returns current temp, conditions, humidity, wind in a voice-friendly sentence.

### 3.3 — Real web fetch + search results  ✅
- `web_search_results(query)` → DuckDuckGo Instant Answer API → returns
  actual text answers (not just opens browser).
- `fetch_webpage(url)` → fetches any URL and returns cleaned text (strips HTML).
  FRIDAY can read articles, docs, paste-bins — anything with a URL.

### 3.4 — Persistent memory  ✅
- Stores facts to `~/.friday/memories.json`.
- `remember(fact)` — FRIDAY memorizes anything you tell her.
- `recall(topic)` — fuzzy search through memories.
- `list_memories()` — dump all stored facts.
- `forget(fact)` — remove a matching memory.

### 3.5 — Reminders & timers  ✅
- Background monitor thread in `friday/skills/timers.py`.
- `set_reminder(message, in_minutes)` — fires TTS after N minutes.
- `set_timer(minutes, label)` — simple countdown, speaks when done.
- `list_reminders()` — what's pending.
- `cancel_reminder(reminder_id)` — cancel by ID.
- Loop wires a speak-callback into the timer system so timers can fire TTS.

---

## Phase 4 — Proactive Intelligence

### 4.1 — Morning briefing
- `morning_briefing()` skill: greeting + weather + pending reminders + a fact.
- Time-aware: "Good morning / afternoon / evening, boss."

### 4.2 — Proactive system monitor
- `friday/agent/monitor.py` — background daemon, runs alongside the agent loop.
- Alerts (spoken, unprompted):
  - Battery < 15% → "Heads up, battery's at 12%."
  - CPU > 90% for 60s → "Something's hammering the CPU — Chrome's the culprit."
  - A fired reminder from the timer system.
- Configurable thresholds in `config.yaml`.

---

## Phase 5 — Code Execution

### 5.1 — Python sandbox
- Wire up `safety.enable_code_exec: true` flag (already in config, not yet wired).
- `run_python(code)` — executes in a subprocess with a 10s timeout.
  Captures stdout + stderr, returns to LLM. No file system access from sandbox.
- Safety gate: only runs when `enable_code_exec = true`.

### 5.2 — Shell commands
- `run_shell(command)` — gated by `shell_allowlist` in config (already exists).
- Powers: directory listing, git status, running scripts, process info.

---

## Phase 6 — Vision / Computer Use

### 6.1 — Screen vision
- `read_screen()` — takes a screenshot → sends to Groq vision model
  (`llama-3.2-11b-vision-preview`) → returns description.
- Enables: "What's on my screen?", "Summarize this article",
  "What does this error say?"
- `friday/computeruse/vision.py` fills the empty `computeruse/` module.

### 6.2 — Analyze image file
- `analyze_image(path)` — same vision pipeline but for a file on disk.

---

## Phase 7 — Wake Word + Always-On

### 7.1 — VAD (Voice Activity Detection)
- Replace PTT-only input with `webrtcvad` / `silero-vad`.
- Auto-detects when user starts and stops speaking.
- PTT still works as fallback / override.
- `audio.vad_enabled: true` in config (flag already exists).

### 7.2 — Wake word "Hey Friday"
- `openwakeword` library — runs locally, no cloud.
- Always-on mic → wake word triggers recording → rest of pipeline unchanged.
- `hotkeys.wake_word: "hey friday"` in config.
- Replaces push-to-talk as primary interaction mode.

---

## Phase 8 — External Integrations

### 8.1 — Google Calendar
- OAuth2 via `google-auth` + `google-api-python-client`.
- `get_calendar_events(days)` — "What's on my calendar today?"
- `create_calendar_event(title, date, time, duration)` — schedule something.
- Credentials cached in `~/.friday/google_token.json`.

### 8.2 — Gmail
- Same OAuth2 credentials as Calendar.
- `get_unread_emails(limit)` — "Any important emails?"
- `send_email(to, subject, body)` — voice-dictated emails.
- `search_emails(query)` — "Find emails from Avi".

### 8.3 — Home Assistant (smart home)
- `friday/skills/smarthome.py`.
- REST API calls to local HA instance (no cloud needed).
- `control_light(name, action, brightness)` — "Dim the lights to 30%."
- `get_home_state()` — "What's on in the house?"
- `HA_URL` + `HA_TOKEN` in `.env`.

---

## Phase 9 — Dashboard / HUD

### 9.1 — Web dashboard
- `friday/dashboard/` — FastAPI app at `http://localhost:7474`.
- Shows: live conversation transcript, active skills, token budget,
  system vitals, pending reminders.
- FRIDAY speaks "Dashboard is up at localhost 7474" on launch when enabled.

### 9.2 — Menu bar icon (macOS)
- `rumps` library — tiny menu bar app.
- Shows FRIDAY status (idle / listening / thinking / speaking).
- Click to mute / unmute, open dashboard, quit.

---

## Personality roadmap (woven through all phases)

The system prompt evolves as capabilities grow:

- **Phase 3:** Time-aware greetings. Calls user "boss" occasionally.
- **Phase 4:** References past memories in responses naturally.
- **Phase 5+:** Comments on what she sees on screen. Proactive observations.
- **Phase 7+:** Wake-word personality ("Yeah?", "On it.", "Got it, boss.").

---

## Working agreements

- One chunk at a time. Stop after each, tell the user how to test, wait for go-ahead.
- No co-author trailers on commits.
- Commit author: `xOAviOx <avishuklacode@gmail.com>`.
- Remote: `git@github.com:xOAviOx/fridayAI.git`, `main` branch.
- Secrets only in `.env`; never commit keys.
- Default to the cheapest / fastest model that works for each turn.

---

## File-by-file status

```
friday/
  main.py              # DONE — Phase 0 boot + log + exit
  config.py            # DONE — pydantic-validated .env + config.yaml loader
  utils/logging.py     # DONE — structured logger + audit helper
  utils/tokens.py      # DONE — RPM/TPM/RPD tracker + 429 backoff
  stt/groq_whisper.py  # DONE — Groq Whisper via openai SDK
  llm/groq_provider.py # DONE — Groq chat + tool calling + budget + streaming
  llm/history.py       # DONE — sliding-window conversation history
  tts/kokoro.py        # DONE — local Kokoro TTS
  audio/               # DONE — capture + playback + hotkey + demo
  skills/registry.py   # DONE — @skill decorator + JSON-schema gen
  skills/builtin.py    # DONE — 20+ built-in skills
  skills/spotify.py    # DONE — Spotify search + play + now-playing
  skills/weather.py    # Phase 3.2 ✅
  skills/web.py        # Phase 3.3 ✅
  skills/memory.py     # Phase 3.4 ✅
  skills/timers.py     # Phase 3.5 ✅
  agent/safety.py      # DONE — gate + allowlists + audit
  agent/router.py      # DONE — ChatResponse → RouteResult
  agent/executor.py    # DONE — one ToolCall through the gate
  agent/loop.py        # DONE (Phase 2) — streaming + barge-in
  agent/monitor.py     # Phase 4.2
  codeexec/sandbox.py  # Phase 5
  computeruse/vision.py# Phase 6
  dashboard/           # Phase 9
```
