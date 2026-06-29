# FRIDAY

A voice-controlled AI assistant for your desktop. Hold a hotkey, say what you want, it does it — files, apps, search, Spotify, weather, reminders, code, screenshots, whatever. Runs on the free tier of Groq + a local TTS model, so the baseline cost is **$0/month**.

```
Hold ctrl+space → talk → release → done
```

There's no web UI, no Electron, no Docker. It's a Python process you run in a terminal that listens to your mic and speaks back through your speakers.

---

## What it can do

### Files & folders
- Read, create, overwrite, and delete files (32 KB cap on reads)
- Create and delete folders (including non-empty ones — destructive flag gated)

### Apps
- Open any app by name (Spotify, Chrome, Discord, etc.)
- Close a running app by name
- List currently running apps

### Web
- Open a Google search in your default browser
- Open a YouTube search in your default browser
- Run a real web search via DuckDuckGo and read the top results back
- Fetch any URL and read its first 3,000 characters back

### Spotify
- Search Spotify (zero setup — opens the app)
- Play a specific song by name (needs Spotify Developer creds in `.env`)
- Report the currently playing track

### System & hardware
- Volume control (0–100)
- Media keys: play/pause, next, previous, volume up/down
- Screenshot the display to a file
- Read and write the system clipboard
- Battery %, CPU load, RAM usage, disk usage, IP address (local + public)
- Current time

### Memory
- Tell FRIDAY to remember a fact (`remember`)
- Search what she knows (`recall`)
- List everything she remembers, forget specific things
- Memory persists across restarts in a local JSON file

### Reminders & timers
- "Remind me in 10 minutes to check the oven"
- "Set a 5 minute pomodoro timer"
- List pending reminders, cancel by id
- She speaks the reminder out loud when it's due — no app, no notification

### Weather
- "What's the weather in Mumbai" / "What's the weather" (auto-detects location)

### Briefing
- "Good morning" → spoken status briefing with greeting, weather, pending reminders, and a random memory she's stored

### Screen vision (optional)
- "What's on my screen" → screenshots and asks the Groq vision model
- "Look at /path/to/image.png and tell me what it is"

### Code execution (optional, off by default)
- Run Python or shell commands and read the output back
- Gated by `safety.enable_code_exec` — opt-in, no surprises

### Keyboard control
- Type any text into whatever app currently has focus

### Proactive monitor
- Background watcher speaks up on its own if battery drops below 15%, CPU stays above 85% for 30s, RAM hits 90%, or disk hits 95%

### Wake word (optional)
- Toggle on in config: "hey friday" wakes her up, no hotkey needed
- Uses webrtcvad + Groq Whisper for any-accent detection (not the openwakeword default which is American-English-only)

### Streaming + barge-in
- LLM tokens stream straight into a sentence splitter and TTS — she starts speaking before the model finishes generating
- Press `ctrl+space` while she's talking to interrupt and ask something new (true barge-in, not "she finishes then listens")

### Safety
- `safety.dry_run: true` logs what she *would* do without executing
- App + shell allowlists for stricter setups
- Every skill call is appended to `friday_audit.log` with a timestamp
- Destructive skills (delete, type, write) are gated by an opt-in flag

---

## Stack

| Layer | Default | Alternative |
|---|---|---|
| **STT** | Groq Whisper (`whisper-large-v3-turbo`) | — |
| **LLM** | Groq (`llama-3.3-70b-versatile`, falls back to `llama-3.1-8b-instant` on rate limit) | Any OpenAI-compatible endpoint: LM Studio, Ollama, vLLM, OpenAI itself |
| **TTS** | Kokoro 82M (local, CPU, no API key) | ElevenLabs cloud (paid for non-default voices) |
| **Audio** | sounddevice + pynput global hotkeys | — |
| **Wake word** | webrtcvad + Groq Whisper | — |
| **Vision** | Groq's vision model (`llama-3.2-90b-vision`) | — |

All defaults run on free tiers. First run pulls ~330 MB of Kokoro weights into the HuggingFace cache. No torch CUDA needed — Kokoro runs CPU-only at ~1s/sentence on a modern laptop.

---

## Setup

### 1. Install

```bash
git clone https://github.com/xOAviOx/fridayAI
cd fridayAI
pip install -e ".[audio,tts-kokoro,stt-groq,llm-groq,skills]"
```

If you want optional bits:

```bash
# Always-on "hey friday" wake word
pip install -e ".[wake-word]"        # may need: pip install webrtcvad-wheels  (Windows)

# Spotify "play this song"
pip install -e ".[spotify]"

# ElevenLabs cloud TTS
pip install -e ".[tts-elevenlabs]"
```

### 2. Get a Groq API key

Free at [console.groq.com](https://console.groq.com). The free tier gives you ~30k tokens/min on the 8b model and 6k tokens/min on the 70b — more than enough for personal use.

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_your_key_here
```

That's the only secret required for the default setup.

### 3. Platform notes

**macOS** — first run prompts for mic + accessibility permissions (the latter is for global hotkeys). If `ctrl+space` doesn't fire, go to **System Settings → Privacy & Security → Accessibility** and add Terminal (or iTerm/whatever).

**Windows** — Windows Defender may flag the global hotkey listener. Allow it. If `pip install webrtcvad` fails (needs MSVC), use `pip install webrtcvad-wheels` instead.

**Linux** — install PortAudio (`sudo apt install portaudio19-dev`) before pip-installing the `audio` extra.

---

## Run

```bash
python -m friday.agent.loop
```

- Hold `ctrl+space` → talk → release
- `ctrl+shift+esc` → panic-quit (cuts audio immediately)
- Press `ctrl+space` *during* a reply → barge-in (interrupts and listens)

If wake word is enabled, just say *"hey friday, what time is it"* in one continuous sentence — no pause after the wake phrase or the recording cuts short.

---

## Example commands

```
"Create a file on my desktop called server.js and write a basic Express app in it"
"Open Spotify and play Blinding Lights"
"What's currently playing on Spotify?"
"Search YouTube for lo-fi beats"
"Google the weather in Mumbai"
"What's the weather here"
"Take a screenshot"
"What's my IP address?"
"What's on my screen right now"   (vision)
"Set volume to 50"
"Open Chrome"
"Close Spotify"
"Copy this to clipboard: hello world"
"Remind me in 10 minutes to check the oven"
"Set a 25 minute pomodoro"
"Remember that my mom's birthday is March 14th"
"What do you know about my mom"
"Good morning"                    (briefing)
"List the files on my desktop"
"Run this python: print(2**100)"  (if code exec enabled)
```

---

## Config

Everything lives in `config.yaml`. The most-edited knobs:

```yaml
providers:
  stt: groq           # only option for now
  llm: groq           # groq | openai_compat
  tts: kokoro         # kokoro (local, free) | elevenlabs (cloud)

llm:
  groq:
    model: llama-3.3-70b-versatile      # smarter, 6k TPM free
    # model: llama-3.1-8b-instant       # faster, 30k TPM free

tts:
  kokoro:
    voice: af_heart                     # af_* female, am_* male, b* British
    speed: 1.0
  elevenlabs:
    voice_id: "21m00Tcm4TlvDq8ikWAM"    # Rachel — must be in your "My Voices"
    model: eleven_turbo_v2_5

audio:
  wake_word:
    enabled: false                       # flip to true for hands-free
    silence_ms: 900
    max_record_s: 15.0

hotkeys:
  push_to_talk: ctrl+space
  panic: ctrl+shift+esc

safety:
  dry_run: false                         # true = log only, don't execute
  enable_code_exec: false                # gate for python/shell skills
  enable_computer_use: false             # gate for vision skills
  skip_confirm_destructive: true         # destructive skills run without confirm

monitor:
  enabled: true                          # proactive battery/CPU/RAM/disk alerts
```

---

## Optional integrations

### Spotify (play by name)

Search and "open Spotify" work with zero setup. To enable *"play Blinding Lights"*, register a dev app:

1. [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) → Create App
2. Set redirect URI: `http://localhost:8888/callback`
3. Add to `.env`:
   ```
   SPOTIFY_CLIENT_ID=...
   SPOTIFY_CLIENT_SECRET=...
   ```
4. `pip install -e ".[spotify]"`

### ElevenLabs cloud TTS

1. Get API key at [elevenlabs.io/app/settings/api-keys](https://elevenlabs.io/app/settings/api-keys) — must check the `text_to_speech` permission box
2. Add a voice to **My Voices** in the dashboard (free tier won't let the API hit library voices that aren't in your personal collection)
3. Copy that voice's id, paste into `config.yaml`:
   ```yaml
   providers:
     tts: elevenlabs
   tts:
     elevenlabs:
       voice_id: "21m00Tcm4TlvDq8ikWAM"
   ```
4. Add `ELEVENLABS_API_KEY` to `.env`

### Local LLMs via LM Studio / Ollama / vLLM

```yaml
providers:
  llm: openai_compat

llm:
  openai_compat:
    base_url: http://localhost:1234/v1
    model: lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF
```

Set `OPENAI_COMPAT_API_KEY` in `.env` (use a placeholder if your local server doesn't check).

---

## Project layout

```
friday/
  agent/
    loop.py            # entry point — PTT → STT → LLM → TTS, streaming, barge-in
    router.py          # parses LLM response into speech + tool calls
    executor.py        # runs tool calls through the safety gate
    safety.py          # dry-run, allowlists, destructive-skill gating, audit log
    monitor.py         # proactive system-vital watcher
  llm/
    groq_provider.py   # Groq + OpenAI-compatible streaming + tool calling
    history.py         # sliding-window conversation history
    base.py            # provider interface
  stt/
    groq_whisper.py    # Groq Whisper speech-to-text
    base.py
  tts/
    kokoro.py          # local Kokoro 82M (default)
    elevenlabs.py      # cloud streaming PCM
    base.py            # iterator-shaped streaming interface
  audio/
    capture.py         # mic stream
    playback.py        # speaker stream + stop-mid-utterance for barge-in
    hotkeys.py         # global pynput hotkey listener
    wakeword.py        # VAD + Whisper wake-phrase detection
    encoding.py        # float32 ↔ pcm16, WAV wrapping
  skills/
    registry.py        # @skill decorator → auto JSON schema
    builtin.py         # file/app/system/clipboard/screenshot/keyboard
    spotify.py         # search, play-by-name, now playing
    web.py             # DuckDuckGo search, fetch URL
    weather.py         # open-meteo current weather
    timers.py          # reminders + countdown timers (spoken)
    memory.py          # persistent fact storage
    briefing.py        # "good morning" status digest
  codeexec/
    sandbox.py         # opt-in python/shell execution
  computeruse/
    vision.py          # opt-in screen vision via Groq
  utils/
    tokens.py          # BudgetTracker — RPM/TPM gating + 429 retry
    sentence.py        # split LLM tokens into TTS-ready sentences
    logging.py         # structured log setup
  config.py            # config.yaml + .env loader (pydantic-validated)
  main.py              # legacy phase 0 entry — use agent.loop instead
```

---

## Notes

- Secrets only in `.env`; never in `config.yaml`. `.env` is gitignored.
- Every skill call appends to `friday_audit.log` (timestamp + skill + arguments + result).
- `safety.dry_run: true` is the safest way to try a new prompt — she'll log what she would have done without touching anything.
- The free Groq tier resets daily. If you blow through the 70b tokens, FRIDAY falls back to the 8b model automatically for the rest of the day.
- Kokoro weights live in `~/.cache/huggingface/`. Delete that folder to reclaim the ~330 MB if you switch to ElevenLabs.
- "Phase 7" in code comments refers to the build phase that landed wake-word + proactive monitor; it's not a runtime mode.
