# FRIDAY

A voice-controlled AI assistant for your Mac. Hold a hotkey, say what you want, it does it — files, apps, search, Spotify, whatever.

```
Hold ctrl+space → talk → release → done
```

## What it can do

| Category | Skills |
|---|---|
| **Files** | Create, read, overwrite, delete files and folders |
| **Apps** | Open and close any app by name |
| **Web** | Google search (opens in browser), YouTube search |
| **Spotify** | Search, play a song, see what's playing |
| **System** | Volume control, clipboard read/write, screenshot, list running apps |
| **Keyboard** | Type any text into whatever app is focused |
| **Info** | CPU, RAM, disk, battery, IP address |
| **Media** | Play, pause, next, previous track |

## Stack

- **STT** — Groq Whisper (fast, accurate, free tier)
- **LLM** — Groq llama-3.1-8b-instant (30k TPM free tier)
- **TTS** — Kokoro (local, offline, no API key, ~1s/sentence on a laptop)

All three run on the free tier. First run downloads ~330 MB of Kokoro weights.

## Setup

**1. Clone and install**

```bash
git clone https://github.com/xOAviOx/fridayAI
cd fridayAI
uv sync --extra audio --extra tts-kokoro --extra stt-groq --extra llm-groq --extra skills
```

No `uv`? `pip install -e ".[audio,tts-kokoro,stt-groq,llm-groq,skills]"` works too.

**2. Add your API key**

```bash
cp .env.example .env
```

Open `.env` and set:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com). That's the only key you need for the default setup.

**3. macOS permissions**

FRIDAY needs mic access and accessibility permissions for global hotkeys. macOS will prompt you the first time. If the hotkey doesn't work, go to **System Settings → Privacy & Security → Accessibility** and add Terminal (or your terminal app).

## Run

```bash
uv run python -m friday.agent.loop
```

Hold `ctrl+space` to talk, release to send. Press `ctrl+shift+esc` to quit. Press `ctrl+space` during a response to interrupt (barge-in).

## Example commands

- *"Create a file on my desktop called server.js and write a basic Express app in it"*
- *"Open Spotify and play Blinding Lights"*
- *"Search YouTube for lo-fi beats"*
- *"Google the weather in Mumbai"*
- *"Take a screenshot"*
- *"What's my IP address?"*
- *"Set volume to 50"*
- *"Open Chrome"*
- *"What's currently playing on Spotify?"*
- *"Close Spotify"*
- *"Copy this to clipboard: hello world"*

## Config

Everything lives in `config.yaml`. Key things you might want to change:

```yaml
providers:
  llm: groq          # groq | openai_compat
  tts: kokoro        # kokoro (local, free) | elevenlabs (cloud)

llm:
  groq:
    model: llama-3.1-8b-instant   # fast, 30k TPM free
    # model: llama-3.3-70b-versatile  # smarter, only 6k TPM free

tts:
  kokoro:
    voice: af_heart   # af_heart (female) | am_adam (male) | bf_emma (British female)
    speed: 1.0

hotkeys:
  push_to_talk: ctrl+space
  panic: ctrl+shift+esc

safety:
  dry_run: false               # set true to log actions without executing
  skip_confirm_destructive: true
```

## Spotify (optional — play songs by name)

Basic search works with zero setup (opens the Spotify app). To enable **play by name**, set up a Spotify Developer app:

1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) → Create App
2. Set redirect URI to `http://localhost:8888/callback`
3. Add to `.env`:

```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

Then: `uv sync --extra spotify`

## Switching LLM

Point FRIDAY at any OpenAI-compatible endpoint — local models via LM Studio, Ollama, vLLM, etc.:

```yaml
# config.yaml
providers:
  llm: openai_compat

llm:
  openai_compat:
    base_url: http://localhost:1234/v1
    model: lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF
```

Set `OPENAI_COMPAT_API_KEY` in `.env`.

## Project layout

```
friday/
  agent/loop.py        # main entry point — PTT → STT → LLM → TTS loop
  agent/safety.py      # guards every tool call (dry-run, allowlists, audit log)
  agent/executor.py    # runs tool calls through the safety gate
  agent/router.py      # parses LLM response into speech + tool calls
  llm/groq_provider.py # Groq / OpenAI-compatible chat with streaming + tool calling
  llm/history.py       # sliding window conversation history
  stt/groq_whisper.py  # Groq Whisper speech-to-text
  tts/kokoro.py        # local Kokoro TTS
  audio/               # mic capture, speaker playback, PTT hotkeys
  skills/builtin.py    # all built-in skills (file, app, web, system, media)
  skills/spotify.py    # Spotify skills
  skills/registry.py   # @skill decorator — auto-generates JSON schemas
  utils/tokens.py      # budget tracker (RPM/TPM gating, 429 retry)
  config.py            # config.yaml + .env loader
```

## Notes

- Secrets only in `.env`, never in `config.yaml`. `.env` is gitignored.
- Every skill call is logged to `friday_audit.log` with a timestamp.
- `safety.dry_run: true` logs what FRIDAY would do without actually doing it — useful for testing.
