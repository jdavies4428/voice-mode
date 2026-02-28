# Voice Mode

A local voice interface for [OpenClaw](https://openclaw.ai) that speaks AI responses aloud using Kokoro TTS, with a pulsing lobster that shows when the AI is thinking.

![Voice Mode](https://img.shields.io/badge/TTS-Kokoro_82M-red) ![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey)

## What it does

- Watches OpenClaw session files for new messages in real time
- Speaks assistant responses using **Kokoro TTS** (82M parameter ONNX model, runs entirely on CPU)
- Shows a pulsing lobster that glows red when thinking and amber when speaking
- Sentence-level streaming with prefetch pipeline for minimal latency
- Auto-detects new OpenClaw sessions — no manual config needed

## Requirements

- **Python 3.11+** (3.12 recommended)
- **macOS** (Apple Silicon or Intel) or Linux
- ~400MB disk space for TTS models
- An [OpenClaw](https://openclaw.ai) installation with sessions at `~/.openclaw/agents/main/sessions/`

## Quick Start

```bash
git clone https://github.com/yourusername/voice-mode.git
cd voice-mode
chmod +x setup.sh restart.sh
./setup.sh    # creates venv, installs deps, downloads models
./restart.sh  # starts the server and opens the browser
```

The server runs at **http://localhost:7823**. Click the lobster to enable audio (browser requirement), then talk to OpenClaw normally.

## How it works

```
OpenClaw session file (.jsonl)
        │
        ▼
   server.py (FastAPI)
   ├── watches session file for new messages (50ms polling)
   ├── detects user messages → sends "working" SSE event
   ├── detects assistant responses → sends "done" SSE event
   └── /tts endpoint generates WAV audio via Kokoro
        │
        ▼
   index.html (browser)
   ├── SSE client receives events
   ├── lobster pulses red (thinking) / amber (speaking)
   ├── fetches TTS audio sentence by sentence
   ├── prefetches next sentence while playing current
   └── Web Audio API playback
```

## Configuration

### Voice

Set the voice via environment variable before starting:

```bash
export KOKORO_VOICE=am_fenrir   # default
export KOKORO_SPEED=1.1         # default
```

Available voices include: `af_jessica`, `am_fenrir`, `af_heart`, `am_adam`, `bf_emma`, `bm_george`, and [50+ more](http://localhost:7823/tts/voices).

### Run as macOS Launch Agent (auto-start on boot)

Create `~/Library/LaunchAgents/com.voice-mode.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.voice-mode</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/voice-mode/.venv/bin/python</string>
        <string>/path/to/voice-mode/server.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/voice-mode</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>5</integer>
    <key>StandardOutPath</key>
    <string>/tmp/voice-mode.out</string>
    <key>StandardErrorPath</key>
    <string>/tmp/voice-mode.err</string>
</dict>
</plist>
```

Then load it:

```bash
launchctl load ~/Library/LaunchAgents/com.voice-mode.plist
```

### Custom session directory

If your OpenClaw sessions are in a different location, edit `SESSIONS_DIR` in `server.py`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Voice Mode UI |
| `/health` | GET | Server status + current session |
| `/events` | GET | SSE stream of OpenClaw events |
| `/tts` | POST | Generate WAV audio from text |
| `/tts/voices` | GET | List available Kokoro voices |

## Troubleshooting

**No audio plays:** Click anywhere on the page first. Browsers require a user gesture before playing audio.

**Lobster stuck on "Thinking":** Click the lobster to force-reset. This can happen if the SSE connection drops.

**Wrong session:** The server auto-detects the most recently modified `.jsonl` file in the sessions directory. If you switch OpenClaw sessions, it follows within 2 seconds.

**Port in use:** `lsof -ti :7823 | xargs kill -9` then restart.

## Credits

- **[Kokoro TTS](https://github.com/hexgrad/kokoro)** — 82M parameter text-to-speech model
- **[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)** — ONNX runtime wrapper
- **[OpenClaw](https://openclaw.ai)** — AI agent framework
