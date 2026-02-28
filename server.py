#!/usr/bin/env python3
"""Voice Mode server — watches session file directly."""

import asyncio
import io
import json
import os
import re
import struct
import sys
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response

PORT = 7823
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"

# Session directory
SESSIONS_DIR = Path("~/.openclaw/agents/main/sessions").expanduser()


def get_latest_session() -> Path | None:
    """Return the most recently modified .jsonl session file."""
    try:
        files = list(SESSIONS_DIR.glob("*.jsonl"))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)
    except Exception:
        return None

# --- Kokoro TTS singleton ---
_kokoro = None
_kokoro_lock = asyncio.Lock()

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "am_fenrir")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.1"))


def _load_kokoro():
    """Load Kokoro model synchronously (called in thread)."""
    from kokoro_onnx import Kokoro
    model_path = str(MODELS_DIR / "kokoro-v1.0.onnx")
    voices_path = str(MODELS_DIR / "voices-v1.0.bin")
    return Kokoro(model_path, voices_path)


async def get_kokoro():
    """Lazy-load Kokoro TTS model (thread-safe)."""
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    async with _kokoro_lock:
        if _kokoro is not None:
            return _kokoro
        _kokoro = await asyncio.to_thread(_load_kokoro)
        print(f"[voice-mode] Kokoro TTS loaded (voice={KOKORO_VOICE}, speed={KOKORO_SPEED})")
        return _kokoro


def _generate_audio(kokoro, text, voice, speed):
    """Generate audio synchronously (called in thread)."""
    samples, sr = kokoro.create(text, voice=voice, speed=speed)
    return samples, sr


def samples_to_wav(samples, sample_rate: int) -> bytes:
    """Convert float32 numpy samples to WAV bytes."""
    import numpy as np
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    data_size = len(pcm) * 2
    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


def fix_pronunciation(text: str) -> str:
    """Fix common pronunciation issues."""
    # Fix Jefe -> Heffe (Spanish pronunciation)
    text = re.sub(r'\bJefe\b', 'Heffe', text)
    text = re.sub(r'\bjefe\b', 'heffe', text)
    return text


def split_sentences(text: str) -> list[str]:
    """Split text into speakable sentences."""
    # Strip markdown formatting
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = []
    for p in parts:
        p = p.strip()
        if p:
            sentences.append(p)
    return sentences


def extract_text_from_content(content):
    """Extract plain text from various content formats."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        texts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
                elif c.get("type") == "thinking":
                    pass  # Skip thinking blocks
            elif isinstance(c, str):
                texts.append(c)
        text = " ".join(texts)
    else:
        text = str(content)
    # Strip OpenClaw directives like [[reply_to_current]]
    text = re.sub(r'\[\[[^\]]*\]\]\s*', '', text).strip()
    return text


def is_valid_response(text: str) -> bool:
    """Check if text is a valid response (not NO_REPLY, HEARTBEAT_OK, empty, etc.)"""
    if not text or not text.strip():
        return False
    text = text.strip()
    if text in ("NO_REPLY", "HEARTBEAT_OK"):
        return False
    if text.startswith("HEARTBEAT_OK"):
        return False
    return True


# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app):
    """Pre-load Kokoro model on server start."""
    asyncio.create_task(_warmup_kokoro())
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def index():
    html_path = SCRIPT_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/health")
async def health():
    latest = get_latest_session()
    return JSONResponse({
        "status": "ok",
        "session": latest.name if latest else None,
        "tts": "kokoro",
        "voice": KOKORO_VOICE,
    })


@app.post("/tts")
async def tts(request: Request):
    """Generate speech from text using Kokoro TTS. Returns WAV audio."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return Response(status_code=400)

    # Fix pronunciation
    text = fix_pronunciation(text)

    voice = body.get("voice", KOKORO_VOICE)
    speed = body.get("speed", KOKORO_SPEED)

    try:
        kokoro = await get_kokoro()
        samples, sr = await asyncio.to_thread(_generate_audio, kokoro, text, voice, speed)
        wav_bytes = samples_to_wav(samples, sr)
        return Response(content=wav_bytes, media_type="audio/wav",
                        headers={"Cache-Control": "no-cache"})
    except Exception as e:
        print(f"[voice-mode] TTS error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/tts/voices")
async def tts_voices():
    """List available Kokoro voices."""
    try:
        kokoro = await get_kokoro()
        voices = kokoro.get_voices()
        return JSONResponse({"voices": voices, "current": KOKORO_VOICE})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/events")
async def events(request: Request):
    """Server-sent events endpoint that watches session file for assistant responses."""
    
    async def generate():
        # Wait for any session file to exist
        session_file = None
        for _ in range(60):
            session_file = get_latest_session()
            if session_file:
                break
            await asyncio.sleep(0.5)

        if not session_file:
            yield f"data: {json.dumps({'type': 'error', 'text': 'No session files found'})}\n\n"
            return

        print(f"[voice-mode] SSE watching: {session_file.name}")

        # Open file and seek to end
        f = open(session_file, "r")
        f.seek(0, 2)  # Seek to end
        last_size = f.tell()
        last_processed_id = None
        pending_working = False
        check_counter = 0

        try:
            while True:
                if await request.is_disconnected():
                    break

                # Every 2 seconds, check if a newer session file appeared
                check_counter += 1
                if check_counter >= 40:  # 40 * 0.05s = 2s
                    check_counter = 0
                    latest = get_latest_session()
                    if latest and latest != session_file:
                        print(f"[voice-mode] Switching to new session: {latest.name}")
                        f.close()
                        session_file = latest
                        f = open(session_file, "r")
                        f.seek(0, 2)
                        last_size = f.tell()
                        last_processed_id = None
                        pending_working = False
                        continue

                # Check if file grew
                f.seek(0, 2)
                current_size = f.tell()

                if current_size > last_size:
                    # New data available, read it
                    f.seek(last_size)
                    new_data = f.read()
                    last_size = current_size

                    # Process each line
                    for line in new_data.strip().split("\n"):
                        if not line.strip():
                            continue

                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if entry.get("type") != "message":
                            continue

                        msg = entry.get("message", {})
                        role = msg.get("role")

                        # User message = OpenClaw is about to think
                        if role == "user":
                            if not pending_working:
                                pending_working = True
                                yield f"data: {json.dumps({'type': 'working'})}\n\n"
                            continue

                        if role != "assistant":
                            continue

                        msg_id = entry.get("id", "")

                        # Skip if already processed
                        if msg_id == last_processed_id:
                            continue
                        last_processed_id = msg_id

                        # Extract text content
                        content = msg.get("content", "")
                        text = extract_text_from_content(content)

                        if is_valid_response(text):
                            yield f"data: {json.dumps({'type': 'done', 'text': text})}\n\n"
                            pending_working = False
                        else:
                            yield f"data: {json.dumps({'type': 'idle'})}\n\n"
                            pending_working = False

                await asyncio.sleep(0.05)

        finally:
            f.close()

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _warmup_kokoro():
    """Load model in background so first TTS request is fast."""
    try:
        kokoro = await get_kokoro()
        # Generate a tiny warmup to JIT-compile ONNX session
        await asyncio.to_thread(_generate_audio, kokoro, "Ready.", KOKORO_VOICE, KOKORO_SPEED)
        print("[voice-mode] Kokoro TTS warmed up")
    except Exception as e:
        print(f"[voice-mode] Kokoro warmup failed: {e}")


if __name__ == "__main__":
    latest = get_latest_session()
    print(f"[voice-mode] Starting on http://localhost:{PORT}")
    print(f"[voice-mode] Sessions dir: {SESSIONS_DIR}")
    print(f"[voice-mode] Latest session: {latest.name if latest else 'none'}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
