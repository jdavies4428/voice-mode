#!/usr/bin/env python3
"""Voice Mode server — watches OpenClaw session files, serves TTS."""

import asyncio
import io
import json
import os
import re
import struct
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response

PORT = 7823
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"

# Session directory
SESSIONS_DIR = Path("~/.openclaw/agents/main/sessions").expanduser()


OPENCLAW_CONFIG = Path("~/.openclaw/openclaw.json").expanduser()
_config_cache: dict = {"mtime": 0, "model": None, "provider": None}


def get_config_model() -> tuple[str | None, str | None]:
    """Read the default model from OpenClaw's config file (cached by mtime)."""
    try:
        mtime = OPENCLAW_CONFIG.stat().st_mtime
        if mtime == _config_cache["mtime"]:
            return _config_cache["model"], _config_cache["provider"]
        config = json.loads(OPENCLAW_CONFIG.read_text())
        primary = config.get("agents", {}).get("defaults", {}).get("model", {}).get("primary")
        provider = primary.split("/")[0] if primary and "/" in primary else None
        _config_cache.update(mtime=mtime, model=primary, provider=provider)
        return primary, provider
    except Exception:
        return None, None


def get_latest_session() -> Path | None:
    """Return the most recently modified .jsonl session file."""
    try:
        files = list(SESSIONS_DIR.glob("*.jsonl"))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)
    except Exception:
        return None


def scan_session_metadata(session_file: Path) -> dict:
    """Scan a session file for the current model and cumulative cost."""
    model_id = None
    provider = None
    total_cost = 0.0
    total_input = 0
    total_output = 0
    try:
        with open(session_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = entry.get("type")
                # Model change events
                if etype == "model_change":
                    model_id = entry.get("modelId", model_id)
                    provider = entry.get("provider", provider)
                # Model snapshots
                elif etype == "custom" and entry.get("customType") == "model-snapshot":
                    data = entry.get("data", {})
                    model_id = data.get("modelId", model_id)
                    provider = data.get("provider", provider)
                # Accumulate usage from assistant messages
                elif etype == "message":
                    msg = entry.get("message", {})
                    usage = msg.get("usage")
                    if usage:
                        total_cost += usage.get("cost", {}).get("total", 0)
                        total_input += usage.get("input", 0)
                        total_output += usage.get("output", 0)
    except Exception:
        pass
    # Config model is authoritative (user may have changed it via `openclaw models set`)
    config_model, config_provider = get_config_model()
    if config_model:
        model_id = config_model
        provider = config_provider

    return {
        "modelId": model_id,
        "provider": provider,
        "totalCost": round(total_cost, 6),
        "totalInput": total_input,
        "totalOutput": total_output,
    }


def _meta_event(model, provider, cost, input_tokens, output_tokens) -> str:
    return f"data: {json.dumps({'type': 'meta', 'model': format_model_name(model), 'provider': provider or '', 'cost': cost, 'inputTokens': input_tokens, 'outputTokens': output_tokens})}\n\n"


def format_model_name(model_id: str | None) -> str:
    """Format model ID for display: 'anthropic/claude-opus-4-6' -> 'claude opus 4.6'"""
    if not model_id:
        return "unknown"
    # Strip provider prefix
    name = model_id.split("/")[-1]
    # Clean up common patterns
    name = name.replace("-", " ").replace("_", " ")
    # Fix version dots: "4 6" -> "4.6"
    name = re.sub(r'(\d+)\s+(\d+)$', r'\1.\2', name)
    name = re.sub(r'(\d+)\s+(\d+)\s+', r'\1.\2 ', name)
    return name


# --- Kokoro TTS singleton ---
_kokoro = None
_kokoro_lock = asyncio.Lock()

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "am_fenrir")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.1"))


def _load_kokoro():
    from kokoro_onnx import Kokoro
    model_path = str(MODELS_DIR / "kokoro-v1.0.onnx")
    voices_path = str(MODELS_DIR / "voices-v1.0.bin")
    return Kokoro(model_path, voices_path)


async def get_kokoro():
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
    samples, sr = kokoro.create(text, voice=voice, speed=speed)
    return samples, sr


def samples_to_wav(samples, sample_rate: int) -> bytes:
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
    text = re.sub(r'\bJefe\b', 'Heffe', text)
    text = re.sub(r'\bjefe\b', 'heffe', text)
    return text


def extract_text_from_content(content):
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        texts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
                elif c.get("type") == "thinking":
                    pass
            elif isinstance(c, str):
                texts.append(c)
        text = " ".join(texts)
    else:
        text = str(content)
    text = re.sub(r'\[\[[^\]]*\]\]\s*', '', text).strip()
    return text


def is_valid_response(text: str) -> bool:
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
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return Response(status_code=400)
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


@app.post("/send")
async def send_message(request: Request):
    """Send a message to OpenClaw via CLI."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return Response(status_code=400)
    try:
        proc = await asyncio.create_subprocess_exec(
            "openclaw", "agent", "--agent", "main", "--message", text, "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Don't wait — let SSE pick up the response from the session file
        # But we do want to know if the command at least started
        # Give it 2 seconds to fail fast if there's an error
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
            if proc.returncode and proc.returncode != 0:
                err = (await proc.stderr.read()).decode()
                print(f"[voice-mode] openclaw error: {err[:200]}")
                return JSONResponse({"error": err[:200]}, status_code=502)
        except asyncio.TimeoutError:
            pass  # Still running = good, means it's processing
        return JSONResponse({"status": "sent"})
    except FileNotFoundError:
        return JSONResponse({"error": "openclaw CLI not found"}, status_code=500)
    except Exception as e:
        print(f"[voice-mode] send error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/tts/voices")
async def tts_voices():
    try:
        kokoro = await get_kokoro()
        voices = kokoro.get_voices()
        return JSONResponse({"voices": voices, "current": KOKORO_VOICE})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/events")
async def events(request: Request):
    async def generate():
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

        # Scan existing file for metadata (model, cost) and send as first event
        meta = scan_session_metadata(session_file)
        current_model = meta["modelId"]
        current_provider = meta["provider"]
        session_cost = meta["totalCost"]
        session_input = meta["totalInput"]
        session_output = meta["totalOutput"]

        yield _meta_event(current_model, current_provider, session_cost, session_input, session_output)

        # Open file and seek to end
        f = open(session_file, "r")
        f.seek(0, 2)
        last_size = f.tell()
        last_processed_id = None
        pending_working = False
        check_counter = 0

        try:
            while True:
                if await request.is_disconnected():
                    break

                check_counter += 1
                if check_counter >= 40:
                    check_counter = 0
                    latest = get_latest_session()
                    if latest and latest != session_file:
                        print(f"[voice-mode] Switching to new session: {latest.name}")
                        f.close()
                        session_file = latest
                        f = open(session_file, "r")
                        # Scan new session metadata
                        meta = scan_session_metadata(session_file)
                        current_model = meta["modelId"]
                        current_provider = meta["provider"]
                        session_cost = meta["totalCost"]
                        session_input = meta["totalInput"]
                        session_output = meta["totalOutput"]
                        yield _meta_event(current_model, current_provider, session_cost, session_input, session_output)
                        f.seek(0, 2)
                        last_size = f.tell()
                        last_processed_id = None
                        pending_working = False
                        continue

                f.seek(0, 2)
                current_size = f.tell()

                if current_size > last_size:
                    f.seek(last_size)
                    new_data = f.read()
                    last_size = current_size

                    for line in new_data.strip().split("\n"):
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        etype = entry.get("type")

                        # Track model changes in real time
                        if etype == "model_change":
                            current_model = entry.get("modelId", current_model)
                            current_provider = entry.get("provider", current_provider)
                            yield _meta_event(current_model, current_provider, session_cost, session_input, session_output)
                            continue

                        if etype == "custom" and entry.get("customType") == "model-snapshot":
                            data = entry.get("data", {})
                            current_model = data.get("modelId", current_model)
                            current_provider = data.get("provider", current_provider)
                            yield _meta_event(current_model, current_provider, session_cost, session_input, session_output)
                            continue

                        if etype != "message":
                            continue

                        msg = entry.get("message", {})
                        role = msg.get("role")

                        if role == "user":
                            if not pending_working:
                                pending_working = True
                                yield f"data: {json.dumps({'type': 'working'})}\n\n"
                            continue

                        if role != "assistant":
                            continue

                        msg_id = entry.get("id", "")
                        if msg_id == last_processed_id:
                            continue
                        last_processed_id = msg_id

                        # Extract usage data
                        usage = msg.get("usage")
                        msg_model = msg.get("model", current_model)
                        msg_cost = 0
                        msg_input = 0
                        msg_output = 0
                        if usage:
                            msg_cost = usage.get("cost", {}).get("total", 0)
                            msg_input = usage.get("input", 0)
                            msg_output = usage.get("output", 0)
                            session_cost += msg_cost
                            session_input += msg_input
                            session_output += msg_output

                        content = msg.get("content", "")
                        text = extract_text_from_content(content)

                        if is_valid_response(text):
                            yield f"data: {json.dumps({'type': 'done', 'text': text, 'model': format_model_name(msg_model), 'tokens': msg_input + msg_output, 'cost': round(msg_cost, 6), 'sessionCost': round(session_cost, 6)})}\n\n"
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
    try:
        kokoro = await get_kokoro()
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
