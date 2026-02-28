#!/bin/bash
# Restart Voice Mode Server

echo "[voice-mode] Stopping any existing servers..."
pkill -f "voice_mode/server.py" 2>/dev/null
lsof -ti :7823 | xargs kill -9 2>/dev/null
sleep 2

echo "[voice-mode] Starting server..."
cd ~/voice_mode
.venv/bin/python server.py &

echo "[voice-mode] Waiting for server + Kokoro warmup..."
sleep 5

if curl -s http://localhost:7823/health >/dev/null; then
    echo "[voice-mode] Server running on http://localhost:7823"
    open http://localhost:7823
else
    echo "[voice-mode] Server failed to start"
    exit 1
fi
