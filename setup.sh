#!/bin/bash
# Voice Mode — Setup Script
# Requires: Python 3.11+, macOS or Linux

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Voice Mode Setup ==="
echo ""

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[1/3] Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "[1/3] Virtual environment already exists"
fi

# 2. Install dependencies
echo "[2/3] Installing dependencies..."
.venv/bin/pip install -q -r requirements.txt

# 3. Download Kokoro TTS models
mkdir -p models
ONNX_FILE="models/kokoro-v1.0.onnx"
VOICES_FILE="models/voices-v1.0.bin"

if [ ! -f "$ONNX_FILE" ]; then
    echo "[3/3] Downloading Kokoro TTS model (325MB)..."
    curl -L -o "$ONNX_FILE" \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
else
    echo "[3/3] Kokoro model already downloaded"
fi

if [ ! -f "$VOICES_FILE" ]; then
    echo "      Downloading voice embeddings (28MB)..."
    curl -L -o "$VOICES_FILE" \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
else
    echo "      Voice embeddings already downloaded"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start the server:"
echo "  ./restart.sh"
echo ""
echo "Or run directly:"
echo "  .venv/bin/python server.py"
echo ""
echo "Then open http://localhost:7823"
