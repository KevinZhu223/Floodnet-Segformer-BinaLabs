#!/usr/bin/env bash

# This script runs the FloodNet Best Model Recorder.
# It monitors evaluations and saves the best checkpoints to a separate archive.

# Path to the python binary
PYTHON_BIN="python3"

# If not found, fall back to python
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    PYTHON_BIN="python"
fi

echo "🚀 Starting FloodNet Best Model Recorder..."
$PYTHON_BIN save_best_floodnet.py
