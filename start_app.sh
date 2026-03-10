#!/usr/bin/env bash
# =============================================================================
# DrugGuard - Application Launcher (Linux/macOS)
# =============================================================================
# Starts the FastAPI backend and Vite frontend development server.
# Usage: ./start_app.sh
# =============================================================================

set -euo pipefail

# Resolve the directory containing this script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  DrugGuard - Application Launcher"
echo "============================================"
echo ""

# -----------------------------------------------
# Prerequisite checks
# -----------------------------------------------

if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "[ERROR] Python is not installed or not in PATH!"
    exit 1
fi

PYTHON=$(command -v python3 2>/dev/null || command -v python)

if ! command -v npm &>/dev/null; then
    echo "[ERROR] Node.js/npm is not installed or not in PATH!"
    exit 1
fi

echo "[INFO] Python: $($PYTHON --version)"
echo "[INFO] Node:   $(node --version)"
echo ""

# -----------------------------------------------
# Backend setup
# -----------------------------------------------

BACKEND_DIR="$ROOT_DIR/backend"

# Create virtual environment if it doesn't exist
if [ ! -d "$BACKEND_DIR/venv" ]; then
    echo "[INFO] Creating Python virtual environment..."
    $PYTHON -m venv "$BACKEND_DIR/venv"
fi

# Activate and install dependencies
echo "[INFO] Installing/verifying backend dependencies..."
source "$BACKEND_DIR/venv/bin/activate"
pip install -r "$BACKEND_DIR/requirements.txt"

# Create .env from example if not present
if [ ! -f "$BACKEND_DIR/.env" ] && [ -f "$BACKEND_DIR/.env.example" ]; then
    cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
    echo "[INFO] Created backend/.env from .env.example — add your GOOGLE_API_KEY to enable AI features."
fi

# -----------------------------------------------
# Frontend setup
# -----------------------------------------------

FRONTEND_DIR="$ROOT_DIR/frontend"

echo "[INFO] Installing/verifying frontend dependencies..."
cd "$FRONTEND_DIR"
npm install

# Create .env from example if not present
if [ ! -f "$FRONTEND_DIR/.env" ] && [ -f "$FRONTEND_DIR/.env.example" ]; then
    cp "$FRONTEND_DIR/.env.example" "$FRONTEND_DIR/.env"
    echo "[INFO] Created frontend/.env from .env.example."
fi

cd "$ROOT_DIR"

# -----------------------------------------------
# Start servers
# -----------------------------------------------

echo ""
echo "[INFO] Starting backend server..."
(
    source "$BACKEND_DIR/venv/bin/activate"
    cd "$BACKEND_DIR"
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
) &
BACKEND_PID=$!

echo "[INFO] Waiting for backend to initialize..."
sleep 5

echo "[INFO] Starting frontend development server..."
(
    cd "$FRONTEND_DIR"
    npm run dev
) &
FRONTEND_PID=$!

echo ""
echo "============================================"
echo "  Application Started!"
echo "============================================"
echo ""
echo "  Backend API:  http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo "  Frontend:     http://localhost:3000"
echo ""
echo "  Press Ctrl+C to stop both servers."
echo "============================================"
echo ""

# Wait for either process to exit; clean up both on exit
trap "echo ''; echo '[INFO] Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

wait $BACKEND_PID $FRONTEND_PID
