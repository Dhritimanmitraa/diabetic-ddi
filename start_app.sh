#!/usr/bin/env bash
# =============================================================================
# DrugGuard - Application Startup Script (Linux/macOS)
# =============================================================================
# Starts the FastAPI backend and Vite frontend development servers.
# Usage: ./start_app.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

BACKEND_PORT=8000
FRONTEND_PORT=5173

# PIDs for cleanup
BACKEND_PID=""
FRONTEND_PID=""

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup: kill both servers on exit / Ctrl+C
# ─────────────────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down servers..."
    [ -n "$BACKEND_PID" ]  && kill "$BACKEND_PID"  2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
require_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "[ERROR] '$1' is not installed or not in PATH."
        echo "        Please install $2 and try again."
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Prerequisite checks
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  DrugGuard - Application Startup"
echo "============================================"
echo ""

require_cmd python3 "Python 3.10+"
require_cmd npm    "Node.js 18+"

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[OK] Python $PYTHON_VERSION"

NODE_VERSION=$(node --version)
echo "[OK] Node.js $NODE_VERSION"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Backend: virtual environment + dependencies
# ─────────────────────────────────────────────────────────────────────────────
echo "--------------------------------------------"
echo "  Setting up Backend"
echo "--------------------------------------------"

cd "$BACKEND_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
# shellcheck disable=SC1091
source venv/bin/activate

# Install/upgrade dependencies
echo "[INFO] Installing Python dependencies (this may take a moment)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create .env from example if it doesn't exist
if [ ! -f ".env" ]; then
    echo "[INFO] Creating backend/.env from .env.example..."
    cp .env.example .env
    echo "[NOTE] Edit backend/.env to add your GOOGLE_API_KEY for AI features."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Frontend: npm install
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--------------------------------------------"
echo "  Setting up Frontend"
echo "--------------------------------------------"

cd "$FRONTEND_DIR"

# Install npm dependencies
echo "[INFO] Installing npm dependencies..."
npm install --silent

# Create .env from example if it doesn't exist
if [ ! -f ".env" ]; then
    echo "[INFO] Creating frontend/.env from .env.example..."
    cp .env.example .env
fi

# ─────────────────────────────────────────────────────────────────────────────
# Start servers
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--------------------------------------------"
echo "  Starting Servers"
echo "--------------------------------------------"

# Start backend
cd "$BACKEND_DIR"
echo "[INFO] Starting backend on http://localhost:$BACKEND_PORT ..."
python -m uvicorn app.main:app --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Poll the health endpoint until the backend is ready (up to 30 s)
echo "[INFO] Waiting for backend to become ready..."
READY=0
for i in $(seq 1 30); do
    if curl -sf "http://localhost:$BACKEND_PORT/health/live" > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done
if [ "$READY" -eq 0 ]; then
    echo "[ERROR] Backend did not start within 30 seconds. Check /tmp/backend.log for details."
    exit 1
fi
echo "[OK] Backend is ready."

# Start frontend
cd "$FRONTEND_DIR"
echo "[INFO] Starting frontend on http://localhost:$FRONTEND_PORT ..."
npm run dev -- --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

# ─────────────────────────────────────────────────────────────────────────────
# Ready
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  DrugGuard is running!"
echo "============================================"
echo ""
echo "  Frontend:  http://localhost:$FRONTEND_PORT"
echo "  Backend:   http://localhost:$BACKEND_PORT"
echo "  API Docs:  http://localhost:$BACKEND_PORT/docs"
echo ""
echo "  Press Ctrl+C to stop all servers."
echo "============================================"
echo ""

# Wait for both background processes
wait "$BACKEND_PID" "$FRONTEND_PID"
