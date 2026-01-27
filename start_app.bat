@echo off
echo ========================================
echo   DrugGuard - Drug Interaction Checker
echo ========================================
echo.

REM Start backend in new window
echo Starting Backend Server...
start "DrugGuard Backend" cmd /k "cd backend && conda activate drugguard && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

REM Start frontend in new window
echo Starting Frontend Server...
start "DrugGuard Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo   Servers are starting...
echo ========================================
echo.
echo   Backend API:  http://localhost:8000
echo   API Docs:     http://localhost:8000/docs
echo   Frontend:     http://localhost:5173
echo.
echo   Press Ctrl+C in each window to stop.
echo ========================================
