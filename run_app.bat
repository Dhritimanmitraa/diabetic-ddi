@echo off
title Drug Interaction Checker - Application Launcher
color 0A

echo ============================================
echo   Drug Interaction Checker - Startup Script
echo ============================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

REM Check if Node.js is available
where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js/npm is not installed or not in PATH!
    pause
    exit /b 1
)

echo [INFO] Starting Backend Server...
echo.

REM Start Backend in a new terminal window (using venv)
start "Backend - FastAPI" cmd /k "cd /d %~dp0backend && call venv\Scripts\activate && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a few seconds for backend to initialize
echo [INFO] Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo [INFO] Starting Frontend Development Server...
echo.

REM Start Frontend in a new terminal window
start "Frontend - Vite" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ============================================
echo   Application Started Successfully!
echo ============================================
echo.
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo   Frontend: http://localhost:5173
echo.
echo   Close the terminal windows to stop the servers.
echo ============================================
echo.

pause
