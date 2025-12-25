@echo off
echo ========================================
echo   DrugGuard - Drug Interaction Checker
echo ========================================
echo.
echo Starting backend and frontend servers...
echo.

REM Start backend in new window
start "DrugGuard Backend" cmd /k "run_backend.bat"

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak

REM Start frontend in new window  
start "DrugGuard Frontend" cmd /k "run_frontend.bat"

echo.
echo ========================================
echo   Servers are starting...
echo ========================================
echo.
echo   Backend API:  http://localhost:8000
echo   API Docs:     http://localhost:8000/docs
echo   Frontend:     http://localhost:3000
echo.
echo   Press Ctrl+C in each window to stop.
echo ========================================

