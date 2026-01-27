@echo off
echo Starting ML Standalone Application...
echo.
echo Starting backend server...
start "ML Backend" cmd /k "cd /d %~dp0backend && if not exist venv (python -m venv venv) && call venv\Scripts\activate && if not exist venv\Lib\site-packages\ollama (pip install -r requirements.txt --no-build-isolation) && uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload"
timeout /t 3 /nobreak >nul
echo.
echo Starting frontend server...
start "ML Frontend" cmd /k "cd /d %~dp0frontend && if not exist node_modules (call npm install) && npm run dev"
echo.
echo Application starting...
echo Backend: http://localhost:8002
echo Frontend: http://localhost:5174
echo.
echo Press any key to exit this window (servers will continue running)...
pause >nul

