@echo off
echo Starting Drug Interaction Frontend...
cd frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
)

REM Start development server
echo Starting Vite development server...
npm run dev

