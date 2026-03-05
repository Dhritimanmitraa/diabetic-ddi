@echo off
REM =============================================================================
REM Ralph Loop — Iterative Task Runner
REM =============================================================================
REM Runs a command repeatedly until it succeeds or hits the max iteration limit.
REM Usage:  ralph_loop.bat "command to run" [max_iterations]
REM Example: ralph_loop.bat "cd frontend && npm run test:run" 5
REM =============================================================================

setlocal enabledelayedexpansion

set "COMMAND=%~1"
set "MAX_ITER=%~2"

if "%COMMAND%"=="" (
    echo.
    echo  Ralph Loop - Iterative Task Runner
    echo  ====================================
    echo.
    echo  Usage:  ralph_loop.bat "command" [max_iterations]
    echo.
    echo  Examples:
    echo    ralph_loop.bat "cd backend && python -m pytest tests/ -x" 10
    echo    ralph_loop.bat "cd frontend && npm run test:run" 5
    echo    ralph_loop.bat "cd frontend && npm run build" 3
    echo.
    exit /b 1
)

if "%MAX_ITER%"=="" set MAX_ITER=10

echo.
echo  =============================================
echo   Ralph Loop - Starting
echo  =============================================
echo   Command:    %COMMAND%
echo   Max Iters:  %MAX_ITER%
echo   Started:    %date% %time%
echo  =============================================
echo.

set ITERATION=0

:loop
set /a ITERATION+=1

echo.
echo  [Iteration %ITERATION%/%MAX_ITER%] Running at %time%...
echo  -----------------------------------------

cmd /c %COMMAND%

if %ERRORLEVEL%==0 (
    echo.
    echo  =============================================
    echo   SUCCESS on iteration %ITERATION%!
    echo   Finished: %date% %time%
    echo  =============================================
    exit /b 0
)

echo.
echo  [Iteration %ITERATION%] FAILED (exit code: %ERRORLEVEL%)

if %ITERATION% GEQ %MAX_ITER% (
    echo.
    echo  =============================================
    echo   STOPPED - Hit max iterations (%MAX_ITER%)
    echo   The task did not succeed after %MAX_ITER% attempts.
    echo   Finished: %date% %time%
    echo  =============================================
    exit /b 1
)

echo  Retrying in 2 seconds...
timeout /t 2 /nobreak >nul

goto loop
