@echo off
title DrugGuard - Full Build (Android + Windows)
color 0B

echo ============================================================
echo   DrugGuard - Full Platform Build Script
echo   Builds and runs Android APK + Windows Development Server
echo ============================================================
echo.

REM Store the root directory
set ROOT_DIR=%~dp0

REM ============================================================
REM   PREREQUISITE CHECKS
REM ============================================================

echo [CHECKING] Prerequisites...
echo.

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)
echo   [OK] Python found

REM Check Node.js
where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js/npm is not installed or not in PATH!
    pause
    exit /b 1
)
echo   [OK] Node.js/npm found

REM Check Java (required for Android)
where java >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Java not found - Android build may fail!
) else (
    echo   [OK] Java found
)

echo.
echo ============================================================
echo   STEP 1: Building Frontend Web Assets
echo ============================================================
echo.

cd /d "%ROOT_DIR%frontend"
echo [INFO] Installing npm dependencies...
call npm install

echo [INFO] Building Vite production bundle...
call npm run build

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Frontend build failed!
    pause
    exit /b 1
)
echo   [OK] Frontend build complete - dist/ created

echo.
echo ============================================================
echo   STEP 2: Syncing to Android (Capacitor)
echo ============================================================
echo.

echo [INFO] Syncing web assets to Android project...
call npx cap sync android

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Capacitor sync failed!
    pause
    exit /b 1
)
echo   [OK] Android project synced

echo.
echo ============================================================
echo   STEP 3: Building Android APK
echo ============================================================
echo.

cd /d "%ROOT_DIR%frontend\android"

echo [INFO] Building debug APK with Gradle...
call gradlew.bat assembleDebug

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Android APK build failed!
    echo [TIP] Make sure Android SDK and Java JDK are properly configured.
    pause
    exit /b 1
)

REM Copy APK to root directory for easy access
set APK_SOURCE=app\build\outputs\apk\debug\app-debug.apk
set APK_DEST=%ROOT_DIR%DrugGuard.apk

if exist "%APK_SOURCE%" (
    copy /Y "%APK_SOURCE%" "%APK_DEST%" >nul
    echo   [OK] APK built and copied to: DrugGuard.apk
) else (
    echo   [WARNING] APK file not found at expected location
)

echo.
echo ============================================================
echo   STEP 4: Starting Windows Development Servers
echo ============================================================
echo.

cd /d "%ROOT_DIR%"

echo [INFO] Starting Backend Server...
start "DrugGuard Backend" cmd /k "cd /d %ROOT_DIR%backend && call venv\Scripts\activate && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for backend to initialize
echo [INFO] Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo [INFO] Starting Frontend Development Server...
start "DrugGuard Frontend" cmd /k "cd /d %ROOT_DIR%frontend && npm run dev"

echo.
echo ============================================================
echo   BUILD COMPLETE - All Platforms Ready!
echo ============================================================
echo.
echo   ANDROID:
echo     APK Location: %ROOT_DIR%DrugGuard.apk
echo     Install: adb install DrugGuard.apk
echo.
echo   WINDOWS (Development):
echo     Backend API:  http://localhost:8000
echo     API Docs:     http://localhost:8000/docs
echo     Frontend:     http://localhost:5173
echo.
echo   NOTE: For Android to connect to backend:
echo     1. Find your PC's IP: ipconfig
echo     2. Update frontend/.env with VITE_API_URL_MOBILE
echo     3. Rebuild: npm run build ^&^& npx cap sync android
echo.
echo ============================================================
echo.

pause
