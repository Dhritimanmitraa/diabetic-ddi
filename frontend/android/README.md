# DrugGuard Android Mobile App

This is the Android version of DrugGuard, built using Capacitor.

## Prerequisites

- **Android Studio** - [Download](https://developer.android.com/studio)
- **JDK 17+** - Usually bundled with Android Studio
- **Android SDK 33+** - Install via Android Studio SDK Manager

## Building the Android App

### 1. First-time Setup

```bash
cd frontend

# Install dependencies (if not done)
npm install

# Build web assets
npm run build

# Sync to Android
npx cap sync android
```

### 2. Open in Android Studio

```bash
npx cap open android
```

Or manually open `frontend/android` folder in Android Studio.

### 3. Configure Backend URL

For the app to communicate with your backend:

1. Find your computer's local IP:
   - Windows: Run `ipconfig` in Command Prompt
   - Look for `IPv4 Address` (e.g., `192.168.1.100`)

2. Create/update `frontend/.env`:
   ```
   VITE_API_URL=http://localhost:8000
   VITE_API_URL_MOBILE=http://YOUR_IP:8001
   ```

3. Rebuild: `npm run build && npx cap sync android`

### 4. Run on Device/Emulator

In Android Studio:
- Select your device or emulator
- Click **Run** (green play button)

## Building APK for Distribution

### Debug APK
```bash
cd frontend/android
./gradlew assembleDebug
```
Output: `frontend/android/app/build/outputs/apk/debug/app-debug.apk`

### Release APK (requires signing)
```bash
cd frontend/android
./gradlew assembleRelease
```

## Features

- ✅ Drug interaction checking
- ✅ Diabetic patient management
- ✅ Native camera for medication scanning
- ✅ Prescription upload & RAG chat
- ✅ ML-powered predictions
- ✅ Safe alternative suggestions

## Permissions

The app requests:
- **INTERNET** - API communication
- **CAMERA** - Medication label scanning
