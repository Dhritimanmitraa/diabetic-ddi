# DrugGuard

**AI-Powered Drug Interaction Checker for Diabetic Patients**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.2-61DAFB?style=flat&logo=react&logoColor=black)
![Google Gemini](https://img.shields.io/badge/Gemini-Vision%20AI-4285F4?style=flat&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

DrugGuard is a clinical decision support system that helps diabetic patients and healthcare providers identify dangerous drug interactions. It combines:

- **Rule-Based Analysis** - 180+ clinical rules from ADA/AACE guidelines
- **ML Models** - XGBoost, Random Forest, LightGBM trained on 2M+ TWOSIDES interactions
- **LLM Intelligence** - Google Gemini for natural language explanations
- **Patient-Specific Context** - Personalized risk based on eGFR, HbA1c, and complications

---

## Features

| Feature | Description |
|---------|-------------|
| 🩺 **Patient Profiles** | Track diabetes type, labs, complications, and medications |
| ⚠️ **Drug Risk Assessment** | Instant risk level (safe → fatal) with explanations |
| 🔬 **Lab Report Analysis** | Upload lab reports for Gemini Vision extraction |
| 📷 **Prescription Scanner** | OCR/camera capture to extract medications |
| 💊 **Alternative Suggestions** | Safer drug options for high-risk prescriptions |
| 📄 **PDF Reports** | Downloadable patient safety reports |
| 🤖 **RAG Chat** | Ask questions about prescriptions using AI |
| 🎤 **Voice Input** | Voice-enabled drug search |
| 📱 **Mobile Ready** | Android APK available via Capacitor |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Tesseract OCR ([Install Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html))
- Google Gemini API Key (for AI features)

### Run the App

```bash
# Clone
git clone https://github.com/Dhritimanmitraa/diabetic-ddi.git
cd diabetic-ddi

# Windows - One-click start
start_app.bat
```

**Manual Setup:**

```bash
# Backend
cd backend
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Create .env file with:
# GEMINI_API_KEY=your_key_here

uvicorn app.main:app --port 8001 --reload

# Frontend (new terminal)
cd frontend
npm install && npm run dev
```

Open `http://localhost:3000`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                   │
│  DiabetesManager │ PrescriptionRAG │ InteractionChecker     │
│  CameraCapture │ VoiceInput │ MedicationSchedule            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Diabetic   │ Prescription │     ML       │   Services     │
│    Module    │    Module    │   Module     │                │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ • Rules (180+)│ • RAG Chat  │ • XGBoost    │ • OCR Service  │
│ • LLM Explainer│ • Vision OCR│ • Random Forest│ • PDF Generator│
│ • ML Predictor│ • LangGraph │ • LightGBM   │ • Robust Fetcher│
│ • Lab Analyzer│ • LangChain │ • Bayesian Opt│ • Cache/Redis  │
└──────────────┴──────────────┴──────────────┴────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              SQLite Database + ChromaDB (RAG)                │
│         Drug Data │ Patient Profiles │ TWOSIDES/OFFSIDES     │
└─────────────────────────────────────────────────────────────┘
```

### ML Performance

| Metric | Value |
|--------|-------|
| NPV | 99.41% (critical for safety) |
| Sensitivity | 85%+ |
| Dataset | TWOSIDES (2M+ interactions) |
| Explainability | SHAP + LIME |

---

## API Reference

### Diabetic Patient Module (`/diabetic`)

```
POST   /diabetic/patients              Create patient profile
GET    /diabetic/patients              List all patients
GET    /diabetic/patients/{id}         Get patient by ID
PUT    /diabetic/patients/{id}         Update patient
DELETE /diabetic/patients/{id}         Delete patient

POST   /diabetic/medications/{id}      Add medication
GET    /diabetic/medications/{id}      List patient medications

POST   /diabetic/risk-check            Check drug risk (fast response)
POST   /diabetic/risk-check/llm        Get LLM analysis (detailed)
POST   /diabetic/medication-list/check Check all medications
POST   /diabetic/alternatives          Find safer alternatives

GET    /diabetic/report/{id}           Get DDI report
GET    /diabetic/report/{id}/pdf       Download PDF report

POST   /diabetic/lab-report/analyze    Analyze lab report (Gemini Vision)
```

### Prescription RAG Module (`/prescription`)

```
POST   /prescription/upload            Upload prescription image/PDF
POST   /prescription/upload/base64     Upload base64 image (mobile)
GET    /prescription/list              List prescriptions
GET    /prescription/{id}              Get prescription
DELETE /prescription/{id}              Delete prescription

POST   /prescription/chat              Ask about prescription (RAG)
GET    /prescription/{id}/history      Get chat history
POST   /prescription/interactions      Check drug interactions
```

### Drug & Interaction APIs

```
GET    /drugs                          List all drugs
GET    /drugs/search?query=            Search drugs by name
GET    /drugs/{id}                     Get drug details
GET    /drugs/{name}/side-effects      Get side effects

POST   /interactions/check             Check interaction (drug pair)
GET    /interactions/check/{d1}/{d2}   Check interaction (GET)
GET    /interactions/drug/{name}       All interactions for a drug

POST   /alternatives                   Get safe alternatives
```

### Health & Stats

```
GET    /                               API status
GET    /health                         Health check
GET    /health/apis                    External API status
GET    /stats                          Database statistics
```

---

## Project Structure

```
diabetic-ddi/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app entry
│   │   ├── diabetic/                  # Diabetic DDI module
│   │   │   ├── router.py              # 43 API endpoints
│   │   │   ├── service.py             # Business logic
│   │   │   ├── rules.py               # 180+ clinical rules
│   │   │   ├── llm_explainer.py       # Gemini explanations
│   │   │   ├── ml_predictor.py        # ML predictions
│   │   │   └── lab_report_analyzer.py # Vision AI extraction
│   │   ├── prescription/              # Prescription RAG
│   │   │   ├── router.py              # RAG endpoints
│   │   │   ├── rag_service.py         # RAG orchestration
│   │   │   ├── langgraph_rag.py       # LangGraph agent
│   │   │   └── vision_ocr.py          # Gemini Vision OCR
│   │   ├── ml/                        # ML models
│   │   │   ├── predictor.py           # Model inference
│   │   │   ├── trainer.py             # Model training
│   │   │   ├── explainability_service.py # SHAP/LIME
│   │   │   └── bayesian_optimizer.py  # Hyperparameter tuning
│   │   └── services/                  # Shared services
│   │       ├── ocr_service.py         # Tesseract OCR
│   │       ├── pdf_generator.py       # Report PDFs
│   │       └── robust_fetcher.py      # API resilience
│   └── scripts/
│       └── train_diabetic_model.py    # Model training script
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # Main app
│   │   ├── components/
│   │   │   ├── DiabetesManager.jsx    # Patient management
│   │   │   ├── PrescriptionRAG.jsx    # Prescription chat
│   │   │   ├── InteractionChecker.jsx # Drug interaction UI
│   │   │   ├── CameraCapture.jsx      # Camera OCR
│   │   │   ├── VoiceInput.jsx         # Voice search
│   │   │   └── ...                    # 20 components total
│   │   └── services/
│   │       └── api.js                 # API client
│   └── android/                       # Capacitor Android
├── ml-standalone/                     # Standalone ML demo
├── DrugGuard.apk                      # Android build
└── start_app.bat                      # One-click launcher
```

---

## Training Models

```bash
cd backend

# Train diabetic-specific ML models
python scripts/train_diabetic_model.py
```

The training script uses:
- **Bayesian optimization** for hyperparameter tuning
- **SMOTE** for handling class imbalance
- **Cross-validation** with stratified splits

---

## Tech Stack

### Backend
- **FastAPI** - Async REST API
- **SQLAlchemy** - Database ORM (SQLite/PostgreSQL)
- **Google Gemini** - Vision AI & LLM explanations
- **ChromaDB** - Vector store for RAG
- **LangChain/LangGraph** - RAG orchestration
- **XGBoost/LightGBM/scikit-learn** - ML models
- **SHAP/LIME** - Model explainability
- **Tesseract/TrOCR** - OCR engines
- **Redis** - Caching (optional)

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool
- **Framer Motion** - Animations
- **Tailwind CSS** - Styling
- **Capacitor** - Android/iOS builds
- **Tesseract.js** - Client-side OCR fallback

---

## Environment Variables

Create `backend/.env`:

```env
# Required for AI features
GEMINI_API_KEY=your_google_gemini_api_key

# Optional
DATABASE_URL=sqlite+aiosqlite:///./drug_interactions.db
REDIS_URL=redis://localhost:6379
DEBUG=false
```

---

## Medical Disclaimer

> [!CAUTION]
> **This software is for research and educational purposes only.**
>
> Not a substitute for professional medical advice. Always consult healthcare providers before making medication decisions.

---

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Dhritiman Mitra**  
[GitHub](https://github.com/Dhritimanmitraa)
