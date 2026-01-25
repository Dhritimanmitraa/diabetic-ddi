# DrugGuard

**Diabetic Drug Interaction Checker - A Clinical Decision Support System**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.2-61DAFB?style=flat&logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## What It Does

DrugGuard assesses drug safety for diabetic patients by combining:

- **Clinical Rules** from ADA/AACE guidelines
- **ML Models** trained on 2M+ drug interaction records (TWOSIDES database)
- **Patient-Specific Analysis** based on eGFR, complications, and comorbidities

### Core Features

| Feature | Description |
|---------|-------------|
| **Renal-Aware Dosing** | Automatic eGFR-based dose adjustments |
| **Contraindication Detection** | Flags fatal drug combinations |
| **Hypoglycemia Risk** | Identifies drugs masking hypo symptoms |
| **Prescription Scanner** | OCR/Camera to extract medications |
| **Alternative Suggestions** | Recommends safer drug options |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Tesseract OCR ([Install Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html))

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
uvicorn app.main:app --port 8001 --reload

# Frontend (new terminal)
cd frontend
npm install && npm run dev
```

Open `http://localhost:3000`

---

## Architecture

```
Frontend (React)
      │
      ▼
Backend (FastAPI)
      │
  ┌───┴───┐
  │       │
Rules   ML Models
Engine  (XGBoost, RF, LGBM)
```

**ML Performance:**
- NPV: 99.41% (critical for safety)
- Sensitivity: 85%+
- Trained on: TWOSIDES (2M+ interactions)

---

## API Reference

### Patient Management
```
POST /diabetic/patients          Create patient
GET  /diabetic/patients/{id}     Get patient
POST /diabetic/risk-check        Check drug safety
POST /diabetic/alternatives      Get safer options
```

### Drug Interactions
```
POST /interactions/check         Check interaction
GET  /interactions/drug/{name}   Get all interactions
```

### OCR
```
POST /prescription/upload        Extract from image
```

---

## Project Structure

```
diabetic-ddi/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI app
│   │   ├── diabetic/         # DDI logic & rules
│   │   └── ml/               # ML models
│   └── scripts/              # Training scripts
├── frontend/
│   ├── src/components/       # React components
│   └── src/services/         # API client
└── ml-standalone/            # Standalone ML demo
```

---

## Training Models

```bash
cd backend

# Train DDI models
python scripts/train_twosides_ml.py

# Train diabetic-specific models
python scripts/train_diabetic_ml.py
```

---

## Medical Disclaimer

**This software is for research and educational purposes only.**

Not a substitute for professional medical advice. Always consult healthcare providers before making medication decisions.

---

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Dhritiman Mitra**
