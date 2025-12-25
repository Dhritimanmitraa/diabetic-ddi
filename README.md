# DrugGuard - Diabetic Drug Interaction Checker

A comprehensive clinical decision support system for assessing drug safety in diabetic patients. This hybrid system combines evidence-based clinical rules with machine learning models to prevent medication-related harm in diabetic populations.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![React](https://img.shields.io/badge/React-18.2-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

Diabetes patients face unique medication risks due to altered pharmacokinetics, renal impairment, and drug interactions that can lead to severe adverse events. Traditional drug interaction checkers fail to account for diabetes-specific factors like kidney function (eGFR), diabetic complications, and glucose-altering effects.

DrugGuard addresses this gap by integrating:
- **Evidence-based clinical rules** from ADA and AACE guidelines
- **Machine learning models** trained on 2M+ drug interaction records
- **Patient-specific risk assessment** considering eGFR, complications, and comorbidities
- **Actionable recommendations** with dose adjustments and safer alternatives

## Key Features

### For Diabetic Patients
- **Renal Function Assessment**: Automatic eGFR-based dose adjustments
- **Contraindication Detection**: Flags fatal drug combinations (e.g., metformin + contrast dye)
- **Hypoglycemia Risk**: Identifies drugs that mask hypoglycemia symptoms
- **Complication-Aware**: Considers nephropathy, retinopathy, cardiovascular disease
- **Potassium Monitoring**: Alerts for hyperkalemia risks with ACE inhibitors/ARBs

### Technical Capabilities
- **Hybrid Architecture**: Rule-based system with ML augmentation
- **Ensemble ML Models**: Random Forest, XGBoost, and LightGBM
- **OCR Integration**: Scan medication labels using camera
- **Real-time API**: FastAPI backend with async support
- **Modern UI**: React frontend with Tailwind CSS

## Architecture

```
┌─────────────────┐
│   React UI      │  ← User Interface
└────────┬────────┘
         │
┌────────▼────────┐
│   FastAPI       │  ← REST API
│   Backend       │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│ Rules │ │  ML   │  ← Decision Engine
│ Engine│ │Models │
└───────┘ └───────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Tesseract OCR ([Installation Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Dhritimanmitraa/diabetic-ddi.git
cd diabetic-ddi
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
cd frontend
npm install
```

4. **Run the Application**

**Windows:**
```bash
# From project root
start_app.bat
```

**Manual Start:**
```bash
# Terminal 1 - Backend
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2 - Frontend
cd frontend
npm run dev
```

Visit `http://localhost:3000` to access the application.

## Machine Learning Models

The system uses ensemble models trained on the TWOSIDES database:

- **Training Data**: 2M+ drug-drug interaction records
- **Models**: Random Forest, XGBoost, LightGBM
- **Optimization**: Bayesian hyperparameter tuning
- **Class Imbalance**: SMOTE/ADASYN resampling
- **Threshold Tuning**: Optimal threshold for safety-critical predictions

### Model Performance
- **NPV (Negative Predictive Value)**: 99.41% - Critical for safety
- **Sensitivity**: 85%+ for detecting interactions
- **Specificity**: 90%+ for safe drug pairs

## Data Sources

- **TWOSIDES**: 2M+ drug-drug interaction records
- **OFFSIDES**: Adverse event database
- **Clinical Guidelines**: ADA, AACE recommendations
- **DrugBank**: Drug information and interactions
- **RxNorm**: Standardized drug nomenclature

## Project Structure

```
diabetic-ddi/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── models.py            # Database models
│   │   ├── schemas.py           # Pydantic schemas
│   │   ├── diabetic/            # Diabetic-specific logic
│   │   │   ├── rules.py         # Clinical rules engine
│   │   │   ├── service.py       # Business logic
│   │   │   └── ml_predictor.py  # ML predictions
│   │   ├── ml/                  # ML components
│   │   │   ├── predictor.py     # Model inference
│   │   │   ├── trainer.py       # Model training
│   │   │   └── feature_engineering.py
│   │   └── services/
│   │       ├── interaction_service.py
│   │       └── ocr_service.py
│   ├── scripts/
│   │   ├── train_twosides_ml.py
│   │   ├── train_diabetic_ml.py
│   │   └── seed_demo_patients.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DiabetesManager.jsx
│   │   │   ├── InteractionChecker.jsx
│   │   │   └── ModelDashboard.jsx
│   │   └── services/
│   │       └── api.js
│   └── package.json
│
└── README.md
```

## API Endpoints

### Diabetic Patient Management
- `POST /diabetic/patients` - Create patient profile
- `GET /diabetic/patients/{id}` - Get patient details
- `POST /diabetic/risk-check` - Check medication safety
- `POST /diabetic/alternatives` - Get safer alternatives

### General Drug Interactions
- `POST /interactions/check` - Check drug-drug interaction
- `GET /interactions/drug/{name}` - Get all interactions for a drug
- `POST /alternatives` - Find safe alternatives

### OCR & Image Processing
- `POST /ocr/extract` - Extract drugs from image
- `POST /ocr/upload` - Upload medication label

### ML Dashboard
- `GET /ml/metrics` - Model performance metrics
- `GET /ml/predictions` - Recent predictions

## Training Models

To train the ML models from scratch:

```bash
cd backend

# Train general DDI models
python scripts/train_twosides_ml.py

# Train diabetic-specific models
python scripts/train_diabetic_ml.py

# Find optimal threshold
python scripts/find_optimal_threshold.py
```

## Example Usage

### Check Drug Safety for Diabetic Patient

```python
import requests

# Create patient profile
patient_data = {
    "name": "John Doe",
    "age": 65,
    "egfr": 45,  # Stage 3 CKD
    "has_nephropathy": True,
    "current_medications": ["Metformin", "Lisinopril"]
}

response = requests.post(
    "http://localhost:8001/diabetic/risk-check",
    json={
        "patient_id": 1,
        "new_drug": "Contrast Dye"
    }
)

print(response.json())
# Returns: Contraindicated - Metformin must be stopped 48h before contrast
```

## Safety Features

1. **Rule Priority**: Clinical rules always override ML predictions for critical alerts
2. **Conservative Thresholds**: Optimized for high NPV to minimize false negatives
3. **Multiple Validation**: Cross-references multiple data sources
4. **Audit Logging**: All predictions and decisions are logged

## Medical Disclaimer

**This software is for research and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.**

Always consult with qualified healthcare providers before making medication decisions. The absence of an interaction in this system does not guarantee safety.

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Dhritiman Mitra**

## Acknowledgments

- TWOSIDES database for interaction data
- Clinical guidelines from ADA and AACE
- OpenFDA and DrugBank for drug information

---

**Made for diabetic patient safety**
