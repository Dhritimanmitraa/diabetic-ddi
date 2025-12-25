# Complete Project Structure Explanation

## 📁 Backend Directory (`backend/`)

### 🗄️ **Core Application Files**

#### `backend/app/main.py`
**Purpose**: Main FastAPI application entry point - the heart of your backend API  
**What it does**:
- Creates the FastAPI app instance with metadata and documentation
- Sets up CORS middleware for frontend communication
- Defines all API endpoints:
  - **Health endpoints**: `/`, `/health`, `/health/live`, `/health/ready`
  - **Drug endpoints**: `/drugs`, `/drugs/search`, `/drugs/{id}`, `/drugs/name/{name}`
  - **Interaction endpoints**: `/interactions/check`, `/interactions/drug/{name}`
  - **Alternative suggestions**: `/alternatives`
  - **OCR endpoints**: `/ocr/extract`, `/ocr/upload`
  - **History endpoints**: `/history`, `/history/stats`, `/history/dangerous`
  - **ML endpoints**: `/ml/predict`, `/ml/model-info`, `/ml/comparison`, `/ml/train`
  - **Job endpoints**: `/jobs/{job_id}`, `/jobs/fetch-data`
- Includes the diabetic patient router (`/diabetes/*` routes)
- Logs all drug search/browse requests to console
- Initializes database on startup
- Handles hybrid decision-making (rules + ML predictions)

#### `backend/app/config.py`
**Purpose**: Application configuration management  
**What it does**:
- Defines all settings using Pydantic (type-safe config)
- Settings include:
  - App name/version
  - Database URL (SQLite by default)
  - Redis URL (for background jobs)
  - API keys (OpenFDA, etc.)
  - Rate limiting thresholds
  - Tesseract OCR path
  - Data directory paths
- Loads from `.env` file if present
- Uses `@lru_cache` for efficient singleton pattern

#### `backend/app/database.py`
**Purpose**: Database connection and session management  
**What it does**:
- Creates async SQLAlchemy engine (connects to SQLite database)
- Creates async session factory (`async_sessionmaker`)
- Provides `get_db()` dependency for FastAPI endpoints
- Defines `Base` class for all SQLAlchemy models
- `init_db()` function creates all tables on startup

#### `backend/app/models.py`
**Purpose**: Database schema definitions (SQLAlchemy ORM models)  
**What it does**:
- Defines all database tables as Python classes:
  - **`Drug`**: Stores drug information (name, generic name, class, mechanism, etc.)
  - **`Category`**: Drug categories/therapeutic classes
  - **`DrugInteraction`**: Drug-drug interaction records with severity
  - **`DrugSimilarity`**: Similarity scores for alternative suggestions
  - **`ComparisonLog`**: Logs all user drug checks (with ML audit fields)
  - **`MLPrediction`**: Stores ML prediction results for analysis
  - **`ModelMetrics`**: Tracks ML model performance over time
  - **`OptimizationResult`**: Stores hyperparameter optimization results
  - **`TwosidesInteraction`**: TWOSIDES dataset interactions
  - **`OffsidesEffect`**: OffSIDES single-drug side effects
- Defines relationships between tables (foreign keys, many-to-many)
- Includes indexes for performance

#### `backend/app/schemas.py`
**Purpose**: Pydantic schemas for API request/response validation  
**What it does**:
- Defines data validation models for:
  - **Drug schemas**: `DrugBase`, `DrugCreate`, `DrugResponse`, `DrugSearch`
  - **Interaction schemas**: `InteractionBase`, `InteractionCreate`, `InteractionResponse`
  - **Request/Response schemas**: `InteractionCheckRequest`, `InteractionCheckResponse`
  - **Alternative schemas**: `AlternativeDrug`, `AlternativeSuggestionResponse`
  - **OCR schemas**: `OCRRequest`, `OCRResponse`
  - **Statistics**: `DatabaseStats`
- Enums: `SeverityLevel`, `EvidenceLevel`
- Validates input data before processing
- Converts database models to JSON responses

---

### 🧠 **Machine Learning Module** (`backend/app/ml/`)

#### `backend/app/ml/predictor.py`
**Purpose**: ML prediction service - loads trained models and makes predictions  
**What it does**:
- `DDIPredictor` class:
  - Loads trained models (Random Forest, XGBoost, LightGBM) from disk
  - Loads optimal threshold from `optimal_threshold.json`
  - `extract_features_simple()`: Converts drug info to feature vectors (hash encoding)
  - `predict()`: Makes ensemble predictions using all 3 models
  - Uses optimal threshold (not default 0.5) for binary classification
  - Maps probabilities to severity levels
- `PredictionResult` class: Container for prediction results
- Singleton pattern: `get_predictor()` returns cached instance

#### `backend/app/ml/feature_engineering.py`
**Purpose**: Feature extraction for ML models  
**What it does**:
- Converts drug properties (name, class, mechanism) into numerical features
- Uses hash encoding for categorical/text features
- Creates feature vectors compatible with trained models

#### `backend/app/ml/models.py`
**Purpose**: ML model definitions and wrappers  
**What it does**:
- Defines model architectures
- Wraps scikit-learn models for consistent interface

#### `backend/app/ml/trainer.py`
**Purpose**: Model training logic  
**What it does**:
- Trains Random Forest, XGBoost, LightGBM models
- Handles hyperparameter optimization
- Saves trained models to disk

#### `backend/app/ml/bayesian_optimizer.py`
**Purpose**: Bayesian hyperparameter optimization using Optuna  
**What it does**:
- Uses Tree-structured Parzen Estimator (TPE) for efficient hyperparameter search
- Optimizes model parameters to maximize performance
- Compares with grid search and random search

---

### 🩺 **Diabetic Patient Module** (`backend/app/diabetic/`)

#### `backend/app/diabetic/service.py`
**Purpose**: Main service for diabetic patient drug risk assessment  
**What it does**:
- `DiabeticDDIService` class:
  - **Patient management**: Create, read, update, delete patient profiles
  - **Medication management**: Add/remove medications from patient's list
  - **Drug risk checking**: `check_drug_risk()` - checks single drug against patient profile
  - **Bulk checking**: `check_all_medications()` - checks all patient's meds at once
  - **Report generation**: Creates comprehensive DDI reports
  - **Alternative suggestions**: Finds safer alternatives
- **Key logic**: Rules are PRIMARY, ML is SUPPLEMENTARY
- Logs ML disagreements for monitoring

#### `backend/app/diabetic/rules.py`
**Purpose**: Clinical rule engine for diabetic drug safety  
**What it does**:
- `DiabeticDrugRules` class with extensive clinical knowledge:
  - **eGFR contraindications**: 50+ drugs with renal thresholds (Metformin, Verapamil, etc.)
  - **Drug class patterns**: Matches entire classes (`-pril`, `-sartan`, `-olol`, etc.)
  - **Severe CKD warnings**: General warnings for unknown drugs in severe CKD
  - **Hypoglycemia risk drugs**: Sulfonylureas, insulin, etc.
  - **Hyperglycemia risk drugs**: Corticosteroids, thiazides, etc.
  - **Nephrotoxic drugs**: NSAIDs, aminoglycosides, etc.
- `assess_drug_risk()`: Main function that evaluates drug safety
- Returns `RiskAssessment` with risk level, score, factors, recommendations

#### `backend/app/diabetic/models.py`
**Purpose**: Database models for diabetic patients  
**What it does**:
- **`DiabeticPatient`**: Stores patient profile (age, diabetes type, labs, complications)
- **`DiabeticMedication`**: Patient's current medications
- **`DiabeticDrugRisk`**: Cached risk assessments
- **`DiabeticDrugRule`**: Custom rules (if needed)

#### `backend/app/diabetic/schemas.py`
**Purpose**: Pydantic schemas for diabetic module API  
**What it does**:
- Request/response models for:
  - Patient creation/updates
  - Medication management
  - Drug risk checks
  - Reports and alternatives

#### `backend/app/diabetic/router.py`
**Purpose**: FastAPI routes for diabetic patient endpoints  
**What it does**:
- Defines API endpoints:
  - `/diabetes/patients` - Create/list patients
  - `/diabetes/patients/{id}` - Get/update patient
  - `/diabetes/patients/{id}/medications` - Manage medications
  - `/diabetes/patients/{id}/check-drug` - Check single drug
  - `/diabetes/patients/{id}/check-all` - Check all meds
  - `/diabetes/patients/{id}/report` - Generate report
  - `/diabetes/patients/{id}/alternatives` - Find alternatives

#### `backend/app/diabetic/ml_predictor.py`
**Purpose**: ML predictor specifically for diabetic drug risk  
**What it does**:
- Loads diabetic-specific ML model (`diabetic_risk_model.pkl`)
- Makes predictions considering patient factors (eGFR, complications, etc.)

#### `backend/app/diabetic/validation.py`
**Purpose**: Input validation for diabetic module  
**What it does**:
- Validates patient data (age, labs, etc.)
- Ensures data consistency

#### `backend/app/diabetic/data/diabetes_medications.json`
**Purpose**: Reference data for diabetes medications  
**What it does**:
- Contains list of common diabetes medications
- Used for autocomplete and validation

---

### 🔧 **Services Module** (`backend/app/services/`)

#### `backend/app/services/interaction_service.py`
**Purpose**: Core drug interaction checking logic  
**What it does**:
- `InteractionService` class:
  - `check_interaction()`: Checks if two drugs interact
  - `search_drugs()`: Searches drugs by name
  - `get_drug_by_name()`: Gets drug details
  - `get_all_interactions_for_drug()`: Lists all interactions for a drug
  - `find_alternatives()`: Suggests safer alternatives

#### `backend/app/services/comparison_logger.py`
**Purpose**: Logs all drug comparison queries  
**What it does**:
- `ComparisonLogger` class:
  - `log_comparison()`: Saves each check to database
  - `get_comparisons()`: Retrieves history with pagination/filtering
  - `get_comparison_stats()`: Statistics about checks
  - `get_dangerous_combinations_found()`: Lists dangerous pairs users checked
  - `get_most_checked_drugs()`: Popular drugs

#### `backend/app/services/ocr_service.py`
**Purpose**: Optical Character Recognition for drug labels  
**What it does**:
- `OCRService` class:
  - `extract_from_base64()`: Extracts text from base64 image
  - Uses Tesseract OCR to read medication labels
  - Detects drug names from extracted text
  - Returns confidence scores

#### `backend/app/services/data_fetcher.py`
**Purpose**: Fetches drug data from external APIs  
**What it does**:
- Fetches from DrugBank, OpenFDA, NIH APIs
- Populates database with real drug data

#### `backend/app/services/cache.py`
**Purpose**: Caching layer for performance  
**What it does**:
- Caches frequently accessed data
- Reduces database queries

#### `backend/app/services/auth.py`
**Purpose**: API key authentication  
**What it does**:
- `require_api_key()`: Dependency for protected endpoints
- Validates API keys for admin operations

#### `backend/app/services/rate_limiter.py`
**Purpose**: Rate limiting to prevent abuse  
**What it does**:
- `rate_limit()`: Limits requests per minute
- Uses Redis for distributed rate limiting

#### `backend/app/services/tasks.py`
**Purpose**: Background job management  
**What it does**:
- `enqueue_training()`: Queues ML model training
- `enqueue_data_refresh()`: Queues data updates
- `get_job()`: Gets job status
- Uses RQ (Redis Queue) for async tasks

---

### 📜 **Scripts Directory** (`backend/scripts/`)

#### `backend/scripts/build_training_set.py`
**Purpose**: Builds ML training dataset from TWOSIDES  
**What it does**:
- Reads TWOSIDES interactions from database
- Creates positive samples (known interactions)
- Generates negative samples (random drug pairs)
- Splits into train/val/test sets
- **Streaming approach**: Writes directly to CSV files (no memory accumulation)
- Handles class imbalance by capping positives

#### `backend/scripts/train_twosides_ml.py`
**Purpose**: Trains ML models on TWOSIDES data  
**What it does**:
- Loads training data
- Trains Random Forest (`class_weight='balanced'`)
- Trains XGBoost (`scale_pos_weight` for imbalance)
- Trains LightGBM (`scale_pos_weight`)
- Saves models to `models/` directory
- Uses optimal threshold from evaluation

#### `backend/scripts/evaluate_models.py`
**Purpose**: Evaluates trained models on test set  
**What it does**:
- Loads test data
- Makes predictions with all models
- Calculates metrics: Accuracy, Precision, Recall, F1, AUC-ROC, NPV, PPV
- `find_optimal_threshold()`: Uses G-Mean and Youden's J to find best threshold
- Saves evaluation results to JSON

#### `backend/scripts/find_optimal_threshold.py`
**Purpose**: Dedicated script to find optimal classification threshold  
**What it does**:
- Loads balanced test sample (10k positive + 10k negative)
- Computes ensemble probabilities
- Finds optimal threshold using G-Mean and Youden's J
- Compares different thresholds
- Saves to `models/optimal_threshold.json`

#### `backend/scripts/map_twosides_severity.py`
**Purpose**: Maps severity levels to TWOSIDES interactions  
**What it does**:
- Reads TWOSIDES interactions
- Maps effect descriptions to severity (minor/moderate/major/contraindicated/fatal)
- Uses cursor-based pagination for performance
- Batch updates using CASE statements
- Creates index on `severity` column

#### `backend/scripts/load_twosides.py`
**Purpose**: Loads TWOSIDES dataset into database  
**What it does**:
- Reads `TWOSIDES.csv.gz` file
- Parses drug pairs and effects
- Inserts into `twosides_interactions` table
- Handles large file efficiently

#### `backend/scripts/load_offsides.py`
**Purpose**: Loads OffSIDES dataset into database  
**What it does**:
- Reads `OFFSIDES.csv` file
- Parses single-drug side effects
- Inserts into `offsides_effects` table

#### `backend/scripts/map_offsides_severity.py`
**Purpose**: Maps severity to OffSIDES effects  
**What it does**:
- Similar to `map_twosides_severity.py` but for OffSIDES

#### `backend/scripts/build_diabetic_pseudolabels.py`
**Purpose**: Builds training data for diabetic-specific ML model  
**What it does**:
- Creates labeled examples for diabetic drug risk
- Uses rule engine to generate pseudo-labels

#### `backend/scripts/train_diabetic_ml.py`
**Purpose**: Trains diabetic-specific ML model  
**What it does**:
- Trains model on diabetic patient data
- Saves to `models/diabetic_risk_model.pkl`

#### `backend/scripts/train_models.py`
**Purpose**: Legacy training script (may be deprecated)  
**What it does**:
- Older training approach

#### `backend/scripts/fetch_real_data.py`
**Purpose**: Fetches real drug data from APIs  
**What it does**:
- Calls DrugBank, OpenFDA APIs
- Populates `drugs` and `drug_interactions` tables

#### `backend/scripts/rq_worker.py`
**Purpose**: Background worker for RQ jobs  
**What it does**:
- Runs background tasks (training, data refresh)
- Must be run separately: `python -m scripts.rq_worker`

---

### 📊 **Data Directory** (`backend/data/`)

#### `backend/data/training/`
**Purpose**: ML training datasets  
**Files**:
- `train.csv`, `val.csv`, `test.csv`: Train/validation/test splits
- `metadata.json`: Dataset statistics

#### `backend/data/diabetic/training/`
**Purpose**: Diabetic-specific training data  
**Files**:
- `train.csv`, `val.csv`, `test.csv`: Diabetic model training sets
- `full_dataset.csv`: Complete dataset
- `metadata.json`: Metadata

#### `backend/data/twosides/TWOSIDES.csv.gz`
**Purpose**: TWOSIDES dataset (compressed)  
**What it is**: Large dataset of drug-drug interactions from FDA adverse event reports

#### `backend/data/offsides/OFFSIDES.csv`
**Purpose**: OffSIDES dataset  
**What it is**: Single-drug side effect signals

#### `backend/data/mimiciv/`
**Purpose**: MIMIC-IV clinical database demo  
**What it is**: Hospital data for research (not actively used in current system)

#### `backend/data/interactions.db`
**Purpose**: Alternative database file (may be legacy)

#### `backend/data/real_drug_data.json`
**Purpose**: Cached real drug data from APIs

---

### 🤖 **Models Directory** (`backend/models/`)

**Purpose**: Trained ML models and metadata

**Files**:
- `random_forest_model.pkl`: Trained Random Forest model
- `xgboost_model.pkl`: Trained XGBoost model
- `lightgbm_model.pkl`: Trained LightGBM model
- `diabetic_risk_model.pkl`: Diabetic-specific model
- `feature_extractor.pkl`: Feature extraction pipeline
- `optimal_threshold.json`: Optimal classification threshold (e.g., 0.35)
- `training_results.json`: Training metrics
- `evaluation_results.json`: Test set evaluation metrics
- `*_comparison.json`: Hyperparameter optimization comparisons

---

### 📝 **Other Backend Files**

#### `backend/requirements.txt`
**Purpose**: Python dependencies  
**What it contains**:
- FastAPI, Uvicorn (web framework)
- SQLAlchemy, aiosqlite (database)
- Pandas, NumPy (data processing)
- scikit-learn, XGBoost, LightGBM (ML)
- Optuna (hyperparameter optimization)
- Tesseract, OpenCV (OCR)
- Redis, RQ (background jobs)
- And more...

#### `backend/drug_interactions.db`
**Purpose**: Main SQLite database (26GB!)  
**What it contains**:
- All drugs, interactions, patient data
- Comparison logs, ML predictions
- Model metrics

#### `backend/MODEL_ACCURACY_REPORT.md`
**Purpose**: Documentation of model performance  
**What it contains**:
- Accuracy metrics
- Evaluation results
- Model comparison

#### `backend/map_severity.log`
**Purpose**: Log file from severity mapping script  
**What it contains**:
- Progress logs from `map_twosides_severity.py`

#### `backend/body.json`, `backend/cols.txt`
**Purpose**: Temporary/debug files (can be deleted)

---

## 🎨 Frontend Directory (`frontend/`)

### 🚀 **Core Application Files**

#### `frontend/package.json`
**Purpose**: Node.js dependencies and scripts  
**What it contains**:
- **Dependencies**:
  - React, React-DOM (UI framework)
  - React Router (routing)
  - Axios (HTTP client)
  - Framer Motion (animations)
  - React Webcam (camera access)
  - Tesseract.js (client-side OCR)
  - Lucide React (icons)
  - React Hot Toast (notifications)
- **Dev Dependencies**:
  - Vite (build tool)
  - Tailwind CSS (styling)
  - ESLint (linting)
- **Scripts**:
  - `npm run dev`: Start development server
  - `npm run build`: Build for production
  - `npm run preview`: Preview production build

#### `frontend/vite.config.js`
**Purpose**: Vite build tool configuration  
**What it does**:
- Configures React plugin
- Sets dev server port to 3000
- Proxies `/api` requests to `http://localhost:8000` (backend)

#### `frontend/index.html`
**Purpose**: HTML entry point  
**What it does**:
- Root HTML file
- Loads React app via `<div id="root">`
- Links to `main.jsx`

---

### 📱 **Source Files** (`frontend/src/`)

#### `frontend/src/main.jsx`
**Purpose**: React application entry point  
**What it does**:
- Renders `<App />` component into DOM
- Imports global CSS
- Wraps in `React.StrictMode` for development checks

#### `frontend/src/App.jsx`
**Purpose**: Main React component - application router  
**What it does**:
- Sets up React Router with routes:
  - `/` - Main interaction checker page
  - `/ml-dashboard` - ML model dashboard
  - `/diabetes` - Diabetic patient manager
- Manages global state:
  - `results`: Interaction check results
  - `alternatives`: Alternative suggestions
  - `mlPrediction`: ML prediction data
  - `isLoading`, `mlLoading`: Loading states
  - `activeTab`: 'text' or 'camera' mode
- Renders:
  - Navbar
  - Hero section
  - Tab switcher (text input vs camera)
  - InteractionChecker or CameraCapture component
  - ResultsDisplay
  - MLPrediction
  - AlternativesDisplay
  - Footer
- Toast notifications for user feedback

#### `frontend/src/index.css`
**Purpose**: Global CSS styles  
**What it contains**:
- Tailwind CSS imports
- Custom color scheme (medical theme)
- Animations (gradient backgrounds, spinners)
- Grid background pattern

---

### 🧩 **Components** (`frontend/src/components/`)

#### `frontend/src/components/Navbar.jsx`
**Purpose**: Top navigation bar  
**What it does**:
- Logo and app name
- Navigation links:
  - Home (Interaction Checker)
  - ML Dashboard
  - Diabetes Manager
- Responsive mobile menu

#### `frontend/src/components/Hero.jsx`
**Purpose**: Hero section on homepage  
**What it does**:
- Welcome message
- App description
- Call-to-action buttons

#### `frontend/src/components/InteractionChecker.jsx`
**Purpose**: Main drug interaction checker form  
**What it does**:
- Two drug input fields with autocomplete
- "Check Interaction" button
- Calls API: `POST /interactions/check`
- Also calls ML prediction endpoint
- Updates parent state with results

#### `frontend/src/components/CameraCapture.jsx`
**Purpose**: Camera-based drug name extraction  
**What it does**:
- Uses React Webcam to access user's camera
- Captures photo
- Uses Tesseract.js (client-side) or sends to backend OCR
- Extracts drug names from image
- Auto-fills drug inputs

#### `frontend/src/components/ResultsDisplay.jsx`
**Purpose**: Displays interaction check results  
**What it does**:
- Shows drug names
- Displays interaction status (Safe/Unsafe)
- Shows severity badge (Minor/Moderate/Major/Contraindicated)
- Displays interaction details (effect, mechanism, management)
- Shows safety message and recommendations
- Color-coded based on severity

#### `frontend/src/components/AlternativesDisplay.jsx`
**Purpose**: Shows safe alternative drugs  
**What it does**:
- Displays when interaction is detected
- Lists alternatives for drug1 and drug2
- Shows similarity scores
- Explains why each alternative is safer
- Allows selecting alternatives

#### `frontend/src/components/MLPrediction.jsx`
**Purpose**: Displays ML model predictions  
**What it does**:
- Shows interaction probability (0-100%)
- Displays individual model predictions (RF, XGBoost, LightGBM)
- Shows ensemble prediction
- Displays confidence level
- Visual probability bar

#### `frontend/src/components/ModelDashboard.jsx`
**Purpose**: ML model performance dashboard  
**What it does**:
- Shows model metrics (accuracy, precision, recall, F1, AUC-ROC)
- Displays confusion matrix
- Shows feature importance
- Compares model performance
- Displays optimization results

#### `frontend/src/components/DiabetesManager.jsx`
**Purpose**: Diabetic patient management interface  
**What it does**:
- **Patient Management**:
  - Create new patient profile
  - List all patients
  - Select active patient
  - Edit patient info (age, labs, complications)
- **Medication Management**:
  - Add medications to patient's list
  - Remove medications
  - View current medications
- **Drug Risk Checking**:
  - Check single drug against patient profile
  - "Check All Meds" button - checks all patient's medications
  - Shows risk level, severity, factors
  - Displays ML predictions (supplementary)
- **Drug Browser**:
  - "Browse All Drugs" button opens modal
  - Lists all 8,263 drugs from database
  - Pagination and search
  - Click to select drug
- **Reports**:
  - Generate comprehensive DDI report
  - Shows all interactions found
  - Lists safe/unsafe medications

#### `frontend/src/components/FloatingElements.jsx`
**Purpose**: Animated background elements  
**What it does**:
- Creates floating particles/shapes
- Adds visual interest
- Animated with Framer Motion

#### `frontend/src/components/Footer.jsx`
**Purpose**: Footer section  
**What it does**:
- Copyright info
- Links to documentation
- Social links (if any)

---

### 🔌 **Services** (`frontend/src/services/`)

#### `frontend/src/services/api.js`
**Purpose**: API client - all backend communication  
**What it does**:
- `apiRequest()`: Base function for all API calls
- **Drug functions**:
  - `searchDrugs()`: Search drugs by name
  - `getDrugById()`, `getDrugByName()`: Get drug details
- **Interaction functions**:
  - `checkInteraction()`: Check drug interaction
  - `getDrugInteractions()`: Get all interactions for a drug
- **Alternative functions**:
  - `getAlternatives()`: Get safe alternatives
- **OCR functions**:
  - `extractFromImage()`: Extract text from image
- **ML functions**:
  - `getMLPrediction()`: Get ML prediction
  - `getMLModelInfo()`: Get model info
  - `getMLComparison()`: Get optimization comparison
- **History functions**:
  - `getHistory()`: Get comparison history
  - `getHistoryStats()`: Get statistics
- **Utility functions**:
  - `getStats()`: Database statistics
  - `healthCheck()`: Health check
- Uses `VITE_API_URL` environment variable (defaults to `http://localhost:8000`)

---

### 🛠️ **Utils** (`frontend/src/utils/`)

#### `frontend/src/utils/debounce.js`
**Purpose**: Debounce utility function  
**What it does**:
- Delays function execution
- Used for search input to avoid excessive API calls
- Example: Only search after user stops typing for 300ms

---

### 🎨 **Public Assets** (`frontend/public/`)

#### `frontend/public/pill.svg`
**Purpose**: Pill icon/image  
**What it is**: SVG icon used in UI

---

### 📦 **Configuration Files**

#### `frontend/tailwind.config.js`
**Purpose**: Tailwind CSS configuration  
**What it does**:
- Defines custom color scheme:
  - `medical`: Primary brand color (teal/green)
  - `slate`: Neutral grays
- Custom animations
- Extends default Tailwind theme

#### `frontend/postcss.config.js`
**Purpose**: PostCSS configuration  
**What it does**:
- Processes CSS with Tailwind and Autoprefixer
- Required for Tailwind CSS

---

## 🔄 **How Everything Works Together**

### **User Flow: Checking Drug Interaction**

1. **User opens app** → `frontend/src/App.jsx` renders
2. **User types drug names** → `InteractionChecker.jsx` component
3. **User clicks "Check"** → `api.js` calls `POST /interactions/check`
4. **Backend receives request** → `backend/app/main.py` endpoint
5. **Backend calls service** → `interaction_service.py` checks database
6. **ML prediction** → `ml/predictor.py` loads models and predicts
7. **Rules check** → `diabetic/rules.py` evaluates clinical rules
8. **Hybrid decision** → Rules PRIMARY, ML SUPPLEMENTARY
9. **Response sent** → JSON with interaction details
10. **Frontend displays** → `ResultsDisplay.jsx` shows results
11. **Log saved** → `comparison_logger.py` saves to database

### **User Flow: Diabetic Patient Management**

1. **User navigates to `/diabetes`** → `DiabetesManager.jsx` renders
2. **User creates patient** → API call to `POST /diabetes/patients`
3. **User adds medications** → API call to `POST /diabetes/patients/{id}/medications`
4. **User checks drug** → API call to `POST /diabetes/patients/{id}/check-drug`
5. **Backend evaluates** → `diabetic/service.py` calls `diabetic/rules.py`
6. **Rules engine assesses** → Checks eGFR, complications, drug class patterns
7. **ML supplements** → `diabetic/ml_predictor.py` provides probability
8. **Response** → Risk level, severity, factors, recommendations
9. **Frontend displays** → Color-coded risk badges, detailed explanations

### **Data Flow: ML Training**

1. **Load TWOSIDES data** → `scripts/load_twosides.py`
2. **Map severity** → `scripts/map_twosides_severity.py`
3. **Build training set** → `scripts/build_training_set.py`
4. **Train models** → `scripts/train_twosides_ml.py`
5. **Evaluate** → `scripts/evaluate_models.py`
6. **Find threshold** → `scripts/find_optimal_threshold.py`
7. **Save models** → `models/` directory
8. **Load in production** → `ml/predictor.py` loads on startup

---

## 🎯 **Key Design Decisions**

1. **Rules Primary, ML Supplementary**: Clinical safety rules always override ML predictions
2. **Hybrid System**: Combines rule-based logic with ML for best of both worlds
3. **Class Imbalance Handling**: Uses class weights and optimal threshold tuning
4. **Streaming Data Processing**: Training scripts write directly to disk (no memory accumulation)
5. **Cursor-Based Pagination**: Efficient database queries without OFFSET
6. **Async Everything**: FastAPI async endpoints for scalability
7. **Comprehensive Logging**: All checks logged for audit and model improvement
8. **Modular Architecture**: Separate modules (diabetic, ML, services) for maintainability

---

## 📚 **Summary**

This is a **hybrid rule-based and machine learning system** for drug interaction prediction, specifically optimized for **diabetic patients**. The backend uses **FastAPI** with **SQLAlchemy** for database operations, **scikit-learn/XGBoost/LightGBM** for ML, and a comprehensive **clinical rule engine**. The frontend is a **React** SPA with **Tailwind CSS** styling, providing an intuitive interface for drug checking and patient management.

The system prioritizes **patient safety** by making rules primary and ML supplementary, ensuring that clinical guidelines always take precedence over potentially flawed ML predictions.

