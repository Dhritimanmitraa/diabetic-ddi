@echo off
echo ========================================
echo   DrugGuard Backend - Drug Interaction API
echo ========================================
cd backend

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check if database exists and has data
if not exist "drug_interactions.db" (
    echo.
    echo ========================================
    echo   DATABASE NOT FOUND - Fetching REAL data
    echo ========================================
    echo Fetching from OpenFDA and RxNorm APIs...
    echo This may take 5-15 minutes...
    echo.
    python -m scripts.fetch_real_data --drugs 2000 --interactions 50000
)

REM Start the server
echo.
echo Starting FastAPI server...
echo API Docs: http://localhost:8000/docs
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

