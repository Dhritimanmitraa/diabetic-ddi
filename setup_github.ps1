# PowerShell script to initialize and push to GitHub
# Run this after installing Git

Write-Host "Setting up GitHub repository..." -ForegroundColor Cyan

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "Git is not installed. Please install Git first:" -ForegroundColor Red
    Write-Host "   Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Initialize git repository
Write-Host ""
Write-Host "Initializing git repository..." -ForegroundColor Cyan
git init

# Add all files
Write-Host "Adding files to staging..." -ForegroundColor Cyan
git add .

# Create initial commit
Write-Host "Creating initial commit..." -ForegroundColor Cyan
git commit -m "Initial commit: DrugGuard - Diabetic Drug Interaction Checker

- Hybrid rule-based and ML system for diabetic patient safety
- FastAPI backend with React frontend
- Ensemble ML models (RF, XGBoost, LightGBM)
- OCR integration for medication label scanning
- Patient-specific risk assessment with eGFR considerations"

# Set main branch
Write-Host "Setting main branch..." -ForegroundColor Cyan
git branch -M main

# Add remote repository
Write-Host "Adding remote repository..." -ForegroundColor Cyan
$remoteCheck = git remote get-url origin 2>&1
if ($LASTEXITCODE -ne 0) {
    git remote add origin https://github.com/Dhritimanmitraa/diabetic-ddi.git
} else {
    Write-Host "Remote already configured" -ForegroundColor Green
}

# Push to GitHub
Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
Write-Host "   (You may need to authenticate)" -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "   Repository: https://github.com/Dhritimanmitraa/diabetic-ddi" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Push failed. You may need to:" -ForegroundColor Yellow
    Write-Host "   1. Set up GitHub authentication (Personal Access Token)" -ForegroundColor Yellow
    Write-Host "   2. Or use: git push -u origin main" -ForegroundColor Yellow
}
