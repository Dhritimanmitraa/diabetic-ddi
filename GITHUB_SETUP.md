# GitHub Repository Setup Guide

This guide will help you upload DrugGuard to your GitHub repository.

## Prerequisites

1. **Install Git** (if not already installed)
   - Download from: https://git-scm.com/download/win
   - Or use: `winget install Git.Git`

2. **GitHub Account**
   - Make sure you're logged in at https://github.com
   - Repository should be created at: https://github.com/Dhritimanmitraa/diabetic-ddi

## Quick Setup (Windows PowerShell)

1. **Open PowerShell in the project directory**

2. **Run the setup script:**
```powershell
.\setup_github.ps1
```

This will:
- Initialize git repository
- Add all files
- Create initial commit
- Push to GitHub

## Manual Setup

If the script doesn't work, follow these steps:

### Step 1: Initialize Git
```powershell
git init
```

### Step 2: Add Files
```powershell
git add .
```

### Step 3: Create Initial Commit
```powershell
git commit -m "Initial commit: DrugGuard - Diabetic Drug Interaction Checker

- Hybrid rule-based and ML system for diabetic patient safety
- FastAPI backend with React frontend
- Ensemble ML models (RF, XGBoost, LightGBM)
- OCR integration for medication label scanning
- Patient-specific risk assessment with eGFR considerations"
```

### Step 4: Set Main Branch
```powershell
git branch -M main
```

### Step 5: Add Remote
```powershell
git remote add origin https://github.com/Dhritimanmitraa/diabetic-ddi.git
```

### Step 6: Push to GitHub
```powershell
git push -u origin main
```

## Authentication

If you get authentication errors:

1. **Use Personal Access Token (Recommended)**
   - Go to: https://github.com/settings/tokens
   - Generate new token (classic)
   - Select scopes: `repo`
   - Copy the token
   - When prompted for password, paste the token

2. **Or use GitHub CLI**
```powershell
winget install GitHub.cli
gh auth login
```

## Files Included

The repository includes:
- ✅ All source code (backend/app/, frontend/src/)
- ✅ Configuration files (requirements.txt, package.json)
- ✅ Documentation (README.md, LICENSE, etc.)
- ✅ Training scripts
- ✅ Example data files

## Files Excluded (via .gitignore)

- ❌ Virtual environments (venv/)
- ❌ Node modules (node_modules/)
- ❌ Database files (*.db, *.sqlite)
- ❌ Large ML model files (*.pkl)
- ❌ Large datasets (TWOSIDES, OFFSIDES CSV files)
- ❌ Log files
- ❌ IDE configuration files

## Troubleshooting

### "Git is not recognized"
- Install Git from https://git-scm.com/download/win
- Restart PowerShell after installation

### "Authentication failed"
- Use Personal Access Token instead of password
- Or set up SSH keys

### "Repository not found"
- Make sure the repository exists at: https://github.com/Dhritimanmitraa/diabetic-ddi
- Check that you have write access

### "Large file warning"
- Some files might be too large for GitHub (>100MB)
- Check .gitignore is working correctly
- Use Git LFS for large files if needed

## After Upload

Once uploaded, your repository will be available at:
**https://github.com/Dhritimanmitraa/diabetic-ddi**

You can:
- View the code online
- Share with others
- Set up GitHub Actions for CI/CD
- Add collaborators
- Create releases

## Next Steps

1. Add a repository description on GitHub
2. Add topics/tags: `diabetes`, `drug-interactions`, `machine-learning`, `healthcare`
3. Consider adding a GitHub Actions workflow for testing
4. Add a `.github/ISSUE_TEMPLATE.md` for bug reports

