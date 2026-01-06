#!/bin/bash
# Bash script to initialize and push to GitHub
# Run this after installing Git
# Last updated: 2026-01-06

echo "Setting up GitHub repository..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed. Please install Git first:"
    echo "   macOS: brew install git"
    echo "   Linux: sudo apt-get install git"
    exit 1
fi

echo "Git found: $(git --version)"

# Initialize git repository
echo ""
echo "Initializing git repository..."
git init

# Add all files
echo "Adding files to staging..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: DrugGuard - Diabetic Drug Interaction Checker

- Hybrid rule-based and ML system for diabetic patient safety
- FastAPI backend with React frontend
- Ensemble ML models (RF, XGBoost, LightGBM)
- OCR integration for medication label scanning
- Patient-specific risk assessment with eGFR considerations"

# Set main branch
echo "Setting main branch..."
git branch -M main

# Add remote repository
echo "Adding remote repository..."
git remote add origin https://github.com/Dhritimanmitraa/diabetic-ddi.git 2>/dev/null || \
git remote set-url origin https://github.com/Dhritimanmitraa/diabetic-ddi.git

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo "   (You may need to authenticate)"
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "Successfully pushed to GitHub!"
    echo "   Repository: https://github.com/Dhritimanmitraa/diabetic-ddi"
else
    echo ""
    echo "WARNING: Push failed. You may need to:"
    echo "   1. Set up GitHub authentication (Personal Access Token)"
    echo "   2. Or run: git push -u origin main"
fi

