# Git Setup Script for Diabetic DDI Project
# This script will help you initialize git and push to GitHub

Write-Host "=== Diabetic DDI - Git Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
$gitInstalled = Get-Command git -ErrorAction SilentlyContinue

if (-not $gitInstalled) {
    Write-Host "Git is not installed or not in PATH." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please install Git for Windows from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Or use winget: winget install --id Git.Git -e --source winget" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After installing Git, restart PowerShell and run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Host "Git found: $($gitInstalled.Version)" -ForegroundColor Green
Write-Host ""

# Check if already a git repository
if (Test-Path .git) {
    Write-Host "Git repository already initialized." -ForegroundColor Yellow
    $continue = Read-Host "Do you want to continue with adding files? (y/n)"
    if ($continue -ne "y") {
        exit 0
    }
} else {
    Write-Host "Initializing git repository..." -ForegroundColor Cyan
    git init
    git branch -M main
}

Write-Host ""
Write-Host "Adding essential files..." -ForegroundColor Cyan

# Add important files
git add README.md
git add .gitignore
git add backend/app/
git add backend/scripts/
git add backend/requirements.txt
git add frontend/src/
git add frontend/package.json
git add frontend/vite.config.js
git add frontend/tailwind.config.js
git add frontend/index.html
git add *.md
git add *.bat

Write-Host ""
Write-Host "Files staged. Reviewing status..." -ForegroundColor Cyan
git status

Write-Host ""
Write-Host "Ready to commit!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Review the files above" -ForegroundColor White
Write-Host "2. Run: git commit -m 'Initial commit: Diabetic DDI Prediction System'" -ForegroundColor White
Write-Host "3. Run: git remote add origin https://github.com/Dhritimanmitraa/diabetic-ddi.git" -ForegroundColor White
Write-Host "4. Run: git push -u origin main" -ForegroundColor White
Write-Host ""

$commit = Read-Host "Do you want to commit now? (y/n)"
if ($commit -eq "y") {
    git commit -m "Initial commit: Diabetic DDI Prediction System"
    Write-Host ""
    Write-Host "Commit successful!" -ForegroundColor Green
    
    $remote = Read-Host "Do you want to add remote and push? (y/n)"
    if ($remote -eq "y") {
        git remote add origin https://github.com/Dhritimanmitraa/diabetic-ddi.git 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Remote added successfully!" -ForegroundColor Green
        } else {
            Write-Host "Remote might already exist, continuing..." -ForegroundColor Yellow
        }
        
        Write-Host ""
        Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
        git push -u origin main
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
            Write-Host "View your repository at: https://github.com/Dhritimanmitraa/diabetic-ddi" -ForegroundColor Cyan
        } else {
            Write-Host ""
            Write-Host "Push failed. You may need to authenticate." -ForegroundColor Yellow
            Write-Host "Consider using GitHub CLI (gh auth login) or setting up SSH keys." -ForegroundColor Yellow
        }
    }
}

