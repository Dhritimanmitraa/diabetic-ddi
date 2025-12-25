# 🚀 Quick Upload Instructions

Your project is ready to upload! Here's what you need to do:

## Step 1: Install Git (if not installed)

**Windows:**
- Download from: https://git-scm.com/download/win
- Or run: `winget install Git.Git`

## Step 2: Run the Setup Script

Open PowerShell in this directory and run:

```powershell
.\setup_github.ps1
```

That's it! The script will:
1. ✅ Initialize git repository
2. ✅ Add all important files
3. ✅ Create professional commit message
4. ✅ Push to your GitHub repo

## What's Included?

✅ **All source code** (backend & frontend)
✅ **Configuration files** (requirements.txt, package.json)
✅ **Documentation** (README.md, LICENSE, guides)
✅ **Training scripts** (ML model training)
✅ **Example data** (small CSV files for training)

## What's Excluded?

❌ Virtual environments (venv/)
❌ Node modules (node_modules/)
❌ Database files (*.db)
❌ Large ML models (*.pkl files)
❌ Large datasets (TWOSIDES, OFFSIDES CSV.gz files)
❌ Log files

## Authentication

When you push, GitHub will ask for credentials:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your password)
  - Create one at: https://github.com/settings/tokens
  - Select scope: `repo`
  - Copy and paste when prompted

## Manual Alternative

If the script doesn't work, see `GITHUB_SETUP.md` for manual steps.

---

**Your repository will be live at:**
https://github.com/Dhritimanmitraa/diabetic-ddi

