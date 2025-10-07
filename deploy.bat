@echo off
echo ========================================
echo   GitHub + Render Deployment Script
echo ========================================
echo.

REM Check if git is initialized
git status >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/6] Initializing Git repository...
    git init
) else (
    echo [1/6] Git already initialized ✓
)

echo.
echo [2/6] Adding all files to Git...
git add .

echo.
echo [3/6] Creating commit...
set /p commit_msg="Enter commit message (default: Production ready deployment): "
if "%commit_msg%"=="" set commit_msg=Production ready deployment
git commit -m "%commit_msg%"

echo.
echo [4/6] GitHub Repository Setup
echo.
echo Please follow these steps:
echo 1. Go to https://github.com/new
echo 2. Repository name: misinformation-detector
echo 3. Make it Public or Private
echo 4. DO NOT initialize with README
echo 5. Click "Create repository"
echo.
set /p github_username="Enter your GitHub username: "
set /p continue="Press Enter when repository is created..."

echo.
echo [5/6] Connecting to GitHub...
git remote add origin https://github.com/%github_username%/misinformation-detector.git 2>nul
git remote set-url origin https://github.com/%github_username%/misinformation-detector.git
git branch -M main

echo.
echo [6/6] Pushing to GitHub...
git push -u origin main

echo.
echo ========================================
echo   ✓ SUCCESS! Code pushed to GitHub
echo ========================================
echo.
echo Next Steps:
echo 1. Go to https://dashboard.render.com/
echo 2. Click "New +" ^> "Web Service"
echo 3. Connect your GitHub repository
echo 4. Render will auto-detect render.yaml
echo 5. Add environment variables (see QUICK_DEPLOY.md)
echo 6. Click "Create Web Service"
echo.
echo Environment Variables to set in Render:
echo   PYTHON_VERSION = 3.12.4
echo   USE_ONLY_RF = true
echo   FLASK_DEBUG = false
echo   SECRET_KEY = [generate random]
echo.
echo Generate SECRET_KEY:
python -c "import secrets; print('SECRET_KEY =', secrets.token_hex(32))"
echo.
echo Full guide: See DEPLOYMENT_GUIDE.md
echo Quick guide: See QUICK_DEPLOY.md
echo.
pause
