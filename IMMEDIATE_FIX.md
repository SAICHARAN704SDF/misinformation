# ⚡ IMMEDIATE FIX - Python 3.13 Issue

## The Problem
Render is using **Python 3.13** (released Oct 2024) which breaks package installation.

## The Solution (3 files changed)

### ✅ Fixed Files:
1. **`runtime.txt`** → Now: `python-3.11.6`
2. **`render.yaml`** → PYTHON_VERSION: `3.11.6`
3. **`requirements.txt`** → Changed `--find-links` to `--extra-index-url`

## 🚀 Deploy NOW

```powershell
# Step 1: Commit
git add runtime.txt render.yaml requirements.txt PYTHON_VERSION_FIX.md
git commit -m "Fix: Force Python 3.11.6 (avoid Python 3.13 compatibility issues)"
git push origin main
```

**Step 2:** Go to Render Dashboard
- Click your service → **Settings**
- Scroll to **Build & Deploy**
- Click **"Clear Build Cache & Deploy"** ← IMPORTANT!

**Step 3:** Wait 3-5 minutes → Success! ✅

## Why This Works

- `runtime.txt` forces Render to use Python 3.11.6
- Clearing cache makes Render read the new file
- Python 3.11.6 is stable and all packages work

## Expected Build Log

✅ **Good:**
```
Using Python version: 3.11.6
Installing requirements from requirements_rf_only.txt
Successfully installed flask-3.0.0 scikit-learn-1.3.0...
Build succeeded!
```

❌ **Bad (before fix):**
```
Using Python version: 3.13.0
Cannot import 'setuptools.build_meta'
Build failed
```

## If It Still Fails

**Nuclear Option:** Delete service and recreate:
1. Save your environment variables
2. Delete Render service
3. Create new service from GitHub
4. Will read `runtime.txt` correctly from start

---

**TL;DR:** 
1. Commit the 3 changed files
2. Clear Render build cache
3. Redeploy
4. Should work now! 🎉
