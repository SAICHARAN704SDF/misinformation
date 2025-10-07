# 🚀 Deployment Guide - GitHub + Render

## Overview
This guide will help you deploy your Misinformation Detection System to **Render** via **GitHub**.

---

## ✅ Prerequisites

1. **Git** installed on your system
2. **GitHub account** - [Sign up here](https://github.com/signup)
3. **Render account** - [Sign up here](https://render.com/signup)

---

## 📦 Step 1: Prepare Your Project

### 1.1 Check Required Files (Already Created ✓)
- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - Tells Render how to run the app
- ✅ `render.yaml` - Render configuration
- ✅ `.gitignore` - Files to exclude from git
- ✅ `README.md` - Project documentation

### 1.2 Update Procfile for Production
Your Procfile should use `gunicorn`:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers=2 --threads=4
```

### 1.3 Add gunicorn to requirements.txt
Make sure `gunicorn` is in your `requirements.txt`:
```
gunicorn==21.2.0
```

---

## 🐙 Step 2: Push to GitHub

### 2.1 Initialize Git (if not already done)
```powershell
git init
```

### 2.2 Add All Files
```powershell
git add .
```

### 2.3 Commit Your Changes
```powershell
git commit -m "Initial commit - Misinformation Detection System"
```

### 2.4 Create GitHub Repository
1. Go to [GitHub](https://github.com)
2. Click the **+** icon → **New repository**
3. Name it: `misinformation-detector`
4. Leave it **Public** (or Private if preferred)
5. **DO NOT** initialize with README (you already have one)
6. Click **Create repository**

### 2.5 Link Local Repository to GitHub
```powershell
git remote add origin https://github.com/YOUR_USERNAME/misinformation-detector.git
git branch -M main
git push -u origin main
```

**Replace** `YOUR_USERNAME` with your actual GitHub username.

---

## 🌐 Step 3: Deploy to Render

### Option A: Deploy via render.yaml (Recommended)

1. **Go to Render Dashboard**: https://dashboard.render.com/

2. **Click "New +"** → **Blueprint**

3. **Connect Your GitHub Repository**:
   - Click "Connect GitHub"
   - Authorize Render to access your repositories
   - Select your `misinformation-detector` repository

4. **Render will automatically detect** your `render.yaml` file

5. **Click "Apply"** - Render will:
   - Install dependencies from `requirements.txt`
   - Train a lightweight model (Random Forest)
   - Start the Flask app with gunicorn

### Option B: Manual Web Service Deployment

1. **Go to Render Dashboard**: https://dashboard.render.com/

2. **Click "New +"** → **Web Service**

3. **Connect Your GitHub Repository**

4. **Configure the Service**:
   - **Name**: `misinformation-detector`
   - **Environment**: `Python 3`
   - **Build Command**: 
     ```
     pip install -r requirements.txt && python train_models.py --data-path data --model rf
     ```
   - **Start Command**: 
     ```
     gunicorn app:app --bind 0.0.0.0:$PORT --workers=2 --threads=4
     ```

5. **Environment Variables** (Add these):
   ```
   PYTHON_VERSION=3.12.4
   USE_ONLY_RF=true
   FLASK_DEBUG=false
   SECRET_KEY=generate-random-secret-key-here
   ```

6. **Click "Create Web Service"**

---

## 🔧 Step 4: Configure Environment Variables on Render

### Required Variables:
| Variable | Value | Description |
|----------|-------|-------------|
| `PYTHON_VERSION` | `3.12.4` | Python version |
| `USE_ONLY_RF` | `true` | Use only Random Forest (lightweight) |
| `FLASK_DEBUG` | `false` | Disable debug mode in production |
| `SECRET_KEY` | `your-secret-key` | Generate a random secret key |

### Generate Secret Key:
```python
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 📊 Step 5: Monitor Deployment

### Check Build Logs
- Render will show real-time logs during deployment
- Wait for: `✓ Build complete`
- Then: `Starting service...`

### Access Your App
- Once deployed, Render provides a URL like:
  ```
  https://misinformation-detector-xxxx.onrender.com
  ```

### Test Your Deployment
1. Visit the URL in your browser
2. Try analyzing some text
3. Check `/models/manage` to see available models

---

## ⚠️ Important Notes

### Free Tier Limitations
- **Render Free Tier**:
  - 512 MB RAM
  - Service sleeps after 15 minutes of inactivity
  - First request after sleep takes ~30 seconds
  - 750 hours/month free

### Model Considerations
- **DO NOT** push model files (`.pkl`, `.pth`) to GitHub (too large)
- The `buildCommand` in `render.yaml` trains a lightweight model on deployment
- For best results on free tier: Use Random Forest only

### Large Models (LSTM, CNN, Ensemble)
- These models are **too large** for Render free tier
- Options:
  1. Train locally, upload to cloud storage (AWS S3, Google Drive)
  2. Download during build (add to buildCommand)
  3. Upgrade to paid Render plan ($7/month)

---

## 🔄 Updating Your Deployment

### Push Updates:
```powershell
git add .
git commit -m "Your update message"
git push origin main
```

Render will **automatically redeploy** when you push to GitHub (if auto-deploy is enabled).

---

## 🐛 Troubleshooting

### Issue: Build Fails
**Solution**: Check Render logs for errors. Common issues:
- Missing dependencies in `requirements.txt`
- Python version mismatch
- Out of memory (reduce model size)

### Issue: App Crashes
**Solution**: 
- Check Start Command is correct
- Ensure `app.py` has `app` variable (not just in `if __name__`)
- Check environment variables are set

### Issue: Models Not Loading
**Solution**: 
- Ensure models are trained during build
- Check `models/` directory exists
- Verify model files are created

### Issue: 502 Bad Gateway
**Solution**: 
- App took too long to start (>60s timeout)
- Reduce model training time in buildCommand
- Use lighter models

---

## 📚 Additional Resources

- **Render Documentation**: https://render.com/docs
- **Flask Deployment**: https://flask.palletsprojects.com/en/latest/deploying/
- **GitHub Guides**: https://guides.github.com/

---

## 🎉 Success Checklist

- [ ] Code pushed to GitHub
- [ ] Render connected to GitHub repo
- [ ] Service deployed successfully
- [ ] App accessible via Render URL
- [ ] Models loading correctly
- [ ] Predictions working
- [ ] Model management UI accessible

---

## 💡 Pro Tips

1. **Enable Auto-Deploy**: Render auto-deploys on git push
2. **Use Environment Variables**: Never hardcode secrets
3. **Monitor Logs**: Check Render dashboard for errors
4. **Health Checks**: Render pings `/health` to verify service
5. **Custom Domain**: Add your own domain in Render settings

---

## 🆘 Need Help?

- **Render Support**: https://render.com/docs/support
- **GitHub Issues**: Create issues in your repository
- **Community**: Stack Overflow, Reddit r/flask

---

**You're ready to deploy! 🚀**
