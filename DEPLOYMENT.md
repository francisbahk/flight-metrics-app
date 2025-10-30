# 🚀 Deployment Guide

Your app has **two parts**: Frontend (React) and Backend (FastAPI). Here are the best deployment options:

## 🎯 Recommended: Split Deployment

### Frontend → Vercel (Free & Easy)
### Backend → Railway/Render (Free tier available)

---

## 📱 Option 1: Frontend on Vercel (RECOMMENDED)

Vercel is perfect for React apps!

### Steps:

1. **Go to Vercel**: https://vercel.com/new

2. **Import your repo**:
   - Click "Import Git Repository"
   - Select `francisbahk/flight-metrics-app`

3. **Configure Build Settings**:
   - Framework Preset: `Create React App`
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `build`

4. **Add Environment Variable**:
   - Key: `REACT_APP_API_URL`
   - Value: `https://your-backend-url.com/api` (we'll get this in step 2)

5. **Deploy!**

Your frontend will be live at: `https://flight-metrics-app.vercel.app`

---

## 🖥️ Option 2: Backend Deployment

Choose ONE of these for your FastAPI backend:

### A. Railway (Easiest for Python)

1. **Go to Railway**: https://railway.app/

2. **New Project** → **Deploy from GitHub repo**

3. **Select**: `francisbahk/flight-metrics-app`

4. **Settings**:
   - Root Directory: `backend`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

5. **Add Environment Variables**:
   ```
   AMADEUS_API_KEY=your_key
   AMADEUS_API_SECRET=your_secret
   MYSQL_USER=your_user
   MYSQL_PASSWORD=your_password
   MYSQL_HOST=your_mysql_host
   MYSQL_PORT=3306
   MYSQL_DB=flights
   ```

6. **Add MySQL Database** (Railway has a plugin):
   - Click "New" → MySQL
   - It will auto-populate the env vars

7. **Deploy!**

Your backend URL: `https://flight-metrics-app-production.up.railway.app`

### B. Render (Alternative)

1. **Go to Render**: https://render.com/

2. **New** → **Web Service**

3. **Connect**: `francisbahk/flight-metrics-app`

4. **Settings**:
   - Name: `flight-metrics-backend`
   - Root Directory: `backend`
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

5. **Add Environment Variables** (same as Railway)

6. **Add MySQL Database**:
   - Create a new PostgreSQL database on Render (free tier)
   - OR use external MySQL (like PlanetScale)

---

## 🗄️ Database Options

Your app needs MySQL. Choose one:

### A. Railway MySQL Plugin (Easiest)
- Built-in with Railway
- Automatically configured
- Free tier: 1GB storage

### B. PlanetScale (Free MySQL)
1. Go to https://planetscale.com/
2. Create free database
3. Get connection string
4. Add to backend env vars

### C. Supabase (PostgreSQL alternative)
1. Go to https://supabase.com/
2. Create project
3. Modify code to use PostgreSQL instead of MySQL

---

## 🔗 Connecting Frontend to Backend

After deploying backend, update frontend:

1. **On Vercel** (frontend settings):
   - Add Environment Variable:
   - `REACT_APP_API_URL` = `https://your-backend-url.railway.app/api`

2. **Update frontend API client** (optional - if using env var):

   Edit `frontend/src/api/flights.js`:
   ```javascript
   const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';
   ```

3. **Redeploy frontend** on Vercel

---

## 🆓 Free Tier Limits

### Vercel (Frontend)
- ✅ Unlimited deployments
- ✅ Automatic HTTPS
- ✅ 100GB bandwidth/month

### Railway (Backend + DB)
- ✅ $5 free credits/month
- ✅ ~500 hours runtime
- ✅ 1GB MySQL storage

### Render (Backend)
- ✅ 750 hours/month free
- ⚠️ Sleeps after 15 min inactivity
- ✅ Auto-wake on request

---

## 🎯 Simplest Path (Start Here!)

### Step 1: Deploy Frontend to Vercel
1. Import repo on Vercel
2. Set root directory: `frontend`
3. Deploy!

### Step 2: Deploy Backend to Railway
1. Create Railway project
2. Add MySQL database plugin
3. Deploy from GitHub
4. Copy backend URL

### Step 3: Connect Them
1. Add `REACT_APP_API_URL` to Vercel with Railway backend URL
2. Redeploy frontend

**Done!** Both parts live and talking to each other! 🎉

---

## 📝 Quick Reference

**Frontend URL**: `https://your-app.vercel.app`
**Backend URL**: `https://your-backend.up.railway.app`
**API Docs**: `https://your-backend.up.railway.app/docs`

---

## 🐛 Common Issues

### CORS Errors
Add your Vercel URL to backend CORS settings in `backend/main.py`:

```python
allow_origins=[
    "http://localhost:3000",
    "https://your-app.vercel.app",  # Add this
]
```

### Database Connection
Make sure all env vars are set correctly in Railway/Render dashboard.

### Build Fails on Vercel
Make sure `vercel.json` is committed to git with correct settings.

---

Need help? Check the logs:
- **Vercel**: Project → Deployments → View Logs
- **Railway**: Project → Deployments → View Logs

Good luck! 🚀
