# Railway Deployment - Next Steps

## ✅ What I Just Fixed

The Railway deployment was failing because the database connection was created at **import time**, causing the app to crash if MySQL wasn't configured.

### Changes Made:

1. **Lazy Database Initialization** ([backend/database.py](backend/database.py))
   - Database engine and sessions are now created only when first accessed
   - App can start without MySQL being available
   - No more crashes at startup!

2. **Better Logging & Diagnostics** ([backend/main.py](backend/main.py))
   - Added root `/` endpoint that works without database
   - Added detailed logging with `flush=True` for Railway logs
   - Shows environment variables at startup for debugging
   - Health check now reports database status without crashing

3. **Resilient Health Checks**
   - `/` → Returns basic status (no database required)
   - `/api/health` → Returns status with database connection check
   - Server stays up even if database is down

## 🚀 What Happens Now

Railway will automatically redeploy your app with the new changes. This time:

1. ✅ Build will succeed (already did before)
2. ✅ App will start successfully
3. ✅ Health checks should pass!
4. ⚠️ Database will show "disconnected" (need to add MySQL service)

## 📋 Next Steps

### Step 1: Check Railway Deployment

Go to Railway dashboard and watch the deployment logs. You should see:

```
============================================================
Starting Flight Metrics API Server
============================================================
Python version: 3.x.x
FastAPI version: 1.0.0
PORT env var: 8000 (or whatever Railway assigns)
MYSQL_HOST: localhost
⚠ Server will start anyway (database not required for health check)
✓ Server startup complete
============================================================
```

Then health checks should **pass** this time!

### Step 2: Get Your Backend URL

Once deployed successfully:
1. Go to your Railway project → Backend service
2. Click "Settings" → "Networking"
3. Click "Generate Domain"
4. Copy the URL (e.g., `https://flight-metrics-production-xxxx.up.railway.app`)

### Step 3: Test the Backend

Visit these URLs in your browser:

- `https://your-backend.up.railway.app/`
  - Should return: `{"service":"Flight Metrics API","status":"running",...}`

- `https://your-backend.up.railway.app/api/health`
  - Should return: `{"status":"healthy","database":"disconnected",...}`

- `https://your-backend.up.railway.app/docs`
  - Should show FastAPI interactive docs

✅ If these work, your backend is deployed successfully!

### Step 4: Add MySQL Database (Optional for Now)

You have two options:

#### Option A: Add MySQL to Railway (Recommended)
1. In Railway project, click "New" → "Database" → "Add MySQL"
2. Railway will auto-configure environment variables
3. Go to MySQL service → "Data" tab
4. Click "Query" → Paste contents of `database/schema.sql` → Run
5. Redeploy backend (or it will auto-redeploy)
6. Health check should now show `"database":"connected"`

#### Option B: Test Without Database First
You can test the Vercel frontend connection first without MySQL:
- Flight search will fail (needs database to store results)
- But you can verify the connection is working
- Add MySQL later when ready

### Step 5: Connect Vercel Frontend

Once backend is deployed and accessible:

1. Go to Vercel → Your project → Settings → Environment Variables

2. Add this variable:
   ```
   Name: REACT_APP_API_URL
   Value: https://your-backend.up.railway.app
   ```
   (Replace with your actual Railway URL)

3. Make sure to add it to all environments:
   - ✅ Production
   - ✅ Preview
   - ✅ Development

4. Go to Deployments → Click "..." on latest → "Redeploy"

### Step 6: Update Backend CORS

Your backend needs to allow requests from Vercel:

1. In Railway → Backend service → Variables
2. Add or update:
   ```
   CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
   ```
   (Replace `your-app.vercel.app` with your actual Vercel domain)

3. Railway will auto-redeploy

### Step 7: Add Amadeus API Credentials

For flight search to work, add these to Railway → Backend service → Variables:

```
AMADEUS_API_KEY=your_key_here
AMADEUS_API_SECRET=your_secret_here
```

(Use the same values from your local `.env` file)

### Step 8: Test End-to-End! 🎉

1. Visit your Vercel app: `https://your-app.vercel.app`
2. Try searching for flights:
   - Origin: **JFK**
   - Destination: **LAX**
   - Date: **2025-11-15**
3. Should return flight results from Amadeus API!

## 🐛 Troubleshooting

### Backend still fails health checks

Check Railway logs for the startup message. If you don't see it, there's an import error.

### "Network Error" in browser

- Check that `REACT_APP_API_URL` is set correctly in Vercel
- Verify the Railway backend URL is accessible
- Check CORS_ORIGINS includes your Vercel domain

### Database connection fails

- Make sure MySQL service is added to Railway
- Verify environment variables are connected
- Check that schema.sql was loaded

### Flight search returns no results

- Check that Amadeus API credentials are set in Railway
- Verify database is connected
- Check Railway backend logs for API errors

## 📚 Helpful Railway CLI Commands

Install Railway CLI:
```bash
npm install -g @railway/cli
```

View logs:
```bash
railway login
railway link
railway logs
```

Run commands in Railway environment:
```bash
railway run mysql -u root -p
```

## 🎯 Expected Final Architecture

```
User Browser
    ↓
Vercel (React Frontend)
    ↓ (REACT_APP_API_URL)
Railway Backend (FastAPI)
    ↓
Railway MySQL Database
    ↓
Amadeus Flight API
```

---

**The fixes are pushed to GitHub. Railway should be redeploying now!**

Check your Railway dashboard to see the new deployment. The health checks should pass this time! 🚀
