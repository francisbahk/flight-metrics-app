# 🚂 Deploy Backend to Railway - Quick Guide

## Why Railway?

Your Vercel frontend can't reach `localhost:8000` - it needs a deployed backend API. Railway makes this super easy!

## Step 1: Create Railway Account

1. Go to [railway.app](https://railway.app)
2. Click "Login" → "Login with GitHub"
3. Authorize Railway to access your GitHub

## Step 2: Deploy Your Backend

1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose: `francisbahk/flight-metrics-app`
4. Railway will auto-detect the `railway.json` configuration!

## Step 3: Add MySQL Database

1. In your Railway project, click "New" → "Database" → "Add MySQL"
2. Railway will create a MySQL database and auto-add connection variables

## Step 4: Set Environment Variables

Click on your backend service → "Variables" tab → Add these:

```bash
# Amadeus API (same as your local .env)
AMADEUS_API_KEY=your_api_key_here
AMADEUS_API_SECRET=your_api_secret_here

# MySQL (Railway auto-provides these, but verify they exist)
MYSQL_HOST=${{MYSQLHOST}}
MYSQL_PORT=${{MYSQLPORT}}
MYSQL_USER=${{MYSQLUSER}}
MYSQL_PASSWORD=${{MYSQLPASSWORD}}
MYSQL_DB=${{MYSQLDATABASE}}

# CORS - Add your Vercel domain
CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
```

**IMPORTANT**: Replace `your-app.vercel.app` with your actual Vercel URL!

## Step 5: Initialize Database Schema

Railway provides a web-based MySQL client:

1. Click on your MySQL database
2. Go to "Data" tab
3. Click "Query" button
4. Copy and paste the contents of `database/schema.sql`
5. Click "Run"

OR use Railway CLI:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Run schema
railway run mysql -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DB < database/schema.sql
```

## Step 6: Get Your Backend URL

1. Click on your backend service
2. Go to "Settings" tab
3. Scroll to "Networking" → Click "Generate Domain"
4. Copy the URL (e.g., `https://your-backend.up.railway.app`)

## Step 7: Update Vercel Frontend

Go to your Vercel project:

1. Settings → Environment Variables
2. Add new variable:
   - **Name**: `REACT_APP_API_URL`
   - **Value**: `https://your-backend.up.railway.app` (your Railway URL)
   - Check all environments (Production, Preview, Development)
3. Click "Save"
4. Go to Deployments → Click "..." on latest → "Redeploy"

## Step 8: Update Backend CORS

You need to allow your Vercel domain in the backend:

1. Go to Railway → Your backend service → Variables
2. Update `CORS_ORIGINS` to include your Vercel URL:
   ```
   https://your-app.vercel.app,http://localhost:3000
   ```
3. Railway will auto-redeploy

## Step 9: Test It! 🎉

Visit your Vercel app: `https://your-app.vercel.app`

Try searching for flights:
- Origin: JFK
- Destination: LAX
- Date: 2025-11-15

It should now return results from the Amadeus API! 🚀

## Troubleshooting

### "Network Error" in browser console

- Check that `REACT_APP_API_URL` is set in Vercel
- Verify CORS_ORIGINS includes your Vercel domain
- Check Railway logs for errors

### "Database connection failed"

- Verify MySQL service is running in Railway
- Check that database schema was loaded
- Verify environment variables are connected

### "Amadeus API error"

- Check that `AMADEUS_API_KEY` and `AMADEUS_API_SECRET` are set in Railway
- Verify they're the same credentials that work locally

## Check Railway Logs

```bash
# Install CLI
npm install -g @railway/cli

# Login and link
railway login
railway link

# View logs
railway logs
```

Or view logs in Railway dashboard:
1. Click on your service
2. "Deployments" tab
3. Click on latest deployment
4. View build and runtime logs

## Cost

Railway offers:
- **Free tier**: $5 usage credit per month (enough for development)
- **Pro plan**: $20/month (for production)

Your app should easily fit in the free tier for testing!

---

**Questions?** Check Railway docs: https://docs.railway.app
