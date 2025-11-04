# 🎈 Deploy to Streamlit Cloud - Quick Guide

## ✅ What We Just Built

You now have a **complete Streamlit app** ([streamlit_app.py](streamlit_app.py)) that combines:
- ✈️ Flight search via Amadeus API
- 🤖 LISTEN-U and LISTEN-T algorithms
- 📊 Manual flight ranking
- 📈 Results comparison

**Local app running at**: http://localhost:8501

## 🚀 Deploy to Streamlit Cloud (Free!)

### Step 1: Rename Requirements File

Streamlit Cloud looks for `requirements.txt` (not `requirements_streamlit.txt`):

```bash
mv requirements.txt requirements_fastapi.txt
mv requirements_streamlit.txt requirements.txt
```

Or just use the existing `requirements_streamlit.txt` by copying it:

```bash
cp requirements_streamlit.txt requirements.txt
```

### Step 2: Add .gitignore for Secrets

```bash
echo ".streamlit/secrets.toml" >> .gitignore
```

This ensures your API keys don't get committed to GitHub!

### Step 3: Commit to GitHub

```bash
git add .
git commit -m "Add Streamlit app version

- Created streamlit_app.py with all features
- Flight search via Amadeus API
- LISTEN-U and LISTEN-T algorithms integrated
- Manual ranking interface
- Results comparison dashboard
"
git push origin main
```

### Step 4: Deploy on Streamlit Cloud

1. **Go to**: https://share.streamlit.io/

2. **Sign in** with GitHub

3. **Click "New app"**

4. **Configure**:
   - **Repository**: `francisbahk/flight-metrics-app`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`

5. **Click "Advanced settings"** → **Secrets**

   Add your environment variables:

   ```toml
   AMADEUS_API_KEY = "your_api_key_here"
   AMADEUS_API_SECRET = "your_api_secret_here"
   ```

   **Get your credentials from**: https://developers.amadeus.com/

   **IMPORTANT**: Never commit these keys to GitHub! They should only be added in:
   - Local: `.env` file (already in .gitignore)
   - Streamlit Cloud: Settings → Secrets (shown above)

   **Note**: You can skip the MySQL variables for now - the app works without a database!

6. **Click "Deploy"**

Streamlit Cloud will:
- Install dependencies from `requirements.txt`
- Start your app
- Give you a URL like: `https://francisbahk-flight-metrics-app-streamlit-app-xyz123.streamlit.app`

## 🎯 Testing Your Deployed App

Once deployed, test these features:

1. **Flight Search**:
   - Origin: JFK
   - Destination: LAX
   - Date: 2025-11-15
   - Click "Search Flights"

2. **LISTEN-U**:
   - Select 5-10 flights
   - Enter preference: "I want cheap flights with few stops"
   - Run LISTEN-U

3. **LISTEN-T**:
   - Select flights
   - Run tournament
   - See the winner!

## 🔧 Optional: Add MySQL Database

If you want to persist flight data and rankings:

### Option 1: Use PlanetScale (Free MySQL)

1. Go to https://planetscale.com
2. Create free database
3. Get connection string
4. Add to Streamlit secrets:

```toml
MYSQL_HOST = "your-db.us-east-1.psdb.cloud"
MYSQL_PORT = "3306"
MYSQL_USER = "your_username"
MYSQL_PASSWORD = "your_password"
MYSQL_DB = "flights"
```

### Option 2: Keep It Simple (No Database)

The app works perfectly without MySQL! It stores flight results in session state during your session.

## 📝 Update Your Streamlit App

To make changes:

1. Edit `streamlit_app.py` locally
2. Test: `streamlit run streamlit_app.py`
3. Commit and push:
   ```bash
   git add streamlit_app.py
   git commit -m "Update streamlit app"
   git push
   ```
4. Streamlit Cloud auto-redeploys! 🚀

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'X'"

Add the missing package to `requirements.txt`:

```txt
streamlit==1.28.0
sqlalchemy==2.0.25
pymysql==1.1.0
cryptography==41.0.7
requests==2.31.0
python-dotenv==1.0.0
pydantic==2.5.3
pandas==2.0.3
pyarrow==14.0.1
```

### "Amadeus API Error"

Check that secrets are set correctly in Streamlit Cloud dashboard.

### App Crashes on Startup

Check the logs in Streamlit Cloud dashboard → Click on your app → "Manage app" → "Logs"

## 🎨 Customizing Your App

Edit [streamlit_app.py](streamlit_app.py):

- Change colors: Modify `.streamlit/config.toml`
- Add features: Add new tabs in the code
- Update UI: Streamlit has tons of components!

Check out: https://docs.streamlit.io/

## 📊 What's Different from FastAPI + React?

| Feature | FastAPI + React | Streamlit |
|---------|----------------|-----------|
| **Frontend** | React (separate) | Built-in UI |
| **Backend** | FastAPI | Python functions |
| **Deployment** | Vercel + Railway | Streamlit Cloud (one place!) |
| **Database** | Required | Optional |
| **Complexity** | High | Low |
| **Learning Curve** | Steep | Gentle |

## ✨ Benefits of Streamlit

✅ **Single file** - Everything in `streamlit_app.py`
✅ **No separate backend** - Python functions handle everything
✅ **Free deployment** - Streamlit Cloud is free for public apps
✅ **Auto-reload** - Changes deploy automatically
✅ **Built-in UI** - No CSS/HTML needed
✅ **Python all the way** - No JavaScript required

---

**Your Streamlit app is ready! Open http://localhost:8501 to test it now!** 🎉

After testing locally, follow the steps above to deploy to Streamlit Cloud.
