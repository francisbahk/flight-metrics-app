# 🚨 IMPORTANT: Streamlit Cloud Deployment Configuration

## Main File Path

When deploying to Streamlit Cloud, use this configuration:

```
Main file path: streamlit_app.py
```

**DO NOT use**: `backend/main.py` (that's the old FastAPI backend)

## Quick Deploy Steps

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. New app
4. Repository: `francisbahk/flight-metrics-app`
5. Branch: `main`
6. **Main file path**: `streamlit_app.py` ⬅️ **CRITICAL!**
7. Advanced settings → Secrets:
   ```toml
   AMADEUS_API_KEY = "your_key"
   AMADEUS_API_SECRET = "your_secret"
   ```
8. Deploy!

## Files Overview

- ✅ **`streamlit_app.py`** - Main Streamlit app (USE THIS)
- ❌ `backend/main.py` - Old FastAPI backend (DON'T USE)
- ✅ `requirements.txt` - Streamlit dependencies
- ✅ `.python-version` - Python 3.11
- ✅ `.streamlit/config.toml` - UI configuration
