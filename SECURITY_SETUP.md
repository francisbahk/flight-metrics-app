# 🔒 Security Setup Guide - Protecting Your API Keys

## ⚠️ What Happened?

Your Amadeus API keys were **exposed in your public GitHub repository** and have been **revoked by Amadeus** for security reasons.

**Files that contained exposed keys:**
- `STREAMLIT_DEPLOY.md` - Contained example keys (now fixed)
- `.env` - Contains placeholder keys (this file is NOT tracked by git)

## ✅ How We Fixed It

### 1. Removed Exposed Keys from Documentation
- Updated `STREAMLIT_DEPLOY.md` to show placeholders instead of real keys
- Added clear security warnings

### 2. Protected Sensitive Files
Your `.gitignore` already protects these files (they will NEVER be committed):
```
.env                          # Local environment variables
.streamlit/secrets.toml       # Local Streamlit secrets
```

### 3. Verified Git Tracking
Confirmed that `.env` and `secrets.toml` are **NOT** tracked by git:
```bash
✅ .env is ignored by git
✅ .streamlit/secrets.toml is ignored by git
```

---

## 🔑 Getting New API Keys

### Step 1: Generate New Credentials

1. Go to https://developers.amadeus.com/
2. Log in with your account
3. Navigate to **"My Self-Service Workspace"**
4. Click on your app (or create a new one)
5. Click **"Create API key"** or **"Regenerate key"**
6. Copy both:
   - **API Key** (Client ID)
   - **API Secret** (Client Secret)

---

## 🛠️ Setting Up Your Environment

### For Local Development

**Step 1:** Open the `.env` file in the root of your project:
```bash
nano .env
```

**Step 2:** Replace the placeholders with your NEW credentials:
```bash
# Amadeus API Credentials
AMADEUS_API_KEY=your_actual_api_key_here
AMADEUS_API_SECRET=your_actual_api_secret_here

# MySQL Database Configuration (optional)
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=flights
```

**Step 3:** Save and close the file

**Step 4:** Verify the app can read the credentials:
```bash
streamlit run streamlit_app.py
```

The app will automatically load credentials from `.env` using `python-dotenv`.

---

### For Streamlit Cloud Deployment

**Step 1:** Go to your app on Streamlit Cloud:
- https://share.streamlit.io/

**Step 2:** Click on your app → **"Settings"** (⚙️ icon)

**Step 3:** Click **"Secrets"** in the left sidebar

**Step 4:** Add your credentials in TOML format:
```toml
AMADEUS_API_KEY = "your_actual_api_key_here"
AMADEUS_API_SECRET = "your_actual_api_secret_here"
```

**Step 5:** Click **"Save"**

Streamlit will automatically restart your app with the new secrets.

---

## 📝 How Environment Variables Work

### In Your Code

The app reads credentials using this pattern:

```python
import os
from dotenv import load_dotenv

# Load .env file (local development)
load_dotenv()

# Try environment variable first, then Streamlit secrets
api_key = os.getenv("AMADEUS_API_KEY") or st.secrets.get("AMADEUS_API_KEY", "")
api_secret = os.getenv("AMADEUS_API_SECRET") or st.secrets.get("AMADEUS_API_SECRET", "")
```

### Priority Order:
1. **Local Development**: Reads from `.env` file
2. **Streamlit Cloud**: Reads from Streamlit Secrets
3. **System Environment**: Reads from OS environment variables

---

## 🚫 What NOT to Do

### ❌ NEVER do these:

1. **Don't commit `.env` to git**
   ```bash
   # BAD - This will expose your keys!
   git add .env
   ```

2. **Don't hardcode keys in your code**
   ```python
   # BAD - Don't do this!
   AMADEUS_API_KEY = "your_actual_key_hardcoded"
   ```

3. **Don't put keys in markdown/documentation files**
   ```markdown
   # BAD - Don't show real keys in docs!
   AMADEUS_API_KEY = "real_key_here"
   ```

4. **Don't share `.env` file publicly**
   - Don't upload to GitHub
   - Don't paste in chat/email
   - Don't share in screenshots

---

## ✅ What TO Do

### ✔️ ALWAYS do these:

1. **Use environment variables**
   ```python
   # GOOD - Read from environment
   api_key = os.getenv("AMADEUS_API_KEY")
   ```

2. **Keep `.env` in `.gitignore`**
   ```bash
   # .gitignore
   .env
   .streamlit/secrets.toml
   ```

3. **Use `.env.example` as a template**
   ```bash
   # .env.example - Safe to commit (no real keys)
   AMADEUS_API_KEY=your_api_key_here
   AMADEUS_API_SECRET=your_api_secret_here
   ```

4. **Verify before committing**
   ```bash
   # Check what will be committed
   git status
   git diff

   # Make sure .env is NOT listed!
   ```

---

## 🔍 Checking Your Setup

### Verify Git Ignores Sensitive Files

```bash
# Check ignored files
git status --ignored

# Should show:
# .env
# .streamlit/secrets.toml
```

### Verify Files Are Not Tracked

```bash
# This should return nothing
git ls-files | grep -E "\.env$|secrets\.toml$"

# If it shows files, they ARE tracked (bad!)
# If it shows nothing: ✅ Good!
```

### Test Environment Variables

```bash
# Run app and check console
streamlit run streamlit_app.py

# Should NOT show:
# "Amadeus API credentials not found"
```

---

## 📚 Additional Resources

- [Amadeus API Documentation](https://developers.amadeus.com/docs)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Python-dotenv Documentation](https://pypi.org/project/python-dotenv/)
- [Git Ignore Documentation](https://git-scm.com/docs/gitignore)

---

## 🆘 Troubleshooting

### "Amadeus API credentials not found"

**Cause:** `.env` file doesn't have valid keys

**Fix:**
1. Open `.env`
2. Add your NEW API keys (get from developers.amadeus.com)
3. Save and restart the app

### "API Key Revoked" Error

**Cause:** Using old keys that were exposed

**Fix:**
1. Generate NEW keys from Amadeus developer portal
2. Update `.env` with new keys
3. Update Streamlit Cloud secrets if deployed

### ".env file not working"

**Cause:** File might not be in the correct location

**Fix:**
1. Make sure `.env` is in the **root directory** (same level as `streamlit_app.py`)
2. Check file permissions: `ls -la .env`
3. Verify format (no quotes around values unless needed):
   ```bash
   AMADEUS_API_KEY=abc123xyz
   ```

---

## ✨ You're All Set!

Once you add your new API keys to `.env`, you can:

1. **Run locally:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Push to GitHub** (safely - `.env` is ignored):
   ```bash
   git add .
   git commit -m "Updated documentation"
   git push origin main
   ```

3. **Deploy to Streamlit Cloud:**
   - Add keys to Streamlit Secrets (Settings → Secrets)
   - App will auto-deploy from GitHub

**Your keys are now safe!** 🔒
