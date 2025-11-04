# 🔑 Quick Guide: Adding Your New Amadeus API Keys

## Step 1: Get New API Keys from Amadeus

1. Go to https://developers.amadeus.com/
2. Log in to your account
3. Navigate to **"My Self-Service Workspace"**
4. Click on your app
5. Click **"Create API key"** or **"Regenerate key"**
6. Copy both:
   - **API Key** (Client ID)
   - **API Secret** (Client Secret)

---

## Step 2: Add Keys Locally (For Development)

### Open the `.env` file:
```bash
nano .env
```

### Replace the placeholders with your NEW keys:
```bash
AMADEUS_API_KEY=your_actual_new_key_here
AMADEUS_API_SECRET=your_actual_new_secret_here
```

### Save and close:
- Press `Ctrl + O` (save)
- Press `Enter` (confirm)
- Press `Ctrl + X` (exit)

---

## Step 3: Test Locally

```bash
streamlit run streamlit_app.py
```

The app should now work with your new credentials!

---

## Step 4: Update Streamlit Cloud (For Deployment)

1. Go to https://share.streamlit.io/
2. Click on your app
3. Click **Settings** (⚙️ icon)
4. Click **Secrets** in the left sidebar
5. Update the secrets:
   ```toml
   AMADEUS_API_KEY = "your_actual_new_key_here"
   AMADEUS_API_SECRET = "your_actual_new_secret_here"
   ```
6. Click **Save**

Streamlit Cloud will automatically restart with the new keys.

---

## ✅ Verification

### Local Development:
- Run `streamlit run streamlit_app.py`
- Try searching for flights (JFK → LAX)
- Should see flight results (not "API credentials not found")

### Streamlit Cloud:
- Visit your deployed app URL
- Try the same flight search
- Should work the same as locally

---

## 🔒 Security Reminder

- ✅ **DO**: Keep keys in `.env` (already in .gitignore)
- ✅ **DO**: Add keys to Streamlit Cloud Secrets
- ❌ **DON'T**: Commit `.env` to GitHub
- ❌ **DON'T**: Hardcode keys in your code
- ❌ **DON'T**: Share keys publicly

---

## 📚 More Information

See [SECURITY_SETUP.md](SECURITY_SETUP.md) for:
- Detailed security best practices
- Troubleshooting guide
- How environment variables work
- What files are protected by .gitignore

---

**You're ready to go!** Just paste your new keys and you'll be back up and running! 🚀
