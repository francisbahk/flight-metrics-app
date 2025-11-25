# Uvicorn Hosting - Important Clarification

## TL;DR: You Cannot Use Uvicorn with Streamlit

**Uvicorn** is an ASGI server for FastAPI/Starlette applications.
**Streamlit** has its own built-in server and is incompatible with Uvicorn.

Your current app is built with Streamlit (`app.py`), which cannot run on Uvicorn.

---

## What is Uvicorn?

Uvicorn is a lightning-fast ASGI server implementation for Python web frameworks like:
- **FastAPI** - Modern API framework
- **Starlette** - Lightweight ASGI framework
- **Django** (with ASGI support)

It's designed for async Python web applications that follow the ASGI standard.

## What is Streamlit?

Streamlit is a **complete framework** with:
- Its own server (Tornado-based)
- WebSocket communication for real-time updates
- Session state management
- Built-in UI components

Streamlit runs with: `streamlit run app.py`

It is **NOT** an ASGI application and cannot run on Uvicorn.

---

## Your Current Deployment (Streamlit Cloud)

**Current Setup:**
- App: `app.py` (Streamlit)
- Hosting: Streamlit Cloud
- Server: Streamlit's built-in server
- Deployment: Auto-deploy from GitHub

**Pros:**
- Zero DevOps - fully managed
- Auto-scaling
- Free tier available
- Automatic SSL/HTTPS
- Built-in secrets management

**Cons:**
- Less control over server configuration
- Shared resources (can be slower)
- Limited to Streamlit apps only

---

## Alternative Hosting Options

### Option 1: Keep Streamlit Cloud (Recommended)

**Why:** It's working perfectly for your use case.

**When to choose this:**
- You want zero DevOps overhead
- Current performance is acceptable
- You don't need custom server configuration

**Cost:** Free tier, then $20-200/month for premium

---

### Option 2: Self-Host Streamlit (VPS + Nginx)

If you want more control but keep Streamlit, deploy on a VPS.

**Architecture:**
```
User → Nginx (reverse proxy) → Streamlit (port 8501) → Railway MySQL
```

**Steps:**
1. Get a VPS (DigitalOcean, AWS EC2, etc.)
2. Install Python 3.11
3. Clone your repo: `git clone [repo] && cd flight_app`
4. Install dependencies: `pip install -r requirements.txt`
5. Run Streamlit as a service: `streamlit run app.py --server.port 8501`
6. Configure Nginx reverse proxy:

```nginx
# /etc/nginx/sites-available/flight-app
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

7. Setup systemd service for auto-restart:

```ini
# /etc/systemd/system/streamlit-app.service
[Unit]
Description=Streamlit Flight App
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/flight_app
ExecStart=/usr/bin/streamlit run app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

**Pros:**
- Full server control
- Can customize Streamlit config
- Choose your own VPS specs

**Cons:**
- You manage DevOps (updates, security, scaling)
- Need to setup SSL (Let's Encrypt)
- More expensive if you want good performance

**Cost:** $5-50/month (VPS) + your time

---

### Option 3: Convert to FastAPI + React (Major Rewrite)

**ONLY if you want to use Uvicorn**, you'd need to completely rebuild as:
- Backend: FastAPI (runs on Uvicorn)
- Frontend: React/Vue/vanilla JS
- Deployment: Backend and frontend separately

**This is what your old README.md described** (FastAPI + React architecture).

**Architecture:**
```
User → Nginx → Uvicorn (FastAPI on :8000) → Railway MySQL
                 ↓
            Static React build
```

**FastAPI Backend Example:**
```python
# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

@app.post("/api/search")
async def search_flights(query: str):
    # Gemini parsing
    # Amadeus search
    # LISTEN ranking
    # Return JSON
    pass

# Serve React frontend
app.mount("/", StaticFiles(directory="frontend/build", html=True))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**React Frontend Example:**
```javascript
// Search.jsx
function FlightSearch() {
  const [results, setResults] = useState([]);

  const handleSearch = async (query) => {
    const res = await fetch('/api/search', {
      method: 'POST',
      body: JSON.stringify({ query })
    });
    setResults(await res.json());
  };

  // Build UI with React components
}
```

**Pros:**
- Full API control
- Can build mobile app later (uses same API)
- Separation of concerns (backend/frontend)
- Use Uvicorn + Gunicorn for production

**Cons:**
- **Complete rewrite required** (weeks of work)
- Need to rebuild all Streamlit UI in React
- Manage two codebases (backend + frontend)
- More complex deployment

**Cost:** Weeks of development time

---

## Recommendation

**Stick with Streamlit Cloud** (Option 1) unless:
1. You need custom server config → Use Option 2 (Self-hosted Streamlit)
2. You want an API for mobile apps → Use Option 3 (FastAPI rewrite)

**Why Streamlit Cloud is best for you:**
- Your app is a research tool, not a commercial product
- User traffic is low/moderate
- You want to focus on research, not DevOps
- Current setup is working perfectly

---

## If You Want Better Performance (Keep Streamlit)

Instead of changing hosting, optimize your Streamlit app:

### 1. Cache LISTEN Results
```python
@st.cache_data(ttl=3600)
def rank_flights_with_listen_main(flights, prompt, prefs):
    # LISTEN only runs once per unique input
    pass
```

### 2. Use Streamlit Spinner
```python
with st.spinner("Running LISTEN (2-3 min)..."):
    results = rank_flights_with_listen_main(...)
```

### 3. Reduce LISTEN Iterations (for testing)
```python
# In development
n_iterations = 5  # Quick test
# In production
n_iterations = 25  # Full learning
```

### 4. Async Flight Search
```python
import asyncio

async def search_all_airports(origins, destinations):
    tasks = [
        search_amadeus(o, d)
        for o in origins
        for d in destinations
    ]
    return await asyncio.gather(*tasks)
```

---

## Summary Table

| Option | Server | Effort | Cost | Best For |
|--------|--------|--------|------|----------|
| Streamlit Cloud | Built-in | None | Free-$20/mo | Current setup (recommended) |
| Self-hosted Streamlit | Built-in | Medium | $5-50/mo | Need server control |
| FastAPI + Uvicorn | Uvicorn | High (rewrite) | $5-50/mo | Want API/mobile app |

---

## Final Answer

**You don't need Uvicorn for this project.**

Your Streamlit app is working great on Streamlit Cloud. Uvicorn is for a completely different type of application (FastAPI).

If you have specific performance concerns with Streamlit Cloud, let me know what they are and we can optimize within Streamlit or discuss self-hosting Streamlit (still not Uvicorn).