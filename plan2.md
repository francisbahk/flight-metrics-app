# Plan 2: Set Up LISTEN as Git Submodule

## Current Status

Right now, the flight app imports LISTEN from `../LISTEN/` (parent directory) using a simple path addition:

```python
# backend/listen_wrapper.py lines 13-16
LISTEN_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'LISTEN')
sys.path.insert(0, LISTEN_PATH)
from LISTEN import DuelingBanditOptimizer
```

This works locally but isn't portable for deployment or collaboration.

## Why Use Git Submodules?

Git submodules allow you to:
1. **Keep LISTEN as a separate repo** while including it in flight_app
2. **Track specific LISTEN versions** (commit hashes) for reproducibility
3. **Share the evaluation repo** without duplicating LISTEN code
4. **Update LISTEN independently** and pull changes when needed

## Implementation Plan

### Step 1: Add LISTEN as a Submodule

**Option A: If LISTEN is on GitHub/remote:**
```bash
cd /Users/francisbahk/flight_app
git submodule add https://github.com/<your-username>/LISTEN.git LISTEN
git commit -m "Add LISTEN as submodule"
```

**Option B: For local-only setup (simpler for now):**

Since LISTEN is a local repo, the easiest approach is to **copy LISTEN into flight_app** as a regular directory:

```bash
cd /Users/francisbahk/flight_app
cp -r /Users/francisbahk/LISTEN ./LISTEN
git add LISTEN
git commit -m "Add LISTEN directory"
```

**Or Option C: Use symbolic link** (works for local development):
```bash
cd /Users/francisbahk/flight_app
ln -s /Users/francisbahk/LISTEN ./LISTEN
```

This creates:
- `flight_app/LISTEN/` directory (linked to LISTEN repo)
- `.gitmodules` file (tracks submodule config) - only for Option A
- `flight_app/.git/modules/LISTEN/` (submodule git data) - only for Option A

### Step 2: Update Import Paths

Change `backend/listen_wrapper.py` to look in the local LISTEN submodule:

```python
# backend/listen_wrapper.py
LISTEN_PATH = os.path.join(os.path.dirname(__file__), '..', 'LISTEN')
sys.path.insert(0, LISTEN_PATH)
from LISTEN import DuelingBanditOptimizer
```

**Before**: `..` (parent dir) → `..` (parent of that) → `LISTEN`
**After**: `..` (parent dir) → `LISTEN` (submodule in flight_app root)

### Step 3: Update .gitignore (if needed)

Make sure `.gitignore` doesn't exclude `LISTEN/`:

```bash
# .gitignore should NOT have:
# LISTEN/
# /LISTEN
```

Submodules are tracked differently than regular directories.

### Step 4: Test the Setup

```bash
# Verify submodule is registered
git submodule status

# Should show:
# <commit-hash> LISTEN (heads/main)

# Test import
python3 -c "import sys; sys.path.insert(0, 'LISTEN'); from LISTEN import DuelingBanditOptimizer; print('✓ Import works')"

# Run the app
streamlit run app_new.py
```

### Step 5: Document for Collaborators

Add to `README.md`:

```markdown
## Setup Instructions

This repo uses LISTEN as a git submodule.

### First-time clone:
```bash
git clone <your-repo-url>
cd flight_app
git submodule init
git submodule update
```

### Updating LISTEN:
```bash
cd LISTEN
git pull origin main
cd ..
git add LISTEN
git commit -m "Update LISTEN submodule"
```
```

## Deployment Considerations

### Streamlit Cloud

Add to `.streamlit/config.toml`:

```toml
[server]
enableStaticServing = true
```

And ensure `requirements.txt` includes LISTEN dependencies:

```
# requirements.txt
numpy<2
scipy
scikit-learn
groq
google-generativeai>=0.3.0
```

Streamlit Cloud will automatically initialize submodules when deploying.

### Alternative: Direct Installation

If submodules cause issues in deployment, you can:

1. **Option A**: Install LISTEN from GitHub directly:
```bash
pip install git+https://github.com/<your-username>/LISTEN.git
```

2. **Option B**: Copy LISTEN code into `flight_app/LISTEN/` as a regular directory (not submodule) for simpler deployment

## Current Directory Structure

```
flight_app/
├── .git/
├── .gitmodules          # ← Created by submodule add
├── LISTEN/              # ← Submodule (will be created)
│   ├── LISTEN.py
│   ├── GP.py
│   ├── LLM.py
│   └── ...
├── backend/
│   ├── listen_wrapper.py  # ← Update import path
│   ├── amadeus_client.py
│   └── ...
├── app_new.py
├── requirements.txt
└── README.md
```

## Benefits of This Approach

1. **Version Control**: Pin specific LISTEN versions for reproducibility
2. **Separation of Concerns**: LISTEN development stays independent
3. **Easy Updates**: `git submodule update` pulls latest LISTEN changes
4. **No Duplication**: Don't copy-paste LISTEN code into flight_app
5. **Collaboration**: Others can clone flight_app and get LISTEN automatically

## Next Steps

1. Run the submodule commands above
2. Update import path in `listen_wrapper.py`
3. Test the app locally
4. Commit changes
5. Deploy to Streamlit Cloud (or other platform)

## Rollback Plan

If submodules cause issues:

```bash
# Remove submodule
git submodule deinit LISTEN
git rm LISTEN
rm -rf .git/modules/LISTEN

# Revert to parent directory import
# (keep current code in listen_wrapper.py)
```

Then use the parent directory import method (current approach) or copy LISTEN directly into flight_app.
