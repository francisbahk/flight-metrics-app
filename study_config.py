"""
Study configuration — manual parameters you change between phases.

Phase 2 prep checklist:
  1. Fill in SEED_PROLIFIC_IDS below
  2. Set MAX_RERANKS_PER_SEED
  3. Run:  python seed_phase2.py
  4. Flip RERANK_ENABLED = True
  5. Restart the server
"""

# ── Re-ranking toggle ──────────────────────────────────────────────────────────
# Set to True before Phase 2. False = standard flow, no re-rank task shown.
RERANK_ENABLED = False

# ── Re-ranking parameters ──────────────────────────────────────────────────────
# How many Phase 2 participants will re-rank each seed prompt.
MAX_RERANKS_PER_SEED = 5

# ── Phase 1 participants to seed for re-ranking ────────────────────────────────
# After Phase 1 completes, paste the Prolific IDs of participants whose
# prompt + flights you want used as re-ranking tasks.
# Then run:  python seed_phase2.py
SEED_PROLIFIC_IDS = [
    # "abc123xyz",
    # "def456uvw",
]
