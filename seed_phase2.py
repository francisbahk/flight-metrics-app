"""
Seed Phase 2 cross-validation data.

Reads SEED_PROLIFIC_IDS from study_config.py, queries the DB for each
participant's completed prompt + flights, and loads them into seed_prompts.

Run once before Phase 2:
    python seed_phase2.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from study_config import SEED_PROLIFIC_IDS, MAX_RERANKS_PER_SEED
from backend.db import SessionLocal, Participant, Ranking, load_seed_prompt


def run():
    if not SEED_PROLIFIC_IDS:
        print("No SEED_PROLIFIC_IDS defined in study_config.py — nothing to seed.")
        return

    db = SessionLocal()
    try:
        seeded = 0
        skipped = 0
        missing = []

        for slot_number, prolific_id in enumerate(SEED_PROLIFIC_IDS, start=1):
            # Get prompt from Participant table
            participant = db.query(Participant).filter_by(prolific_id=prolific_id).first()
            if not participant or not participant.prompt:
                print(f"  [!] No prompt found for {prolific_id} — skipping.")
                missing.append(prolific_id)
                continue

            # Use all flights the participant saw (not just their top 20 ranked ones)
            if participant.all_flights_json:
                flights = json.loads(participant.all_flights_json)
            else:
                # Fallback to ranked flights if all_flights not stored
                ranking_rows = (
                    db.query(Ranking)
                    .filter(Ranking.prolific_id == prolific_id)
                    .order_by(Ranking.rank)
                    .all()
                )
                if not ranking_rows:
                    print(f"  [!] No flights found for {prolific_id} — skipping.")
                    missing.append(prolific_id)
                    continue
                flights = [json.loads(r.flight_json) for r in ranking_rows]
                print(f"  [!] No all_flights_json for {prolific_id} — using ranked flights only ({len(flights)})")

            flights_json = json.dumps(flights)

            loaded = load_seed_prompt(
                slot_number=slot_number,
                prolific_id=prolific_id,
                prompt_text=participant.prompt,
                flights_json=flights_json,
                overwrite=True,
            )

            if loaded:
                print(f"  [✓] Seeded slot {slot_number}: {prolific_id} ({len(flights)} flights)")
                seeded += 1
            else:
                print(f"  [!] Failed to seed: {prolific_id}")
                skipped += 1

    finally:
        db.close()

    print()
    print(f"Done. Seeded: {seeded} | Already existed: {skipped} | Missing: {len(missing)}")
    print(f"Each seed prompt will be re-ranked by up to {MAX_RERANKS_PER_SEED} participants.")
    if missing:
        print(f"Missing IDs (no data in DB): {missing}")


if __name__ == "__main__":
    run()
