"""
Retro-award CV overlap bonuses for reviewers already in the DB.

Usage:
    # Dry run — prints what would happen, no API call:
    python scripts/award_cv_bonuses.py

    # Actually create bonuses on Prolific:
    PROLIFIC_API_TOKEN=... PROLIFIC_BONUS_ENABLED=true \
        python scripts/award_cv_bonuses.py --live

Notes:
  - Idempotent: reviewers with an existing BonusAward row are skipped.
  - "Creating" a bonus does NOT pay it. Open the Prolific dashboard and
    hit "Pay bonuses", or add a follow-up POST to .../bulk-bonus-payments/<id>/pay/.
  - Reviewers with study_id=NULL are recorded as `pending` — supply a
    --study-id override if you want to force-award them.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db import (
    SessionLocal, CVRanking, SeedPrompt, Participant, init_db,
)
from backend.prolific_bonus import maybe_award_cv_overlap_bonus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Actually call the Prolific API. Without this, runs dry.")
    parser.add_argument("--study-id", default=None,
                        help="Override study_id for reviewers with none on file.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only process these reviewer prolific_ids.")
    args = parser.parse_args()

    if args.live:
        os.environ["PROLIFIC_BONUS_ENABLED"] = "true"
        if not os.environ.get("PROLIFIC_API_TOKEN"):
            print("ERROR: --live requires PROLIFIC_API_TOKEN in the environment.")
            sys.exit(1)
    else:
        os.environ["PROLIFIC_BONUS_ENABLED"] = "false"
        print("[DRY RUN] No API calls will be made. Re-run with --live to award.")

    init_db()
    db = SessionLocal()
    try:
        reviewers = [r for (r,) in db.query(CVRanking.reviewer_prolific_id).distinct().all()]
        if args.only:
            reviewers = [r for r in reviewers if r in args.only]

        for rev in reviewers:
            rows = (db.query(CVRanking)
                    .filter_by(reviewer_prolific_id=rev)
                    .order_by(CVRanking.rank).all())
            by_seed = {}
            for r in rows:
                by_seed.setdefault(r.seed_prompt_id, []).append(r)

            p = db.query(Participant).filter_by(prolific_id=rev).first()
            study_id = (p.study_id if p and p.study_id else None) or args.study_id

            for seed_id, seed_rows in by_seed.items():
                flights = [json.loads(r.flight_json) for r in seed_rows]
                result = maybe_award_cv_overlap_bonus(
                    reviewer_prolific_id=rev,
                    seed_prompt_id=seed_id,
                    reviewer_cv_flights=flights,
                    study_id=study_id,
                )
                print(f"{rev}  seed={seed_id}  ->  {result}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
