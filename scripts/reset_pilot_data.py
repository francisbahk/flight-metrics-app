"""
Reset all pilot study data for a fresh slate.

What this deletes for all 20 pilot tokens (GA01-GD05):
  - searches + all cascades (flights_shown, user_rankings, flight_csvs, lilo_sessions)
  - cross_validations where reviewer_token is a pilot token
  - session_progress for pilot tokens
  - completion_tokens generated for pilot sessions
  - Resets access_tokens (is_used=0, used_at=NULL, completion_token=NULL)

What this does NOT touch:
  - DEMO / DATA seed data used by cross-validation queues
  - Non-pilot data

Usage:
    python reset_pilot_data.py          # dry-run (shows counts, no changes)
    python reset_pilot_data.py --confirm  # actually deletes
"""
import sys
from backend.db import (
    SessionLocal, Search, AccessToken, CompletionToken,
    CrossValidation, SessionProgress,
)
from pilot_tokens import PILOT_TOKENS

TOKENS = list(PILOT_TOKENS.keys())  # ['GA01', ..., 'GD05']
DRY_RUN = "--confirm" not in sys.argv


def main():
    db = SessionLocal()
    try:
        # ------------------------------------------------------------------ #
        # Count what will be affected
        # ------------------------------------------------------------------ #
        searches = db.query(Search).filter(Search.completion_token.in_(TOKENS)).all()
        search_ids = [s.search_id for s in searches]

        cross_vals = db.query(CrossValidation).filter(
            CrossValidation.reviewer_token.in_(TOKENS)
        ).count()

        session_rows = db.query(SessionProgress).filter(
            SessionProgress.access_token.in_(TOKENS)
        ).count()

        # Completion tokens linked through access_token records
        access_records = db.query(AccessToken).filter(AccessToken.token.in_(TOKENS)).all()
        comp_token_vals = [r.completion_token for r in access_records if r.completion_token]
        comp_tokens = db.query(CompletionToken).filter(
            CompletionToken.token.in_(comp_token_vals)
        ).count() if comp_token_vals else 0

        used_count = sum(1 for r in access_records if r.is_used)

        print("=" * 55)
        print("  Pilot Data Reset")
        print("=" * 55)
        print(f"  Access tokens found     : {len(access_records)} / {len(TOKENS)}")
        print(f"  Tokens currently used   : {used_count}")
        print(f"  Searches to delete      : {len(searches)}")
        print(f"  Cross-validations       : {cross_vals}")
        print(f"  Session progress rows   : {session_rows}")
        print(f"  Completion tokens       : {comp_tokens}")
        print()

        if DRY_RUN:
            print("  DRY RUN — no changes made.")
            print("  Run with --confirm to apply.")
            print()
            return

        print("  Applying reset...")

        # ------------------------------------------------------------------ #
        # 1. Delete searches (cascades to flights_shown, user_rankings,
        #    flight_csvs, lilo_sessions, cross_validations.reviewed_search_id)
        # ------------------------------------------------------------------ #
        for s in searches:
            db.delete(s)
        db.flush()

        # ------------------------------------------------------------------ #
        # 2. Delete cross_validations where reviewer is a pilot token
        #    (not already caught by cascade above)
        # ------------------------------------------------------------------ #
        db.query(CrossValidation).filter(
            CrossValidation.reviewer_token.in_(TOKENS)
        ).delete(synchronize_session=False)

        # ------------------------------------------------------------------ #
        # 3. Delete session_progress for pilot tokens
        # ------------------------------------------------------------------ #
        db.query(SessionProgress).filter(
            SessionProgress.access_token.in_(TOKENS)
        ).delete(synchronize_session=False)

        # ------------------------------------------------------------------ #
        # 4. Delete completion_tokens linked to pilot sessions
        # ------------------------------------------------------------------ #
        if comp_token_vals:
            db.query(CompletionToken).filter(
                CompletionToken.token.in_(comp_token_vals)
            ).delete(synchronize_session=False)

        # ------------------------------------------------------------------ #
        # 5. Reset access_tokens: mark unused, clear completion token
        # ------------------------------------------------------------------ #
        for record in access_records:
            record.is_used = 0
            record.used_at = None
            record.completion_token = None

        db.commit()
        print("  Done. All pilot data cleared and tokens reset.")
        print()

    except Exception as e:
        db.rollback()
        print(f"  ERROR: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
