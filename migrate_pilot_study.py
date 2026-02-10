"""
Migration script for pilot study features.

This script:
1. Creates the SessionProgress table for page refresh recovery
2. Adds new columns to CrossValidation table (rerank_sequence, source_token)
3. Creates pilot tokens in the access_tokens table

Run this script before starting the pilot study:
    python migrate_pilot_study.py
"""

import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, '.')

from backend.db import Base, engine, SessionLocal, AccessToken, init_db
from pilot_tokens import PILOT_TOKENS
from sqlalchemy import text


def run_migration():
    """Run the pilot study database migration."""
    print("=" * 60)
    print("Pilot Study Migration")
    print("=" * 60)

    # Step 1: Create/update tables (including SessionProgress)
    print("\n1. Creating/updating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("   Tables created successfully")
    except Exception as e:
        print(f"   Error creating tables: {e}")
        return False

    # Step 2: Add new columns to cross_validations if they don't exist
    print("\n2. Checking cross_validations table columns...")
    db = SessionLocal()
    try:
        # Check if columns exist (SQLite-compatible approach)
        result = db.execute(text("PRAGMA table_info(cross_validations)"))
        columns = [row[1] for row in result.fetchall()]

        if 'rerank_sequence' not in columns:
            print("   Adding rerank_sequence column...")
            db.execute(text("ALTER TABLE cross_validations ADD COLUMN rerank_sequence INTEGER"))
            db.commit()
            print("   rerank_sequence column added")
        else:
            print("   rerank_sequence column already exists")

        if 'source_token' not in columns:
            print("   Adding source_token column...")
            db.execute(text("ALTER TABLE cross_validations ADD COLUMN source_token VARCHAR(255)"))
            db.commit()
            print("   source_token column added")
        else:
            print("   source_token column already exists")

    except Exception as e:
        print(f"   Note: Column check/add failed (may be MySQL): {e}")
        print("   Columns will be created by SQLAlchemy if they don't exist")
        db.rollback()

    finally:
        db.close()

    # Step 3: Create pilot tokens
    print("\n3. Creating pilot tokens...")
    db = SessionLocal()
    created_count = 0
    existing_count = 0

    try:
        for token in PILOT_TOKENS.keys():
            existing = db.query(AccessToken).filter(AccessToken.token == token).first()
            if not existing:
                new_token = AccessToken(
                    token=token,
                    is_used=0,
                    created_at=datetime.utcnow()
                )
                db.add(new_token)
                created_count += 1
                print(f"   Created token: {token}")
            else:
                existing_count += 1

        db.commit()
        print(f"   Created {created_count} new tokens, {existing_count} already existed")

    except Exception as e:
        db.rollback()
        print(f"   Error creating tokens: {e}")
        return False

    finally:
        db.close()

    # Step 4: Verify
    print("\n4. Verification...")
    db = SessionLocal()
    try:
        # Count pilot tokens
        pilot_count = db.query(AccessToken).filter(
            AccessToken.token.in_(list(PILOT_TOKENS.keys()))
        ).count()
        print(f"   Pilot tokens in database: {pilot_count}/15")

        # Check SessionProgress table exists
        from backend.db import SessionProgress
        result = db.query(SessionProgress).limit(1).all()
        print(f"   SessionProgress table: OK")

    except Exception as e:
        print(f"   Verification error: {e}")

    finally:
        db.close()

    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)

    return True


def list_pilot_tokens():
    """List all pilot tokens and their status."""
    db = SessionLocal()
    try:
        print("\nPilot Token Status:")
        print("-" * 40)
        for token in sorted(PILOT_TOKENS.keys()):
            record = db.query(AccessToken).filter(AccessToken.token == token).first()
            if record:
                status = "USED" if record.is_used else "AVAILABLE"
                print(f"  {token}: {status}")
            else:
                print(f"  {token}: NOT IN DB")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_pilot_tokens()
    else:
        run_migration()
        list_pilot_tokens()
