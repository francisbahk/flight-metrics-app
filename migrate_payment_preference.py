"""
Database migration script to add payment_preference column to survey_responses table.
"""
from backend.db import SessionLocal, engine
from sqlalchemy import text

def migrate_database():
    """Add payment_preference column to survey_responses table."""
    db = SessionLocal()

    try:
        print("🔧 Starting database migration...")

        # Check if column already exists
        result = db.execute(text("""
            SELECT COUNT(*) as count
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = 'survey_responses'
            AND COLUMN_NAME = 'payment_preference'
        """))

        column_exists = result.fetchone()[0] > 0

        if column_exists:
            print("✓ Column already exists - migration not needed")
            return

        print("📝 Adding payment_preference column to survey_responses table...")

        # Add payment_preference column
        db.execute(text("""
            ALTER TABLE survey_responses
            ADD COLUMN payment_preference VARCHAR(50) NULL
        """))
        print("  ✓ Added payment_preference column")

        db.commit()

        print("\n✅ Migration completed successfully!")

    except Exception as e:
        db.rollback()
        print(f"\n❌ Migration failed: {str(e)}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    migrate_database()
