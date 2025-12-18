"""
Database migration script to add cross-validation columns.
"""
from backend.db import SessionLocal, engine
from sqlalchemy import text

def migrate_database():
    """Add new columns for cross-validation feature."""
    db = SessionLocal()

    try:
        print("🔧 Starting database migration...")

        # Check if columns already exist
        result = db.execute(text("""
            SELECT COUNT(*) as count
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = 'searches'
            AND COLUMN_NAME = 'amadeus_flights_json'
        """))

        column_exists = result.fetchone()[0] > 0

        if column_exists:
            print("✓ Columns already exist - migration not needed")
            return

        print("📝 Adding new columns to searches table...")

        # Add amadeus_flights_json column
        db.execute(text("""
            ALTER TABLE searches
            ADD COLUMN amadeus_flights_json JSON NULL
        """))
        print("  ✓ Added amadeus_flights_json column")

        # Add listen_ranked_flights_json column
        db.execute(text("""
            ALTER TABLE searches
            ADD COLUMN listen_ranked_flights_json JSON NULL
        """))
        print("  ✓ Added listen_ranked_flights_json column")

        db.commit()

        print("\n📝 Creating cross_validations table...")

        # Create cross_validations table
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS cross_validations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                reviewer_session_id VARCHAR(255) NOT NULL,
                reviewer_token VARCHAR(255) NULL,
                reviewed_session_id VARCHAR(255) NOT NULL,
                reviewed_search_id INT NULL,
                reviewed_prompt TEXT NOT NULL,
                reviewed_flights_json JSON NOT NULL,
                selected_flight_ids JSON NOT NULL,
                selected_flights_data JSON NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_reviewer_session (reviewer_session_id),
                INDEX idx_reviewer_token (reviewer_token),
                INDEX idx_reviewed_session (reviewed_session_id),
                FOREIGN KEY (reviewed_search_id)
                    REFERENCES searches(search_id)
                    ON DELETE CASCADE,
                INDEX idx_created_at (created_at)
            )
        """))
        print("  ✓ Created cross_validations table")

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
