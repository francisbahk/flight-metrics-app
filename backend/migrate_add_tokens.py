"""
Migration script to add token table and token column to searches table.
"""
from sqlalchemy import create_engine, text
from db import DATABASE_URL, get_config

def migrate():
    """Add tokens table and token column to searches table."""
    engine = create_engine(DATABASE_URL, echo=True)

    with engine.connect() as conn:
        try:
            # Create tokens table
            print("Creating tokens table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tokens (
                    token VARCHAR(255) PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    used_at DATETIME NULL,
                    is_used INT DEFAULT 0
                )
            """))
            conn.commit()
            print("✓ Created tokens table")

            # Check if token column exists in searches table
            result = conn.execute(text("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'searches'
                AND COLUMN_NAME = 'token'
            """))
            existing_columns = {row[0] for row in result}

            # Add token column if it doesn't exist
            if 'token' not in existing_columns:
                print("Adding token column to searches table...")
                conn.execute(text("""
                    ALTER TABLE searches
                    ADD COLUMN token VARCHAR(255) NULL
                """))
                conn.commit()
                # Add index on token column
                conn.execute(text("""
                    CREATE INDEX idx_searches_token ON searches(token)
                """))
                conn.commit()
                print("✓ Added token column to searches table")
            else:
                print("✓ token column already exists in searches table")

            print("\n✓ Migration complete!")

        except Exception as e:
            print(f"\n✗ Migration failed: {str(e)}")
            raise

if __name__ == "__main__":
    print("Running migration to add token support...")
    migrate()
