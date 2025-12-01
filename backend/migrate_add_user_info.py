"""
Migration script to add user_name and user_email columns to the searches table.
"""
from sqlalchemy import create_engine, text
from db import DATABASE_URL, get_config

def migrate():
    """Add user_name and user_email columns to searches table."""
    engine = create_engine(DATABASE_URL, echo=True)

    with engine.connect() as conn:
        try:
            # Check if columns already exist
            result = conn.execute(text("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'searches'
                AND COLUMN_NAME IN ('user_name', 'user_email')
            """))
            existing_columns = {row[0] for row in result}

            # Add user_name if it doesn't exist
            if 'user_name' not in existing_columns:
                print("Adding user_name column...")
                conn.execute(text("""
                    ALTER TABLE searches
                    ADD COLUMN user_name VARCHAR(255) NULL
                """))
                conn.commit()
                print("✓ Added user_name column")
            else:
                print("✓ user_name column already exists")

            # Add user_email if it doesn't exist
            if 'user_email' not in existing_columns:
                print("Adding user_email column...")
                conn.execute(text("""
                    ALTER TABLE searches
                    ADD COLUMN user_email VARCHAR(255) NULL
                """))
                conn.commit()
                print("✓ Added user_email column")
            else:
                print("✓ user_email column already exists")

            print("\n✓ Migration complete!")

        except Exception as e:
            print(f"\n✗ Migration failed: {str(e)}")
            raise

if __name__ == "__main__":
    print("Running migration to add user_name and user_email columns...")
    migrate()
