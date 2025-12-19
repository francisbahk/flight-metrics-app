"""
Migration: Add completion_token column to searches table
"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Database connection
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = os.getenv('MYSQL_PORT', '3306')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'flight_rankings')
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

def migrate():
    """Add completion_token column to searches table."""
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT COUNT(*) as count
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = :db_name
            AND TABLE_NAME = 'searches'
            AND COLUMN_NAME = 'completion_token'
        """), {"db_name": MYSQL_DATABASE})

        exists = result.fetchone()[0] > 0

        if exists:
            print("✓ Column completion_token already exists in searches table")
            return

        # Add the column
        print("Adding completion_token column to searches table...")
        conn.execute(text("""
            ALTER TABLE searches
            ADD COLUMN completion_token VARCHAR(255) NULL,
            ADD INDEX idx_completion_token (completion_token)
        """))
        conn.commit()

        print("✓ Successfully added completion_token column to searches table")

if __name__ == "__main__":
    print("Running migration: Add completion_token to searches table")
    migrate()
    print("Migration complete!")
