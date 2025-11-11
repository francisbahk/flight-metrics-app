"""
Setup script for MySQL database.
Run this once to create all required tables.
"""
from backend.db import init_db, test_connection

if __name__ == "__main__":
    print("=" * 60)
    print("Flight Ranking App - Database Setup")
    print("=" * 60)
    print()

    # Test connection first
    print("Step 1: Testing database connection...")
    if not test_connection():
        print()
        print("⚠️  Database connection failed!")
        print()
        print("Please check your .env file and ensure:")
        print("1. MySQL server is running")
        print("2. Database credentials are correct")
        print("3. Database exists (or create it with: CREATE DATABASE flight_rankings;)")
        print()
        exit(1)

    print()
    print("Step 2: Creating database tables...")
    try:
        init_db()
        print()
        print("=" * 60)
        print("✅ Database setup complete!")
        print("=" * 60)
        print()
        print("Tables created:")
        print("  - searches (stores user queries)")
        print("  - flights_shown (stores all flights shown to users)")
        print("  - user_rankings (stores user's top 5 rankings)")
        print()
        print("You can now run the app with: streamlit run app_new.py")
        print()

    except Exception as e:
        print()
        print(f"✗ Error creating tables: {str(e)}")
        print()
        exit(1)