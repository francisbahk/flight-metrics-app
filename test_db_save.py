#!/usr/bin/env python3
"""
Test script to verify database save functionality.
"""
import sys
from datetime import datetime
from backend.db import save_search_and_csv, SessionLocal, Search, FlightCSV, test_connection, init_db

def test_database_connection():
    """Test 1: Verify database connection works."""
    print("=" * 80)
    print("TEST 1: Database Connection")
    print("=" * 80)
    result = test_connection()
    assert result, "Database connection failed"
    print("✓ PASSED: Database connection successful\n")
    return True

def test_database_tables():
    """Test 2: Verify tables exist."""
    print("=" * 80)
    print("TEST 2: Database Tables")
    print("=" * 80)
    try:
        init_db()
        print("✓ PASSED: Database tables created/verified\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

def test_save_search():
    """Test 3: Save a test search and verify it's saved."""
    print("=" * 80)
    print("TEST 3: Save Search to Database")
    print("=" * 80)

    # Create test data
    session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    user_prompt = "Test flight from NYC to LAX"
    parsed_params = {
        'origins': ['NYC'],
        'destinations': ['LAX'],
        'departure_date': '2025-12-01',
        'preferences': {}
    }
    all_flights = [
        {'price': 300, 'duration': '5h', 'stops': 0, 'airline': 'Test Air'},
        {'price': 250, 'duration': '6h', 'stops': 1, 'airline': 'Budget Air'}
    ]
    selected_flights = [all_flights[0]]
    csv_data = "name,price,duration\nFlight 1,300,5h"

    try:
        search_id = save_search_and_csv(
            session_id=session_id,
            user_prompt=user_prompt,
            parsed_params=parsed_params,
            all_flights=all_flights,
            selected_flights=selected_flights,
            csv_data=csv_data
        )

        print(f"✓ Save function returned search_id: {search_id}")

        # Verify it's in the database
        db = SessionLocal()
        try:
            search = db.query(Search).filter(Search.search_id == search_id).first()
            if not search:
                print(f"✗ FAILED: Search {search_id} not found in database")
                return False

            print(f"✓ Search found in database: {search.user_prompt}")

            csv_records = db.query(FlightCSV).filter(FlightCSV.search_id == search_id).all()
            if not csv_records:
                print(f"✗ FAILED: CSV record not found for search {search_id}")
                return False

            print(f"✓ CSV record found: {len(csv_records)} record(s)")
            print("✓ PASSED: Search saved and verified in database\n")
            return True

        finally:
            db.close()

    except Exception as e:
        import traceback
        print(f"✗ FAILED: Error during save")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        return False

def test_retrieve_latest():
    """Test 4: Verify we can retrieve the latest search."""
    print("=" * 80)
    print("TEST 4: Retrieve Latest Search")
    print("=" * 80)

    try:
        db = SessionLocal()
        try:
            search = db.query(Search).order_by(Search.created_at.desc()).first()
            if not search:
                print("✗ FAILED: No searches found in database")
                return False

            print(f"✓ Latest search: ID={search.search_id}")
            print(f"  Prompt: {search.user_prompt}")
            print(f"  Created: {search.created_at}")
            print("✓ PASSED: Successfully retrieved latest search\n")
            return True

        finally:
            db.close()
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DATABASE SAVE TEST SUITE")
    print("=" * 80 + "\n")

    tests = [
        test_database_connection,
        test_database_tables,
        test_save_search,
        test_retrieve_latest
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ TEST CRASHED: {e}\n")
            results.append(False)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
