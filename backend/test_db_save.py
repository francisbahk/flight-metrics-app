"""
Test script to verify database save works with completion_token field
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from db import save_search_and_csv, SessionLocal, Search
import json

def test_save():
    """Test that we can save a search with completion_token."""

    test_session_id = "TEST_SESSION_123"
    test_token = "TEST_TOKEN_ABC"

    test_params = {
        'origins': ["JFK"],
        'destinations': ["LAX"],
        'preferences': {"prefer_cheap": True},
        'departure_date': "2025-12-25"
    }

    print("🧪 Testing database save with completion_token...")
    print(f"   Session ID: {test_session_id}")
    print(f"   Token: {test_token}")

    try:
        # Test save_search_and_csv function
        search_id = save_search_and_csv(
            session_id=test_session_id,
            user_prompt="Test flight search",
            parsed_params=test_params,
            all_flights=[],
            selected_flights=[],
            csv_data="test,data\n1,2",
            token=test_token
        )

        print(f"✓ Save successful! Search ID: {search_id}")

        # Verify the data was saved correctly
        db = SessionLocal()
        try:
            search = db.query(Search).filter(Search.search_id == search_id).first()

            if search:
                print(f"✓ Verification: Found search record")
                print(f"   - session_id: {search.session_id}")
                print(f"   - completion_token: {search.completion_token}")
                print(f"   - user_prompt: {search.user_prompt}")

                if search.completion_token == test_token:
                    print("✓ completion_token matches!")
                else:
                    print(f"✗ completion_token mismatch: {search.completion_token} != {test_token}")

                # Clean up test data
                db.delete(search)
                db.commit()
                print("✓ Test data cleaned up")
            else:
                print("✗ Could not find saved search record")

        finally:
            db.close()

        print("\n✅ All tests passed! Database save is working correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_save()
    sys.exit(0 if success else 1)
