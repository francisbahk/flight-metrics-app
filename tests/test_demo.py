"""
Test script to validate the interactive demo works without errors.
"""
from components.interactive_demo import get_fake_flight_data
from datetime import datetime

def test_fake_flights():
    """Test that fake flight data has all required fields."""
    flights = get_fake_flight_data()

    required_fields = [
        'id', 'airline', 'flight_number', 'origin', 'destination',
        'departure_time', 'arrival_time', 'duration', 'duration_min',
        'stops', 'price', 'segments'
    ]

    print(f"Testing {len(flights)} fake flights...")

    for i, flight in enumerate(flights):
        # Check all required fields exist
        missing = [f for f in required_fields if f not in flight]
        if missing:
            print(f"❌ Flight {i+1} missing fields: {missing}")
            return False

        # Test datetime parsing (this is what was failing)
        try:
            dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
            arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
        except Exception as e:
            print(f"❌ Flight {i+1} datetime parse error: {e}")
            return False

        # Validate data types
        if not isinstance(flight['price'], (int, float)):
            print(f"❌ Flight {i+1} price is not numeric: {type(flight['price'])}")
            return False

        if not isinstance(flight['duration_min'], int):
            print(f"❌ Flight {i+1} duration_min is not int: {type(flight['duration_min'])}")
            return False

        if not isinstance(flight['stops'], int):
            print(f"❌ Flight {i+1} stops is not int: {type(flight['stops'])}")
            return False

        print(f"✓ Flight {i+1}: {flight['airline']} {flight['flight_number']} - ${flight['price']}")

    print(f"\n✅ All {len(flights)} flights validated successfully!")
    return True

if __name__ == "__main__":
    success = test_fake_flights()
    exit(0 if success else 1)
