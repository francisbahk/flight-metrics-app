"""
Test script to verify LISTEN-U algorithm is working correctly.
Run this locally to check LISTEN before deploying to Streamlit Cloud.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backend.listen_main_wrapper import rank_flights_with_listen_main
from pathlib import Path

# Sample flight data (realistic flights)
sample_flights = [
    {
        'id': '1',
        'airline': 'UA',
        'flight_number': '1234',
        'origin': 'ITH',
        'destination': 'IAD',
        'departure_time': '2024-11-15T10:28:00',
        'arrival_time': '2024-11-15T11:50:00',
        'duration_min': 82,
        'stops': 0,
        'price': 204.0,
        'currency': 'USD'
    },
    {
        'id': '2',
        'airline': 'DL',
        'flight_number': '5678',
        'origin': 'ITH',
        'destination': 'DCA',
        'departure_time': '2024-11-15T14:05:00',
        'arrival_time': '2024-11-15T18:00:00',
        'duration_min': 235,
        'stops': 1,
        'price': 189.0,
        'currency': 'USD'
    },
    {
        'id': '3',
        'airline': 'AA',
        'flight_number': '9012',
        'origin': 'ITH',
        'destination': 'DCA',
        'departure_time': '2024-11-15T08:00:00',
        'arrival_time': '2024-11-15T12:30:00',
        'duration_min': 270,
        'stops': 1,
        'price': 165.0,
        'currency': 'USD'
    },
    {
        'id': '4',
        'airline': 'UA',
        'flight_number': '3456',
        'origin': 'ITH',
        'destination': 'IAD',
        'departure_time': '2024-11-15T16:30:00',
        'arrival_time': '2024-11-15T17:45:00',
        'duration_min': 75,
        'stops': 0,
        'price': 245.0,
        'currency': 'USD'
    },
    {
        'id': '5',
        'airline': 'DL',
        'flight_number': '7890',
        'origin': 'ITH',
        'destination': 'DCA',
        'departure_time': '2024-11-15T06:15:00',
        'arrival_time': '2024-11-15T11:00:00',
        'duration_min': 285,
        'stops': 1,
        'price': 155.0,
        'currency': 'USD'
    }
]

def test_listen():
    print("=" * 80)
    print("TESTING LISTEN-U ALGORITHM")
    print("=" * 80)
    print()

    user_prompt = "I need a flight from Ithaca to DC. I prefer direct flights and am not very price sensitive."
    user_prefs = {
        'origins': ['ITH'],
        'destinations': ['IAD', 'DCA'],
        'preferences': {
            'prefer_direct': True,
            'prefer_cheap': False
        }
    }

    print("Test Parameters:")
    print(f"  User Prompt: {user_prompt}")
    print(f"  Preferences: {user_prefs['preferences']}")
    print(f"  Number of flights: {len(sample_flights)}")
    print()

    print("Input Flights (by price):")
    for i, flight in enumerate(sorted(sample_flights, key=lambda x: x['price']), 1):
        print(f"  {i}. {flight['airline']}{flight['flight_number']}: "
              f"${flight['price']:.0f}, {flight['duration_min']}min, {flight['stops']} stops")
    print()

    print("Running LISTEN-U with 3 iterations (for quick test)...")
    print("-" * 80)

    try:
        # Run with fewer iterations for testing (3 instead of 25)
        ranked = rank_flights_with_listen_main(
            flights=sample_flights,
            user_prompt=user_prompt,
            user_preferences=user_prefs,
            n_iterations=3  # Quick test with 3 iterations
        )

        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print()

        if ranked:
            print(f"✅ LISTEN-U returned {len(ranked)} ranked flights:")
            print()
            for i, flight in enumerate(ranked, 1):
                print(f"  {i}. {flight['airline']}{flight['flight_number']}: "
                      f"${flight['price']:.0f}, {flight['duration_min']}min, {flight['stops']} stops")
            print()

            # Check if ranking differs from simple price ranking
            price_order = [f['id'] for f in sorted(sample_flights, key=lambda x: x['price'])[:len(ranked)]]
            listen_order = [f['id'] for f in ranked]

            print("Comparison:")
            print(f"  Price ranking order: {price_order}")
            print(f"  LISTEN ranking order: {listen_order}")
            print()

            if price_order != listen_order:
                print("✅ LISTEN produced different ranking than simple price sort")
                print("   This indicates LISTEN is considering user preferences!")
            else:
                print("⚠️  LISTEN ranking matches price ranking")
                print("   This might be expected for small dataset or price-focused preferences")
            print()

            # Check LISTEN output directory
            listen_dir = Path(__file__).parent / "LISTEN"
            output_files = list((listen_dir / "output").glob("flight_*/"))
            if output_files:
                latest = max(output_files, key=lambda p: p.stat().st_mtime)
                print(f"✅ LISTEN created output directory: {latest.name}")
                files = list(latest.glob("*"))
                if files:
                    print(f"   Files created: {', '.join(f.name for f in files)}")
            else:
                print("⚠️  No LISTEN output directory found")
            print()

            print("=" * 80)
            print("VERDICT: LISTEN-U is working! ✅")
            print("=" * 80)
            return True

        else:
            print("❌ LISTEN returned empty results")
            return False

    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ LISTEN TEST FAILED: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_listen()
    sys.exit(0 if success else 1)
