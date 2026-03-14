"""
Pre-fetch static flight data from Amadeus for the pilot study.

Fetches all 28 curated routes for March 1-7 2026 (Sun-Sat),
parses each offer, and saves everything to static_flights.json.

Run once before the pilot study:
    python fetch_static_flights.py
"""
import json
import os
import time

# Load credentials from .streamlit/secrets.toml if not already in environment
def _load_streamlit_secrets():
    secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
    if not os.path.exists(secrets_path):
        return
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            # Fallback: simple key=value parsing for the keys we need
            with open(secrets_path) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        k, _, v = line.partition('=')
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k in ('AMADEUS_API_KEY', 'AMADEUS_API_SECRET', 'AMADEUS_BASE_URL'):
                            os.environ.setdefault(k, v)
            return
    with open(secrets_path, 'rb') as f:
        secrets = tomllib.load(f)
    for k in ('AMADEUS_API_KEY', 'AMADEUS_API_SECRET', 'AMADEUS_BASE_URL'):
        if k in secrets:
            os.environ.setdefault(k, secrets[k])

_load_streamlit_secrets()

from backend.amadeus_client import AmadeusClient

ROUTES = [
    ("JFK", "LAX"),
    ("LAX", "JFK"),
    ("JFK", "SFO"),
    ("SFO", "JFK"),
    ("JFK", "MIA"),
    ("MIA", "JFK"),
    ("JFK", "ORD"),
    ("ORD", "JFK"),
    ("BOS", "LAX"),
    ("LAX", "BOS"),
    ("BOS", "MIA"),
    ("MIA", "BOS"),
    ("ATL", "LAX"),
    ("LAX", "ATL"),
    ("ATL", "JFK"),
    ("ORD", "LAX"),
    ("LAX", "ORD"),
    ("ORD", "MIA"),
    ("DFW", "LAX"),
    ("LAX", "DFW"),
    ("DFW", "JFK"),
    ("SEA", "LAX"),
    ("LAX", "SEA"),
    ("DEN", "JFK"),
    ("JFK", "DEN"),
    ("SFO", "ORD"),
    ("ORD", "SFO"),
    ("PHX", "JFK"),
]

# March 1 2026 = Sunday
DATES = {
    "2026-03-01": "Sunday",
    "2026-03-02": "Monday",
    "2026-03-03": "Tuesday",
    "2026-03-04": "Wednesday",
    "2026-03-05": "Thursday",
    "2026-03-06": "Friday",
    "2026-03-07": "Saturday",
}


def fetch_all():
    client = AmadeusClient()
    all_flights = []
    total = len(ROUTES) * len(DATES)
    done = 0

    for date_str, day_name in DATES.items():
        for origin, dest in ROUTES:
            done += 1
            print(f"[{done}/{total}] {origin} → {dest} on {date_str} ({day_name})")
            try:
                offers = client.search_flights(
                    origin=origin,
                    destination=dest,
                    departure_date=date_str,
                    adults=1,
                    max_results=250,
                )
                for offer in offers:
                    parsed = client.parse_flight_offer(offer)
                    if parsed:
                        parsed["day_of_week"] = day_name
                        parsed["actual_date"] = date_str
                        all_flights.append(parsed)
                print(f"  ✓ {len(offers)} flights fetched ({len(all_flights)} total so far)")
            except Exception as e:
                print(f"  ✗ Error: {e}")

            # Be polite to the API
            time.sleep(0.3)

    # Look up airline names for all carriers in the dataset
    print("\nLooking up airline names...")
    all_codes = list(set(f.get("carrier_code", "") for f in all_flights if f.get("carrier_code")))
    airline_map = client.get_airline_names(all_codes)
    for flight in all_flights:
        code = flight.get("carrier_code", "")
        if code and code in airline_map:
            flight["airline_name"] = airline_map[code]

    output_path = "static_flights.json"
    with open(output_path, "w") as f:
        json.dump(all_flights, f)

    print(f"\n✅ Done. {len(all_flights)} flights saved to {output_path}")
    print(f"   Routes: {len(ROUTES)}, Dates: {len(DATES)}")

    # Summary
    unique_origins = sorted(set(f["origin"] for f in all_flights))
    unique_dests = sorted(set(f["destination"] for f in all_flights))
    print(f"   Unique origins: {unique_origins}")
    print(f"   Unique destinations: {unique_dests}")


if __name__ == "__main__":
    fetch_all()
