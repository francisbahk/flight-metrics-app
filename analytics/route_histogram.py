"""
route_histogram.py
------------------
Shows the valid route+day inventory from static_flights_filtered.json:
  - Total number of valid routes (those with >= 50 itineraries)
  - Histogram: number of routes vs itinerary count range

Usage:
    python analytics/route_histogram.py
"""

import json
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "static_flights_filtered.json"
MIN_FLIGHTS = 50  # threshold used in app.py to define "valid"
BIN_WIDTH = 50    # width of each histogram bucket


def load_flights(path=DATA_PATH):
    with open(path) as f:
        return json.load(f)


def route_day_counts(flights):
    counts = defaultdict(int)
    for flight in flights:
        key = (flight["origin"], flight["destination"], flight["day_of_week"])
        counts[key] += 1
    return counts


def make_buckets(values, bin_width=BIN_WIDTH):
    """Build (lo, hi, label) bucket tuples that cover all values."""
    if not values:
        return []
    lo = (min(values) // bin_width) * bin_width
    hi = ((max(values) // bin_width) + 1) * bin_width
    buckets = []
    for start in range(lo, hi, bin_width):
        end = start + bin_width
        label = f"{start} – {end - 1}"
        buckets.append((start, end, label))
    return buckets


def main():
    flights = load_flights()
    counts = route_day_counts(flights)

    # Only valid routes (>= MIN_FLIGHTS)
    valid = {k: v for k, v in counts.items() if v >= MIN_FLIGHTS}
    total_valid = len(valid)
    values = list(valid.values())

    print(f"\n{'='*55}")
    print(f"  Static Flight Database — Route Inventory")
    print(f"{'='*55}")
    print(f"  Total (route, day) pairs with >= {MIN_FLIGHTS} itineraries: {total_valid}")
    if values:
        print(f"  Min itineraries per route: {min(values)}")
        print(f"  Max itineraries per route: {max(values)}")
        avg = sum(values) / len(values)
        print(f"  Avg itineraries per route: {avg:.1f}")
    print()

    # Histogram
    buckets = make_buckets(values, BIN_WIDTH)
    max_count = max((sum(1 for v in values if lo <= v < hi) for lo, hi, _ in buckets), default=1)
    bar_scale = 40 / max(max_count, 1)

    print(f"  {'Range':<14}  {'# Routes':>8}  {'Bar'}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*42}")

    for lo, hi, label in buckets:
        n = sum(1 for v in values if lo <= v < hi)
        bar = "█" * round(n * bar_scale)
        print(f"  {label:<14}  {n:>8}  {bar}")

    print()

    # Sorted top routes
    print(f"  Top 10 routes by itinerary count:")
    print(f"  {'-'*50}")
    top = sorted(valid.items(), key=lambda x: x[1], reverse=True)[:10]
    for (origin, dest, day), cnt in top:
        print(f"  {origin} -> {dest} ({day:<9})  {cnt:>5} itineraries")
    print()


if __name__ == "__main__":
    main()
