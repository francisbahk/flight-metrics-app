"""
route_frequency.py
------------------
For every (route, day) pair in static_flights.json, bucket by flight count
and print the proportion that falls in each range.

Usage:
    python analytics/route_frequency.py
"""

import json
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "static_flights.json"

BUCKETS = [
    (0,  10,  "< 10"),
    (10, 20,  "10 – 19"),
    (20, 30,  "20 – 29"),
    (30, 40,  "30 – 39"),
    (40, 50,  "40 – 49"),
    (50, float("inf"), "50+"),
]


def load_flights(path=DATA_PATH):
    with open(path) as f:
        return json.load(f)


def route_day_counts(flights):
    counts = defaultdict(int)
    for flight in flights:
        key = (flight["origin"], flight["destination"], flight["day_of_week"])
        counts[key] += 1
    return counts


def main():
    flights = load_flights()
    counts = route_day_counts(flights)
    values = list(counts.values())
    total = len(values)

    print(f"\nTotal (route, day) pairs: {total}\n")
    print(f"  {'Range':<10}  {'Count':>6}  {'Proportion':>10}  {'Bar'}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*30}")

    for lo, hi, label in BUCKETS:
        n = sum(1 for v in values if lo <= v < hi)
        pct = n / total
        bar = "█" * round(pct * 40)
        print(f"  {label:<10}  {n:>6}  {pct:>9.1%}  {bar}")

    print()


if __name__ == "__main__":
    main()
