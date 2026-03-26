"""One-time script to pre-compute the 5 nearest commercial airports for every airport.

Uses OurAirports CSV filtered to large_airport + medium_airport.
Outputs: data/nearest_airports.json

Run: python -m scripts.precompute_nearest_airports
"""

import csv
import json
import math
import time

OURAIRPORTS_PATH = "data/ourairports.csv"
OUTPUT_PATH = "data/nearest_airports.json"


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_commercial_airports():
    airports = []
    with open(OURAIRPORTS_PATH) as f:
        for row in csv.DictReader(f):
            iata = (row.get("iata_code") or "").strip()
            ap_type = row.get("type", "")
            if not iata or ap_type not in ("large_airport", "medium_airport"):
                continue
            airports.append({
                "iata": iata,
                "lat": float(row.get("latitude_deg") or 0),
                "lon": float(row.get("longitude_deg") or 0),
            })
    return airports


def main():
    airports = load_commercial_airports()
    print(f"Computing nearest 5 for {len(airports)} commercial airports...")
    start = time.time()

    result = {}
    for i, ap in enumerate(airports):
        distances = []
        for other in airports:
            if other["iata"] == ap["iata"]:
                continue
            d = haversine_miles(ap["lat"], ap["lon"], other["lat"], other["lon"])
            distances.append({"iata": other["iata"], "distance_mi": round(d, 1)})
        distances.sort(key=lambda x: x["distance_mi"])
        result[ap["iata"]] = distances[:5]

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            pct = (i + 1) / len(airports) * 100
            print(f"  {i + 1}/{len(airports)} ({pct:.0f}%) - {elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    size_mb = len(json.dumps(result, separators=(",", ":"))) / 1024 / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_mb:.1f} MB, {len(result)} airports)")


if __name__ == "__main__":
    main()
