"""Offline city-based airport search and nearest-airport lookup using OurAirports data."""

import csv
import json
import math
import os
import streamlit as st

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OURAIRPORTS_PATH = os.path.join(_DATA_DIR, "ourairports.csv")
REGIONS_PATH = os.path.join(_DATA_DIR, "regions.csv")
NEAREST_AIRPORTS_PATH = os.path.join(_DATA_DIR, "nearest_airports.json")


@st.cache_data
def _load_region_names() -> dict[str, str]:
    """Load region code -> name mapping from OurAirports regions CSV."""
    regions = {}
    with open(REGIONS_PATH) as f:
        for row in csv.DictReader(f):
            code = row.get("code", "").strip()
            name = row.get("name", "").strip()
            if code and name:
                regions[code] = name
    return regions


@st.cache_data
def load_airports() -> list[dict]:
    """Load large + medium airports with IATA codes from OurAirports CSV."""
    regions = _load_region_names()
    airports = []
    with open(OURAIRPORTS_PATH) as f:
        for row in csv.DictReader(f):
            iata = (row.get("iata_code") or "").strip()
            ap_type = row.get("type", "")
            if not iata or ap_type not in ("large_airport", "medium_airport"):
                continue
            region_code = row.get("iso_region", "").strip()
            region_name = regions.get(region_code, "")
            airports.append({
                "iata": iata,
                "name": row.get("name", ""),
                "city": row.get("municipality", ""),
                "region": region_name,
                "country": row.get("iso_country", ""),
                "lat": float(row.get("latitude_deg") or 0),
                "lon": float(row.get("longitude_deg") or 0),
                "type": ap_type,
            })
    return airports


@st.cache_data
def _load_cities() -> list[dict]:
    """Build deduplicated city list with their airports grouped."""
    airports = load_airports()

    # Group airports by (city, region, country)
    city_map: dict[tuple[str, str, str], list[dict]] = {}
    for ap in airports:
        city = ap["city"].strip()
        region = ap["region"].strip()
        country = ap["country"].strip()
        if not city:
            continue
        key = (city, region, country)
        if key not in city_map:
            city_map[key] = []
        city_map[key].append(ap)

    cities = []
    for (city, region, country), aps in city_map.items():
        codes = [a["iata"] for a in aps]
        if region:
            label = f"{city}, {region}, {country}"
        else:
            label = f"{city}, {country}"
        avg_lat = sum(a["lat"] for a in aps) / len(aps)
        avg_lon = sum(a["lon"] for a in aps) / len(aps)
        cities.append({
            "city": city,
            "region": region,
            "country": country,
            "label": label,
            "key": f"{city}|{region}|{country}",
            "airports": codes,
            "lat": avg_lat,
            "lon": avg_lon,
        })
    return cities


@st.cache_data
def _load_nearest_lookup() -> dict:
    """Load the precomputed nearest-airports JSON."""
    with open(NEAREST_AIRPORTS_PATH) as f:
        return json.load(f)


def get_countries() -> list[str]:
    """Return sorted list of unique country codes from the city data."""
    cities = _load_cities()
    return sorted({c["country"] for c in cities if c["country"]})


def get_regions(country: str | None = None) -> list[str]:
    """Return sorted list of unique region names, optionally filtered by country."""
    cities = _load_cities()
    regions = set()
    for c in cities:
        if c["region"] and (country is None or c["country"] == country):
            regions.add(c["region"])
    return sorted(regions)


def search_cities(
    query: str,
    country: str | None = None,
    region: str | None = None,
    max_results: int = 25,
) -> list[tuple[str, str]]:
    """Search cities by name or region. Returns (label, city_key) tuples for st_searchbox."""
    cities = _load_cities()

    # Apply optional filters
    filtered = cities
    if country:
        filtered = [c for c in filtered if c["country"] == country]
    if region:
        filtered = [c for c in filtered if c["region"] == region]

    # If filters are set but no query, return all filtered cities (sorted alphabetically)
    has_filters = country or region
    if not query or not query.strip():
        if has_filters:
            sorted_cities = sorted(filtered, key=lambda c: c["city"].upper())
            return [(c["label"], c["key"]) for c in sorted_cities[:max_results]]
        return []

    q = query.strip().upper()
    if len(q) < 2 and not has_filters:
        return []

    prefix = []
    substring = []

    for c in filtered:
        city_upper = c["city"].upper()
        region_upper = c["region"].upper()
        if city_upper.startswith(q):
            prefix.append(c)
        elif q in city_upper or q in region_upper:
            substring.append(c)

    results = prefix + substring
    return [(c["label"], c["key"]) for c in results[:max_results]]


def _haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_airports_for_city(city_key: str, max_total: int = 6) -> list[dict]:
    """Get airports for a selected city: all airports in that city + nearest neighbors.

    Returns list of dicts: [{"iata": "JFK", "distance_mi": 0.0}, ...]
    """
    cities = _load_cities()
    city = next((c for c in cities if c["key"] == city_key), None)
    if not city:
        return []

    # Start with all airports in this city (distance 0)
    result = [{"iata": code, "distance_mi": 0.0} for code in city["airports"]]
    seen = set(city["airports"])

    # Fill remaining slots from precomputed nearest neighbors
    lookup = _load_nearest_lookup()
    for code in city["airports"]:
        for neighbor in lookup.get(code, []):
            if neighbor["iata"] not in seen and len(result) < max_total:
                result.append(neighbor)
                seen.add(neighbor["iata"])

    # Sort: city airports first (distance 0), then by distance
    result.sort(key=lambda x: x["distance_mi"])
    return result[:max_total]
