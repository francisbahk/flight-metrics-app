"""
Utility functions for parsing durations, calculating distances, and flight data processing.
"""
import re
import math
from typing import Tuple, List, Dict


# Airport coordinates (latitude, longitude) for major airports
AIRPORT_COORDINATES = {
    # US Airports
    "JFK": (40.6413, -73.7781),  # New York JFK
    "LAX": (33.9416, -118.4085),  # Los Angeles
    "ORD": (41.9742, -87.9073),  # Chicago O'Hare
    "ATL": (33.6407, -84.4277),  # Atlanta
    "DFW": (32.8998, -97.0403),  # Dallas Fort Worth
    "DEN": (39.8561, -104.6737),  # Denver
    "SFO": (37.6213, -122.3790),  # San Francisco
    "SEA": (47.4502, -122.3088),  # Seattle
    "LAS": (36.0840, -115.1537),  # Las Vegas
    "MCO": (28.4312, -81.3081),  # Orlando
    "MIA": (25.7959, -80.2870),  # Miami
    "PHX": (33.4352, -112.0101),  # Phoenix
    "BOS": (42.3656, -71.0096),  # Boston
    "IAH": (29.9902, -95.3368),  # Houston
    "EWR": (40.6895, -74.1745),  # Newark
    "MSP": (44.8848, -93.2223),  # Minneapolis
    "DTW": (42.2162, -83.3554),  # Detroit
    "PHL": (39.8744, -75.2424),  # Philadelphia
    "LGA": (40.7769, -73.8740),  # LaGuardia
    "BWI": (39.1774, -76.6684),  # Baltimore
    "SLC": (40.7899, -111.9791),  # Salt Lake City
    "IAD": (38.9531, -77.4565),  # Washington Dulles
    "DCA": (38.8521, -77.0377),  # Washington Reagan
    "PDX": (45.5898, -122.5951),  # Portland
    "SAN": (32.7336, -117.1897),  # San Diego

    # International Airports
    "LHR": (51.4700, -0.4543),  # London Heathrow
    "CDG": (49.0097, 2.5479),  # Paris Charles de Gaulle
    "FRA": (50.0379, 8.5622),  # Frankfurt
    "AMS": (52.3105, 4.7683),  # Amsterdam
    "MAD": (40.4983, -3.5676),  # Madrid
    "BCN": (41.2974, 2.0833),  # Barcelona
    "FCO": (41.8003, 12.2389),  # Rome Fiumicino
    "MUC": (48.3537, 11.7750),  # Munich
    "ZRH": (47.4582, 8.5556),  # Zurich
    "VIE": (48.1103, 16.5697),  # Vienna
    "CPH": (55.6180, 12.6508),  # Copenhagen
    "ARN": (59.6519, 17.9186),  # Stockholm
    "OSL": (60.1939, 11.1004),  # Oslo
    "HEL": (60.3172, 24.9633),  # Helsinki
    "IST": (41.2753, 28.7519),  # Istanbul
    "DXB": (25.2532, 55.3657),  # Dubai
    "DOH": (25.2731, 51.6082),  # Doha
    "SIN": (1.3644, 103.9915),  # Singapore
    "HKG": (22.3080, 113.9185),  # Hong Kong
    "NRT": (35.7720, 140.3929),  # Tokyo Narita
    "HND": (35.5494, 139.7798),  # Tokyo Haneda
    "ICN": (37.4602, 126.4407),  # Seoul Incheon
    "PEK": (40.0799, 116.6031),  # Beijing
    "PVG": (31.1443, 121.8083),  # Shanghai Pudong
    "BKK": (13.6900, 100.7501),  # Bangkok
    "KUL": (2.7456, 101.7072),  # Kuala Lumpur
    "SYD": (-33.9399, 151.1753),  # Sydney
    "MEL": (-37.6690, 144.8410),  # Melbourne
    "YYZ": (43.6777, -79.6248),  # Toronto
    "YVR": (49.1967, -123.1815),  # Vancouver
    "MEX": (19.4363, -99.0721),  # Mexico City
    "GRU": (-23.4356, -46.4731),  # São Paulo
    "EZE": (-34.8222, -58.5358),  # Buenos Aires
    "GIG": (-22.8099, -43.2505),  # Rio de Janeiro
    "BOG": (4.7016, -74.1469),  # Bogotá
    "LIM": (-12.0219, -77.1143),  # Lima
    "SCL": (-33.3930, -70.7858),  # Santiago
}


def parse_duration_to_minutes(duration_str: str) -> float:
    """
    Parse ISO 8601 duration format to minutes.

    Format: PT{hours}H{minutes}M
    Examples:
        PT2H30M -> 150.0 (2 hours 30 minutes)
        PT1H15M -> 75.0
        PT45M -> 45.0
        PT3H -> 180.0

    Args:
        duration_str: ISO 8601 duration string (e.g., "PT2H30M")

    Returns:
        Duration in minutes as a float
    """
    if not duration_str:
        return 0.0

    # Remove 'PT' prefix
    duration_str = duration_str.replace('PT', '')

    hours = 0
    minutes = 0

    # Extract hours if present
    hour_match = re.search(r'(\d+)H', duration_str)
    if hour_match:
        hours = int(hour_match.group(1))

    # Extract minutes if present
    minute_match = re.search(r'(\d+)M', duration_str)
    if minute_match:
        minutes = int(minute_match.group(1))

    return float(hours * 60 + minutes)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using Haversine formula.

    Args:
        lat1: Latitude of point 1 in degrees
        lon1: Longitude of point 1 in degrees
        lat2: Latitude of point 2 in degrees
        lon2: Longitude of point 2 in degrees

    Returns:
        Distance in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    distance = R * c
    return distance


def calculate_flight_distances(origin: str, destination: str) -> Tuple[float, float]:
    """
    Calculate distances from a reference point (e.g., user location) to origin and destination airports.

    For this implementation, we'll use JFK as the default reference point.
    In a production system, this could be parameterized based on user location.

    Args:
        origin: Origin airport code (e.g., "LAX")
        destination: Destination airport code (e.g., "JFK")

    Returns:
        Tuple of (distance_from_reference_to_origin, distance_from_reference_to_destination) in km
        Returns (None, None) if airport coordinates not found
    """
    # Default reference point (could be made configurable)
    reference_airport = "JFK"

    # Get coordinates
    ref_coords = AIRPORT_COORDINATES.get(reference_airport)
    origin_coords = AIRPORT_COORDINATES.get(origin)
    dest_coords = AIRPORT_COORDINATES.get(destination)

    # Return None if any coordinates are missing
    if not ref_coords or not origin_coords or not dest_coords:
        return None, None

    # Calculate distances
    dis_from_origin = haversine_distance(
        ref_coords[0], ref_coords[1],
        origin_coords[0], origin_coords[1]
    )

    dis_from_dest = haversine_distance(
        ref_coords[0], ref_coords[1],
        dest_coords[0], dest_coords[1]
    )

    return dis_from_origin, dis_from_dest


def interleave_rankings(ranking_a: List[int], ranking_b: List[int]) -> List[Dict]:
    """
    Interleave two ranked lists for Team Draft evaluation.

    Creates an interleaved list alternating between the two rankings,
    tracking which algorithm each item came from.

    Args:
        ranking_a: List of flight IDs from algorithm A
        ranking_b: List of flight IDs from algorithm B

    Returns:
        List of dictionaries with 'flight_id' and 'source' ('a' or 'b')

    Example:
        ranking_a = [1, 2, 3]
        ranking_b = [4, 5, 6]
        Result: [
            {'flight_id': 1, 'source': 'a'},
            {'flight_id': 4, 'source': 'b'},
            {'flight_id': 2, 'source': 'a'},
            {'flight_id': 5, 'source': 'b'},
            ...
        ]
    """
    interleaved = []
    max_len = max(len(ranking_a), len(ranking_b))

    for i in range(max_len):
        # Add from ranking A
        if i < len(ranking_a):
            interleaved.append({
                'flight_id': ranking_a[i],
                'source': 'a'
            })

        # Add from ranking B
        if i < len(ranking_b):
            interleaved.append({
                'flight_id': ranking_b[i],
                'source': 'b'
            })

    return interleaved


def calculate_time_of_day_seconds(time_str: str) -> int:
    """
    Calculate seconds from midnight for a given time.

    Args:
        time_str: Time string in ISO format (e.g., "2024-01-15T14:30:00")

    Returns:
        Seconds from midnight (0-86399)
    """
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        return dt.hour * 3600 + dt.minute * 60 + dt.second
    except Exception:
        return 0
