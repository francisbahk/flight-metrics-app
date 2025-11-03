"""
Natural Language Parser for Flight Search
Extracts flight parameters from natural language text.
"""
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple


# Common airport code mappings
AIRPORT_MAPPINGS = {
    # Cities to airport codes
    "ithaca": "ITH",
    "san francisco": "SFO",
    "san fran": "SFO",
    "sf": "SFO",
    "new york": "JFK",
    "nyc": "JFK",
    "los angeles": "LAX",
    "la": "LAX",
    "boston": "BOS",
    "chicago": "ORD",
    "miami": "MIA",
    "seattle": "SEA",
    "denver": "DEN",
    "atlanta": "ATL",
    "dallas": "DFW",
    "houston": "IAH",
    "phoenix": "PHX",
    "philadelphia": "PHL",
    "washington": "IAD",
    "dc": "IAD",
    "orlando": "MCO",
    "las vegas": "LAS",
    "detroit": "DTW",
    "minneapolis": "MSP",
    "tampa": "TPA",
    "portland": "PDX",
    "nashville": "BNA",
    "austin": "AUS",
    "charlotte": "CLT",
    "san diego": "SAN",
    "raleigh": "RDU",
    "salt lake city": "SLC",
    "london": "LON",
    "paris": "PAR",
    "madrid": "MAD",
    "barcelona": "BCN",
    "delhi": "DEL",
    "mumbai": "BOM",
    "berlin": "BER",
    "frankfurt": "FRA",
    "munich": "MUC",
}

# Major airports that work well in Amadeus test API
TEST_API_MAJOR_AIRPORTS = [
    "JFK", "LAX", "ORD", "ATL", "DFW", "DEN", "SFO", "LAS", "SEA", "MCO",
    "EWR", "BOS", "MIA", "IAH", "PHX", "LON", "MAD", "BCN", "DEL", "BOM",
    "BER", "FRA", "MUC"
]

# Month name mappings
MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def extract_airport_code(text: str, is_destination: bool = False) -> Optional[str]:
    """
    Extract airport code from text.

    Args:
        text: Text to search for airport/city names
        is_destination: If True, looks after 'to'; if False, looks after 'from'

    Returns:
        3-letter IATA code or None
    """
    text_lower = text.lower()

    # Try to find explicit airport codes (3 capital letters)
    airport_codes = re.findall(r'\b([A-Z]{3})\b', text)
    if airport_codes:
        return airport_codes[0] if not is_destination else airport_codes[-1]

    # Look for "from X to Y" pattern
    from_to_pattern = r'from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s+on|\s+in|\s+\d|$)'
    match = re.search(from_to_pattern, text_lower)

    if match:
        origin_text = match.group(1).strip()
        dest_text = match.group(2).strip()

        if is_destination:
            # Check if destination matches known city
            for city, code in AIRPORT_MAPPINGS.items():
                if city in dest_text:
                    return code
        else:
            # Check if origin matches known city
            for city, code in AIRPORT_MAPPINGS.items():
                if city in origin_text:
                    return code

    # Fallback: search entire text for city names
    for city, code in AIRPORT_MAPPINGS.items():
        if city in text_lower:
            return code

    return None


def extract_date(text: str) -> Optional[str]:
    """
    Extract date from natural language text.

    Supports formats:
    - 10/31, 10/31/2024, 10-31, 10-31-2024
    - October 31, Oct 31, October 31st
    - in November (defaults to mid-month)

    Returns:
        Date in YYYY-MM-DD format or None
    """
    text_lower = text.lower()
    current_year = datetime.now().year

    # Try MM/DD or MM/DD/YYYY format
    date_slash = re.search(r'(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?', text)
    if date_slash:
        month = int(date_slash.group(1))
        day = int(date_slash.group(2))
        year = int(date_slash.group(3)) if date_slash.group(3) else current_year

        # Handle 2-digit year
        if year < 100:
            year += 2000

        try:
            date_obj = datetime(year, month, day)
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Try "Month Day" format (e.g., "October 31", "Nov 15")
    for month_name, month_num in MONTH_NAMES.items():
        # Pattern: "October 31" or "Oct 31st"
        pattern = rf'{month_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?'
        match = re.search(pattern, text_lower)
        if match:
            day = int(match.group(1))
            try:
                date_obj = datetime(current_year, month_num, day)
                # If date is in the past, assume next year
                if date_obj < datetime.now():
                    date_obj = datetime(current_year + 1, month_num, day)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                pass

    # Try "in Month" format (e.g., "in November")
    in_month_pattern = r'in\s+([a-z]+)'
    match = re.search(in_month_pattern, text_lower)
    if match:
        month_text = match.group(1)
        if month_text in MONTH_NAMES:
            month_num = MONTH_NAMES[month_text]
            # Default to 15th of the month
            try:
                date_obj = datetime(current_year, month_num, 15)
                # If date is in the past, assume next year
                if date_obj < datetime.now():
                    date_obj = datetime(current_year + 1, month_num, 15)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                pass

    return None


def extract_preferences(text: str) -> Dict[str, any]:
    """
    Extract flight preferences from text.

    Args:
        text: Natural language text

    Returns:
        Dict with preferences like max_stops, prefer_nonstop, etc.
    """
    text_lower = text.lower()
    preferences = {}

    # Check for direct flight preferences
    if any(word in text_lower for word in ["direct", "nonstop", "non-stop", "no stops"]):
        preferences['max_stops'] = 0
        preferences['prefer_nonstop'] = True

    # Check for comfort/class preferences
    if any(word in text_lower for word in ["comfortable", "comfort", "business", "first class"]):
        preferences['prefer_comfort'] = True

    # Check for cheap/budget preferences
    if any(word in text_lower for word in ["cheap", "cheapest", "budget", "affordable", "inexpensive"]):
        preferences['prefer_cheap'] = True

    # Check for fast/quick preferences
    if any(word in text_lower for word in ["fast", "quick", "fastest", "shortest"]):
        preferences['prefer_fast'] = True

    return preferences


def get_fallback_airport(airport_code: str) -> Tuple[str, str]:
    """
    Get a major airport fallback if the requested airport isn't in test API.

    Args:
        airport_code: Requested airport code

    Returns:
        Tuple of (fallback_code, reason_message)
    """
    if airport_code in TEST_API_MAJOR_AIRPORTS:
        return airport_code, ""

    # Fallback mappings for common small airports
    fallbacks = {
        "ITH": ("JFK", "Ithaca (ITH) not in test API, using JFK (New York) instead"),
        "SYR": ("JFK", "Syracuse (SYR) not in test API, using JFK (New York) instead"),
        "ALB": ("JFK", "Albany (ALB) not in test API, using JFK (New York) instead"),
        "ROC": ("JFK", "Rochester (ROC) not in test API, using JFK (New York) instead"),
        "BUF": ("JFK", "Buffalo (BUF) not in test API, using JFK (New York) instead"),
    }

    if airport_code in fallbacks:
        return fallbacks[airport_code]

    # Default fallback
    return "JFK", f"{airport_code} not in test API, using JFK (New York) instead"


def parse_flight_query(text: str) -> Dict[str, any]:
    """
    Parse natural language flight search query.

    Args:
        text: Natural language query (e.g., "I want to go from Ithaca to San Francisco on 10/31")

    Returns:
        Dictionary with extracted parameters and any warnings
    """
    result = {
        'origin': None,
        'destination': None,
        'date': None,
        'preferences': {},
        'warnings': [],
        'original_query': text,
        'parsed_successfully': False
    }

    # Extract origin
    origin = extract_airport_code(text, is_destination=False)
    if origin:
        result['origin'] = origin
        # Check if we need a fallback
        fallback_origin, warning = get_fallback_airport(origin)
        if warning:
            result['warnings'].append(warning)
            result['origin'] = fallback_origin
            result['original_origin'] = origin

    # Extract destination
    destination = extract_airport_code(text, is_destination=True)
    if destination:
        result['destination'] = destination
        # Check if we need a fallback
        fallback_dest, warning = get_fallback_airport(destination)
        if warning:
            result['warnings'].append(warning)
            result['destination'] = fallback_dest
            result['original_destination'] = destination

    # Extract date
    date = extract_date(text)
    if date:
        result['date'] = date
    else:
        # Default to 30 days from now
        default_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        result['date'] = default_date
        result['warnings'].append(f"No date found, using {default_date}")

    # Extract preferences
    result['preferences'] = extract_preferences(text)

    # Check if parsing was successful
    if result['origin'] and result['destination']:
        result['parsed_successfully'] = True

    return result
