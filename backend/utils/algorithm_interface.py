"""
Flight Ranking Algorithm Interface

All algorithms must implement the rank_flights function with this exact signature.
"""
from typing import List, Dict, Any


def rank_flights(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """
    Rank flights based on your algorithm's logic.

    REQUIRED FUNCTION SIGNATURE - Do not change!

    Args:
        flights: List[Dict] - List of flight dictionaries with these fields:
            - id: str - Unique flight identifier
            - price: float - Ticket price in USD
            - duration_min: float - Total flight duration in minutes
            - stops: int - Number of stops/layovers (0 = direct)
            - airline: str - Airline code (e.g., "AA", "DL", "UA")
            - flight_number: str - Flight number
            - departure_time: str - Departure datetime ISO format
            - arrival_time: str - Arrival datetime ISO format
            - currency: str - Price currency (usually "USD")
            - origin: str (optional) - Origin airport code
            - destination: str (optional) - Destination airport code

        preferences: Dict - User preferences extracted from NL query:
            - prefer_cheap: bool - User wants cheap flights
            - prefer_fast: bool - User wants fast flights
            - prefer_nonstop: bool - User wants direct flights
            - max_stops: int - Maximum stops (if specified)
            - prefer_comfort: bool - User wants comfortable flights
            - original_query: str - The full natural language query

    Returns:
        List[Dict] - The same flights list, sorted by your ranking (best first)

    Example Implementation (Cheapest):
        def rank_flights(flights, preferences):
            return sorted(flights, key=lambda x: x['price'])

    Example Implementation (Fastest):
        def rank_flights(flights, preferences):
            return sorted(flights, key=lambda x: x['duration_min'])

    Example Implementation (Preference-aware):
        def rank_flights(flights, preferences):
            if preferences.get('prefer_cheap'):
                return sorted(flights, key=lambda x: x['price'])
            elif preferences.get('prefer_fast'):
                return sorted(flights, key=lambda x: x['duration_min'])
            else:
                # Default: balance price and duration
                return sorted(flights, key=lambda x: x['price'] + x['duration_min']/10)

    Notes:
        - Your algorithm MUST return all input flights (don't filter them out)
        - The returned list MUST be sorted (best flight first)
        - You can add custom fields to flights if needed
        - Don't modify the original flight dictionaries (make copies if needed)
        - Algorithm should run quickly (<1 second for 50 flights)
    """
    raise NotImplementedError("Algorithm must implement rank_flights function")


# Validation function
def validate_algorithm(algorithm_func, test_flights: List[Dict] = None) -> tuple[bool, str]:
    """
    Validate that an algorithm meets the interface requirements.

    Returns:
        (is_valid, error_message)
    """
    import inspect

    # Check function signature
    sig = inspect.signature(algorithm_func)
    params = list(sig.parameters.keys())

    if len(params) != 2:
        return False, f"Algorithm must take exactly 2 parameters (flights, preferences), got {len(params)}"

    if params != ['flights', 'preferences']:
        return False, f"Parameters must be named 'flights' and 'preferences', got {params}"

    # Test with sample data if provided
    if test_flights:
        try:
            test_prefs = {'prefer_cheap': True}
            result = algorithm_func(test_flights, test_prefs)

            if not isinstance(result, list):
                return False, f"Algorithm must return a list, got {type(result)}"

            if len(result) != len(test_flights):
                return False, f"Algorithm must return all flights ({len(test_flights)}), got {len(result)}"

        except Exception as e:
            return False, f"Algorithm failed on test data: {str(e)}"

    return True, "Algorithm is valid"
