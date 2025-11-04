"""
Example Algorithm: Cheapest Direct Flights First
This algorithm prioritizes direct flights and sorts them by price.
"""

def rank_flights(flights, preferences):
    """
    Rank flights by prioritizing direct flights, then by price.

    Args:
        flights: List of flight dictionaries
        preferences: Dictionary with user preferences (not used in this simple example)

    Returns:
        List of flights sorted (best first)
    """
    # Sort by: stops first (0 stops = direct), then by price
    return sorted(flights, key=lambda x: (x.get('stops', 999), x.get('price', float('inf'))))
