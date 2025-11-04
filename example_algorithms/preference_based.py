"""
Example Algorithm: Preference-Based Ranker
This algorithm adapts based on user preferences from their query.
"""

def rank_flights(flights, preferences):
    """
    Rank flights based on user's stated preferences.

    This algorithm checks the preferences dictionary and adjusts
    the ranking strategy accordingly.

    Args:
        flights: List of flight dictionaries
        preferences: Dictionary with these keys:
            - prefer_cheap (bool): User wants cheap flights
            - prefer_fast (bool): User wants fast flights
            - prefer_nonstop (bool): User wants direct flights
            - prefer_comfort (bool): User wants comfortable flights
            - original_query (str): User's original search query

    Returns:
        List of flights sorted based on preferences
    """
    # Check what user prefers
    prefer_cheap = preferences.get('prefer_cheap', False)
    prefer_fast = preferences.get('prefer_fast', False)
    prefer_nonstop = preferences.get('prefer_nonstop', False)

    # If user explicitly wants cheap flights
    if prefer_cheap:
        return sorted(flights, key=lambda x: x.get('price', float('inf')))

    # If user explicitly wants fast flights
    elif prefer_fast:
        return sorted(flights, key=lambda x: x.get('duration_min', float('inf')))

    # If user explicitly wants direct flights
    elif prefer_nonstop:
        return sorted(flights, key=lambda x: (x.get('stops', 999), x.get('price', float('inf'))))

    # Default: balanced approach
    else:
        def balanced_score(flight):
            # Weighted average of normalized factors
            price_weight = 0.4
            duration_weight = 0.3
            stop_weight = 0.3

            price_norm = flight.get('price', 1000) / 500
            duration_norm = flight.get('duration_min', 600) / 300
            stop_norm = flight.get('stops', 0)

            return (price_weight * price_norm +
                    duration_weight * duration_norm +
                    stop_weight * stop_norm)

        return sorted(flights, key=balanced_score)
