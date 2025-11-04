"""
Example Algorithm: Balanced Score
This algorithm balances price, duration, and stops with a weighted score.
"""

def rank_flights(flights, preferences):
    """
    Rank flights using a balanced score considering multiple factors.

    Scoring formula:
    - Lower price = better (normalized)
    - Shorter duration = better (normalized)
    - Fewer stops = better (penalty per stop)

    Args:
        flights: List of flight dictionaries
        preferences: Dictionary with user preferences

    Returns:
        List of flights sorted by score (lowest = best)
    """
    def calculate_score(flight):
        # Normalize price (assume average flight is $500)
        price_score = flight.get('price', 1000) / 500

        # Normalize duration (assume average is 5 hours = 300 minutes)
        duration_score = flight.get('duration_min', 600) / 300

        # Penalty for stops (each stop adds 0.5 to score)
        stop_penalty = flight.get('stops', 0) * 0.5

        # Combined score (lower is better)
        total_score = price_score + duration_score + stop_penalty

        return total_score

    # Sort by score (lowest first)
    return sorted(flights, key=calculate_score)
