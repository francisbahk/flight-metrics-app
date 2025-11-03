"""
Built-in Flight Ranking Algorithms

These are the default algorithms available in the system.
"""
from typing import List, Dict
from backend.utils.listen_algorithms import ListenU, ListenT


def cheapest(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """
    Rank flights by price (lowest to highest).

    Simple algorithm: just sorts by price.
    """
    return sorted(flights, key=lambda x: x.get('price', float('inf')))


def fastest(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """
    Rank flights by duration (shortest to longest).

    Simple algorithm: sorts by total flight time.
    """
    return sorted(flights, key=lambda x: x.get('duration_min', float('inf')))


def fewest_stops(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """
    Rank flights by number of stops (direct flights first).

    Prioritizes non-stop flights, then 1 stop, etc.
    """
    return sorted(flights, key=lambda x: x.get('stops', float('inf')))


def best_value(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """
    Rank flights by value: price per hour of flight time.

    Balances cheap flights with reasonable duration.
    """
    def value_score(flight):
        price = flight.get('price', float('inf'))
        duration_hours = flight.get('duration_min', 1) / 60.0
        if duration_hours == 0:
            duration_hours = 1
        return price / duration_hours

    return sorted(flights, key=value_score)


def direct_then_cheap(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """
    Rank flights: direct flights first, then by price within each group.

    Two-tier ranking: non-stop flights at top, then sorted by price.
    """
    return sorted(flights, key=lambda x: (x.get('stops', 999), x.get('price', float('inf'))))


def listen_u_algorithm(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """
    LISTEN-U: Utility-based ranking with preference learning.

    Uses iterative refinement to learn user preferences.
    """
    # Extract preference utterance
    preference_text = preferences.get('original_query', 'Find the best flight')

    # Run LISTEN-U
    listen_u = ListenU()
    results = listen_u.rank_flights(flights, preference_text, max_iterations=3)

    return results['ranked_flights']


def listen_t_algorithm(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """
    LISTEN-T: Tournament-based selection.

    Uses batch-wise tournament to find best flights.
    """
    # Extract preference utterance
    preference_text = preferences.get('original_query', 'Find the best flight')

    # Run LISTEN-T
    listen_t = ListenT()
    results = listen_t.rank_flights(flights, preference_text, num_rounds=3, batch_size=4)

    return results['ranked_flights']


def preference_aware(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """
    Adaptive algorithm that changes strategy based on detected preferences.

    - If prefer_cheap → sort by price
    - If prefer_fast → sort by duration
    - If prefer_nonstop → sort by stops
    - Otherwise → balanced score
    """
    if preferences.get('prefer_cheap'):
        return sorted(flights, key=lambda x: x.get('price', float('inf')))
    elif preferences.get('prefer_fast'):
        return sorted(flights, key=lambda x: x.get('duration_min', float('inf')))
    elif preferences.get('prefer_nonstop'):
        return sorted(flights, key=lambda x: (x.get('stops', 999), x.get('price', float('inf'))))
    else:
        # Balanced: weighted combination of price and duration
        def balanced_score(flight):
            price_norm = flight.get('price', 500) / 500  # Normalize to ~0-1
            duration_norm = flight.get('duration_min', 300) / 600  # Normalize to ~0-1
            stops_penalty = flight.get('stops', 0) * 0.2
            return price_norm + duration_norm + stops_penalty

        return sorted(flights, key=balanced_score)


# Registry of built-in algorithms
BUILTIN_ALGORITHMS = {
    "Cheapest": {
        "function": cheapest,
        "description": "Sort by price (lowest first)",
        "category": "Simple"
    },
    "Fastest": {
        "function": fastest,
        "description": "Sort by duration (shortest first)",
        "category": "Simple"
    },
    "Fewest Stops": {
        "function": fewest_stops,
        "description": "Prioritize direct flights",
        "category": "Simple"
    },
    "Best Value": {
        "function": best_value,
        "description": "Balance price and duration (price per hour)",
        "category": "Simple"
    },
    "Direct Then Cheap": {
        "function": direct_then_cheap,
        "description": "Direct flights first, then sorted by price",
        "category": "Simple"
    },
    "LISTEN-U": {
        "function": listen_u_algorithm,
        "description": "Preference-based utility learning (parametric)",
        "category": "Advanced"
    },
    "LISTEN-T": {
        "function": listen_t_algorithm,
        "description": "Tournament-based selection (non-parametric)",
        "category": "Advanced"
    },
    "Preference Aware": {
        "function": preference_aware,
        "description": "Adapts to detected preferences automatically",
        "category": "Smart"
    }
}


def get_algorithm(name: str):
    """Get algorithm function by name."""
    if name in BUILTIN_ALGORITHMS:
        return BUILTIN_ALGORITHMS[name]["function"]
    return None


def list_algorithms():
    """List all available algorithms."""
    return list(BUILTIN_ALGORITHMS.keys())
