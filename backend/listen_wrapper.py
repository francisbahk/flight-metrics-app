"""
LISTEN-U Algorithm Wrapper for Flight Ranking
Maps Amadeus flight data to LISTEN features and runs dueling bandit optimization.
"""
import sys
import os
import numpy as np
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv

# Add LISTEN to path (now using submodule in flight_app/LISTEN)
LISTEN_PATH = os.path.join(os.path.dirname(__file__), '..', 'LISTEN')
sys.path.insert(0, LISTEN_PATH)

from LISTEN import BatchDuelingBanditOptimizer, BatchComparisonResult

load_dotenv()


def flights_to_listen_features(flights: List[Dict], user_preferences: Dict = None) -> np.ndarray:
    """
    Convert Amadeus flight data to LISTEN feature vectors.

    LISTEN expects these features (based on LISTEN repo):
    - price: cost of flight (normalized)
    - duration_min: flight duration in minutes (normalized)
    - stops: number of stops (0, 1, 2+)
    - departure_seconds: departure time as seconds since epoch (normalized)
    - arrival_seconds: arrival time as seconds since epoch (normalized)
    - dis_from_origin: distance from preferred origin (0 if exact match)
    - dis_from_dest: distance from preferred destination (0 if exact match)

    Amadeus provides:
    - price: float
    - duration_min: int (already calculated from ISO duration)
    - stops: int
    - departure_time: ISO datetime string
    - arrival_time: ISO datetime string
    - origin: airport code
    - destination: airport code
    - airline: carrier code

    Args:
        flights: List of Amadeus flight dictionaries
        user_preferences: Optional dict with preferred origins/destinations

    Returns:
        Feature matrix (n_flights x n_features)
    """
    if not flights:
        return np.array([])

    features_list = []

    # Get normalization values
    prices = [f['price'] for f in flights]
    durations = [f['duration_min'] for f in flights]

    max_price = max(prices) if prices else 1000
    max_duration = max(durations) if durations else 600

    # Parse preferred airports if provided
    preferred_origins = user_preferences.get('origins', []) if user_preferences else []
    preferred_dests = user_preferences.get('destinations', []) if user_preferences else []

    for flight in flights:
        # 1. Price (normalized 0-1)
        price_norm = flight['price'] / max_price if max_price > 0 else 0

        # 2. Duration (normalized 0-1)
        duration_norm = flight['duration_min'] / max_duration if max_duration > 0 else 0

        # 3. Stops (0, 1, 2+, cap at 2)
        stops = min(flight['stops'], 2)

        # 4. Departure time as seconds since epoch (normalized)
        try:
            dep_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
            dep_seconds = dep_dt.timestamp()
            # Normalize to 0-1 based on time of day (0 = midnight, 1 = 11:59pm)
            dep_hour = dep_dt.hour + dep_dt.minute / 60.0
            dep_norm = dep_hour / 24.0
        except:
            dep_norm = 0.5  # Default to noon

        # 5. Arrival time as seconds since epoch (normalized)
        try:
            arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
            arr_seconds = arr_dt.timestamp()
            arr_hour = arr_dt.hour + arr_dt.minute / 60.0
            arr_norm = arr_hour / 24.0
        except:
            arr_norm = 0.5

        # 6. Distance from preferred origin (binary: 0 if match, 1 if not)
        origin_match = 1.0 if (not preferred_origins or flight['origin'] in preferred_origins) else 0.0
        dis_from_origin = 1.0 - origin_match  # 0 = perfect match, 1 = no match

        # 7. Distance from preferred destination (binary: 0 if match, 1 if not)
        dest_match = 1.0 if (not preferred_dests or flight['destination'] in preferred_dests) else 0.0
        dis_from_dest = 1.0 - dest_match

        # Create feature vector (7 features)
        feature_vector = [
            price_norm,
            duration_norm,
            stops / 2.0,  # Normalize stops to 0-1
            dep_norm,
            arr_norm,
            dis_from_origin,
            dis_from_dest
        ]

        features_list.append(feature_vector)

    return np.array(features_list)


def rank_flights_with_listen_u(
    flights: List[Dict],
    user_prompt: str,
    user_preferences: Dict = None,
    n_iterations: int = 5,
    top_k: int = 10
) -> List[Dict]:
    """
    Rank flights using LISTEN-U dueling bandit algorithm.

    Args:
        flights: List of Amadeus flight dictionaries
        user_prompt: User's natural language query
        user_preferences: Dict with parsed preferences (origins, destinations, etc.)
        n_iterations: Number of LISTEN iterations (default 5 for speed)
        top_k: Number of top flights to return

    Returns:
        List of flights ranked by learned utility (best first)
    """
    if len(flights) < 2:
        return flights

    print(f"🤖 Running LISTEN-U on {len(flights)} flights...")

    # Convert flights to feature matrix
    features = flights_to_listen_features(flights, user_preferences)

    if features.shape[0] == 0:
        print("⚠️ No valid features extracted")
        return flights

    print(f"✓ Extracted {features.shape[0]} flight feature vectors with {features.shape[1]} features each")

    # Initialize LLM client for preference comparisons using Gemini
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not gemini_key or gemini_key == "your_gemini_key_here":
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY required for LISTEN-U. Please set it in .env file.")

    print("✓ Using Gemini LLM for flight comparisons")

    # Import our custom Gemini client
    from backend.gemini_llm_client import GeminiLLMPreferenceClient

    client = GeminiLLMPreferenceClient(
        api_key=gemini_key,
        model_name="gemini-2.0-flash",  # Fast, latest model
        simple=False,
        rate_limit_delay=0.2,  # Slightly slower for Gemini rate limits
        max_tokens=512,
        max_retries=10
    )

    # Initialize LISTEN-U optimizer (batch-based)
    optimizer = BatchDuelingBanditOptimizer(
        all_options=list(range(len(flights))),
        features=features,
        batch_size=min(4, len(flights)),  # Show 4 flights at a time
        acquisition="eubo",  # Expected utility-based optimization
        C=1.0,  # Regularization
        model_type="gp",  # Use Gaussian Process
        random_seed=42
    )

    print(f"✓ Initialized LISTEN-U optimizer (batch mode)")
    print(f"  - Batch size: {min(4, len(flights))}")
    print(f"  - Iterations: {n_iterations}")
    print(f"  - Model: Gaussian Process")

    # Run LISTEN iterations
    for iteration in range(n_iterations):
        print(f"  Iteration {iteration + 1}/{n_iterations}...")

        # Select next batch to compare
        batch_indices = optimizer.select_next_batch()

        # Build prompt with all flights in batch
        batch_flights_text = ""
        option_labels = []
        for i, flight_idx in enumerate(batch_indices):
            flight = flights[flight_idx]
            label = chr(65 + i)  # A, B, C, D...
            option_labels.append(label)
            batch_flights_text += f"""
Flight {label}: {flight['airline']}{flight['flight_number']}
- Price: ${flight['price']:.0f}
- Duration: {flight['duration_min']} min ({flight['duration_min']//60}h {flight['duration_min']%60}m)
- Stops: {flight['stops']}
- Departs: {flight['departure_time']}
- Arrives: {flight['arrival_time']}
"""

        comparison_text = f"""You are helping a user choose the BEST flight from these options.

User's original request: {user_prompt}

{batch_flights_text}

Based on the user's request, which flight is the BEST choice?
Respond with ONLY the letter of the best flight: "FINAL: A" or "FINAL: B" or "FINAL: C" or "FINAL: D"
"""

        # Get best flight from Gemini LLM
        try:
            choice, response_text = client.call_oracle(comparison_text)
            # Map letter back to index
            choice_idx = ord(choice) - 65
            if 0 <= choice_idx < len(batch_indices):
                winner_idx = batch_indices[choice_idx]
                print(f"    Batch comparison: Flight {choice} wins (Gemini choice)")
            else:
                # Fallback if invalid choice
                winner_idx = batch_indices[0]
                print(f"    Warning: Invalid choice '{choice}', using first option")
        except Exception as e:
            print(f"    Warning: LLM comparison failed ({str(e)}), using feature heuristic")
            # Fallback: pick the one with best feature scores
            batch_scores = []
            for flight_idx in batch_indices:
                score = -features[flight_idx, 0] - features[flight_idx, 1] - features[flight_idx, 2]
                batch_scores.append(score)
            winner_idx = batch_indices[np.argmax(batch_scores)]

        # Record batch comparison result
        batch_features = features[batch_indices]
        winner_features = features[winner_idx]

        # Calculate feature deltas (winner - losers)
        feature_deltas = []
        for flight_idx in batch_indices:
            if flight_idx != winner_idx:
                delta = winner_features - features[flight_idx]
                feature_deltas.append(delta)
        feature_deltas = np.array(feature_deltas) if feature_deltas else np.array([[]])

        result = BatchComparisonResult(
            batch_indices=batch_indices,
            winner_idx=winner_idx,
            features_batch=batch_features,
            feature_deltas=feature_deltas
        )
        optimizer.comparison_history.append(result)

        # Update appearance and win counts
        for idx in batch_indices:
            optimizer.appearance_counts[idx] += 1
        optimizer.win_counts[winner_idx] += 1

        # Train model
        if len(optimizer.comparison_history) >= optimizer.min_appearances_for_model:
            optimizer.model.fit(optimizer.comparison_history)

    print("✓ LISTEN-U optimization complete!")

    # Get final utilities for all flights
    if optimizer.model.ready():
        utilities = optimizer.model.posterior_mean(features)
        print(f"✓ Learned utility scores (mean: {utilities.mean():.3f}, std: {utilities.std():.3f})")
    else:
        print("⚠️ Model not ready, using win rate ranking")
        utilities = optimizer.win_counts / (optimizer.appearance_counts + 1e-6)

    # Rank flights by utility (highest first)
    ranked_indices = np.argsort(utilities)[::-1]

    # Return top-k flights
    ranked_flights = [flights[i] for i in ranked_indices[:top_k]]

    print(f"✓ Returning top {len(ranked_flights)} flights")

    return ranked_flights


# For testing
if __name__ == "__main__":
    # Test with sample flights
    test_flights = [
        {
            'id': '1',
            'price': 299.0,
            'duration_min': 360,
            'stops': 0,
            'departure_time': '2024-11-15T08:00:00',
            'arrival_time': '2024-11-15T14:00:00',
            'airline': 'AA',
            'flight_number': '123',
            'origin': 'JFK',
            'destination': 'LAX'
        },
        {
            'id': '2',
            'price': 450.0,
            'duration_min': 240,
            'stops': 1,
            'departure_time': '2024-11-15T10:00:00',
            'arrival_time': '2024-11-15T14:00:00',
            'airline': 'UA',
            'flight_number': '456',
            'origin': 'JFK',
            'destination': 'LAX'
        }
    ]

    preferences = {
        'origins': ['JFK'],
        'destinations': ['LAX']
    }

    ranked = rank_flights_with_listen_u(
        test_flights,
        "I want a cheap direct flight",
        preferences,
        n_iterations=3,
        top_k=10
    )

    print("\nRanked flights:")
    for i, flight in enumerate(ranked, 1):
        print(f"{i}. {flight['airline']}{flight['flight_number']} - ${flight['price']:.0f}")
