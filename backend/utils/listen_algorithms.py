"""
LISTEN Algorithm Implementations:
- LISTEN-U: Utility Refinement (parametric, iterative linear utility model)
- LISTEN-T: Tournament Selection (non-parametric, batchwise tournament)
"""
import numpy as np
from typing import List, Dict, Any, Tuple
import json


class ListenU:
    """
    LISTEN-U: Utility Refinement Algorithm

    Assumes a linear utility function over numerical flight attributes.
    LLM iteratively refines the weight vector on these attributes.
    """

    def __init__(self, llm_client=None):
        """
        Initialize LISTEN-U algorithm.

        Args:
            llm_client: Client for LLM API calls (e.g., OpenAI, Anthropic)
        """
        self.llm_client = llm_client
        self.numerical_attributes = [
            "price", "duration_min", "stops",
            "dis_from_origin", "dis_from_dest",
            "departure_seconds", "arrival_seconds"
        ]

    def normalize_attributes(self, flights: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Normalize numerical attributes to [0, 1] range.

        Returns:
            Tuple of (normalized_flights, normalization_params)
        """
        norm_params = {}
        normalized_flights = [flight.copy() for flight in flights]

        for attr in self.numerical_attributes:
            values = [f.get(attr, 0) for f in flights if f.get(attr) is not None]
            if not values:
                continue

            min_val = min(values)
            max_val = max(values)

            norm_params[attr] = {'min': min_val, 'max': max_val}

            # Normalize
            if max_val - min_val > 0:
                for flight in normalized_flights:
                    if flight.get(attr) is not None:
                        flight[f"{attr}_normalized"] = (flight[attr] - min_val) / (max_val - min_val)
                    else:
                        flight[f"{attr}_normalized"] = 0.5
            else:
                for flight in normalized_flights:
                    flight[f"{attr}_normalized"] = 0.5

        return normalized_flights, norm_params

    def format_for_llm(self, flights: List[Dict], preference_utterance: str) -> str:
        """
        Format flight data for LLM consumption.

        Args:
            flights: List of flight dictionaries
            preference_utterance: User's natural language preferences

        Returns:
            Formatted JSON string for LLM
        """
        formatted_data = {
            "preference_utterance": preference_utterance,
            "metrics": {
                "price": "Ticket price in USD",
                "duration_min": "Total flight duration in minutes",
                "stops": "Number of stops/layovers",
                "departure_seconds": "Seconds from midnight for departure time",
                "arrival_seconds": "Seconds from midnight for arrival time",
                "dis_from_origin": "Distance from reference point to origin (km)",
                "dis_from_dest": "Distance from reference point to destination (km)"
            },
            "numerical_attributes": self.numerical_attributes,
            "flights": []
        }

        for i, flight in enumerate(flights):
            formatted_flight = {
                "flight_id": f"F{str(i+1).zfill(3)}",
                "original_id": flight.get("id"),
                "airline": flight.get("name", "Unknown"),
                "price": flight.get("price", 0),
                "duration_min": flight.get("duration_min", 0),
                "stops": flight.get("stops", 0),
                "departure_time": flight.get("departure_time", ""),
                "arrival_time": flight.get("arrival_time", ""),
                "departure_seconds": flight.get("departure_seconds", 0),
                "arrival_seconds": flight.get("arrival_seconds", 0),
                "dis_from_origin": flight.get("dis_from_origin", 0),
                "dis_from_dest": flight.get("dis_from_dest", 0),
                "origin": flight.get("origin", ""),
                "destination": flight.get("destination", ""),
            }
            formatted_data["flights"].append(formatted_flight)

        return json.dumps(formatted_data, indent=2)

    def get_initial_weights_prompt(self, formatted_data: str) -> str:
        """Generate prompt for initial weight vector."""
        return f"""You are a flight ranking assistant. Based on the user's preferences and the flight attributes,
propose an initial weight vector for the numerical attributes.

{formatted_data}

Please analyze the preference utterance and return ONLY a JSON object with normalized weights (sum to 1.0) for each numerical attribute.
Lower is better for: price, duration_min, stops, dis_from_origin, dis_from_dest
Time preferences depend on context (departure_seconds, arrival_seconds)

Return format:
{{
  "weights": {{
    "price": 0.3,
    "duration_min": 0.25,
    "stops": 0.2,
    "dis_from_origin": 0.05,
    "dis_from_dest": 0.05,
    "departure_seconds": 0.075,
    "arrival_seconds": 0.075
  }},
  "reasoning": "Brief explanation of weight choices"
}}"""

    def compute_utility_scores(self, flights: List[Dict], weights: Dict[str, float]) -> List[float]:
        """
        Compute utility score for each flight using weight vector.

        For attributes where lower is better (price, duration, stops, distances),
        we use (1 - normalized_value) so higher score is better.
        """
        scores = []

        for flight in flights:
            score = 0.0

            # Attributes where lower is better - invert them
            for attr in ["price", "duration_min", "stops", "dis_from_origin", "dis_from_dest"]:
                norm_attr = f"{attr}_normalized"
                if norm_attr in flight and attr in weights:
                    # Invert: lower values get higher scores
                    score += weights[attr] * (1 - flight[norm_attr])

            # Time attributes - use as-is (depends on preference)
            for attr in ["departure_seconds", "arrival_seconds"]:
                norm_attr = f"{attr}_normalized"
                if norm_attr in flight and attr in weights:
                    score += weights[attr] * flight[norm_attr]

            scores.append(score)

        return scores

    def get_refinement_prompt(self, current_weights: Dict, best_flight: Dict,
                             original_preference: str, iteration: int) -> str:
        """Generate prompt for weight refinement."""
        return f"""You are refining a flight ranking utility function.

Original preference: "{original_preference}"

Iteration: {iteration}

Current weights: {json.dumps(current_weights, indent=2)}

Current best flight:
{json.dumps(best_flight, indent=2)}

Please critique this selection and propose refined weights. Consider:
1. Does this flight match the user's preferences?
2. Are any attributes over/under-weighted?
3. What adjustments would better capture the preference?

Return format:
{{
  "weights": {{
    "price": <value>,
    "duration_min": <value>,
    "stops": <value>,
    "dis_from_origin": <value>,
    "dis_from_dest": <value>,
    "departure_seconds": <value>,
    "arrival_seconds": <value>
  }},
  "critique": "Analysis of current selection",
  "changes": "What was adjusted and why"
}}

Weights must sum to 1.0."""

    def rank_flights(self, flights: List[Dict], preference_utterance: str,
                    max_iterations: int = 3) -> Dict:
        """
        Main LISTEN-U algorithm: iteratively refine weights and rank flights.

        Args:
            flights: List of flight data
            preference_utterance: User's natural language preferences
            max_iterations: Number of refinement iterations

        Returns:
            Dictionary with final weights, best flight, and rankings
        """
        # Normalize attributes
        normalized_flights, norm_params = self.normalize_attributes(flights)

        # Format for LLM
        formatted_data = self.format_for_llm(flights, preference_utterance)

        # Initialize weights (if no LLM, use default)
        weights = {
            "price": 0.30,
            "duration_min": 0.25,
            "stops": 0.20,
            "dis_from_origin": 0.05,
            "dis_from_dest": 0.05,
            "departure_seconds": 0.075,
            "arrival_seconds": 0.075
        }

        iterations_log = []

        # Iterative refinement
        for iteration in range(max_iterations):
            # Compute scores
            scores = self.compute_utility_scores(normalized_flights, weights)

            # Find best flight
            best_idx = np.argmax(scores)
            best_flight = flights[best_idx].copy()
            best_flight['utility_score'] = scores[best_idx]

            iterations_log.append({
                "iteration": iteration + 1,
                "weights": weights.copy(),
                "best_flight_id": best_flight.get("id"),
                "best_score": scores[best_idx],
                "all_scores": scores
            })

            # Note: In production, you would call LLM here to refine weights
            # For now, we'll use the initial weights
            # weights = self.call_llm_for_refinement(...)

        # Final ranking
        final_scores = self.compute_utility_scores(normalized_flights, weights)
        ranked_indices = np.argsort(final_scores)[::-1]  # Descending order

        ranked_flights = []
        for rank, idx in enumerate(ranked_indices):
            flight_copy = flights[idx].copy()
            flight_copy['rank'] = rank + 1
            flight_copy['utility_score'] = final_scores[idx]
            ranked_flights.append(flight_copy)

        return {
            "algorithm": "LISTEN-U",
            "final_weights": weights,
            "normalization_params": norm_params,
            "iterations": iterations_log,
            "best_flight": ranked_flights[0],
            "ranked_flights": ranked_flights,
            "preference_utterance": preference_utterance
        }


class ListenT:
    """
    LISTEN-T: Tournament Selection Algorithm

    Samples random batches of flights and asks LLM to pick a batch champion,
    then runs a final playoff among champions.
    """

    def __init__(self, llm_client=None):
        """
        Initialize LISTEN-T algorithm.

        Args:
            llm_client: Client for LLM API calls (e.g., OpenAI, Anthropic)
        """
        self.llm_client = llm_client

    def format_batch_for_llm(self, flights: List[Dict], preference_utterance: str,
                            round_num: int) -> str:
        """Format a batch of flights for LLM tournament round."""
        formatted = {
            "round": round_num,
            "preference_utterance": preference_utterance,
            "task": "Select the single flight that best matches the user's preferences",
            "flights": []
        }

        for i, flight in enumerate(flights):
            formatted["flights"].append({
                "option": chr(65 + i),  # A, B, C, D...
                "flight_id": flight.get("id"),
                "airline": flight.get("name", "Unknown"),
                "price": f"${flight.get('price', 0):.2f}",
                "duration": f"{flight.get('duration_min', 0):.0f} min",
                "stops": flight.get("stops", 0),
                "departure": flight.get("departure_time", ""),
                "arrival": flight.get("arrival_time", ""),
                "route": f"{flight.get('origin', '')} → {flight.get('destination', '')}"
            })

        return json.dumps(formatted, indent=2)

    def get_tournament_prompt(self, formatted_batch: str) -> str:
        """Generate prompt for tournament round."""
        return f"""You are a flight selection assistant conducting a tournament to find the best flight option.

{formatted_batch}

Analyze each flight option and select the ONE that best matches the user's preference utterance.

Return format:
{{
  "selected_option": "A",
  "flight_id": <id>,
  "reasoning": "Why this flight best matches the preferences"
}}"""

    def sample_batch(self, flights: List[Dict], batch_size: int,
                    exclude_ids: List[int] = None) -> List[Dict]:
        """Randomly sample a batch of flights."""
        if exclude_ids is None:
            exclude_ids = []

        available_flights = [f for f in flights if f.get("id") not in exclude_ids]

        if len(available_flights) <= batch_size:
            return available_flights

        indices = np.random.choice(len(available_flights), size=batch_size, replace=False)
        return [available_flights[i] for i in indices]

    def rank_flights(self, flights: List[Dict], preference_utterance: str,
                    num_rounds: int = 3, batch_size: int = 4) -> Dict:
        """
        Main LISTEN-T algorithm: tournament selection.

        Args:
            flights: List of flight data
            preference_utterance: User's natural language preferences
            num_rounds: Number of preliminary rounds (T-1)
            batch_size: Number of flights per batch

        Returns:
            Dictionary with tournament results and winner
        """
        champions = []
        tournament_log = []

        # Preliminary rounds
        for round_num in range(1, num_rounds + 1):
            # Sample batch
            batch = self.sample_batch(
                flights,
                batch_size,
                exclude_ids=[c.get("id") for c in champions]
            )

            if not batch:
                break

            # Format for LLM
            formatted_batch = self.format_batch_for_llm(batch, preference_utterance, round_num)

            # In production, call LLM here
            # For now, select first flight as champion (placeholder)
            champion = batch[0].copy()
            champion['tournament_round'] = round_num
            champions.append(champion)

            tournament_log.append({
                "round": round_num,
                "batch_size": len(batch),
                "batch_flight_ids": [f.get("id") for f in batch],
                "champion_id": champion.get("id"),
                "reasoning": "Placeholder: LLM selection would go here"
            })

        # Final playoff
        if len(champions) > 1:
            final_batch = self.format_batch_for_llm(champions, preference_utterance, "FINAL")
            # In production, call LLM for final selection
            winner = champions[0]  # Placeholder

            tournament_log.append({
                "round": "FINAL",
                "batch_size": len(champions),
                "batch_flight_ids": [f.get("id") for f in champions],
                "winner_id": winner.get("id"),
                "reasoning": "Placeholder: Final LLM selection would go here"
            })
        else:
            winner = champions[0] if champions else flights[0]

        # Rank all flights (winner first, then champions, then rest)
        champion_ids = {c.get("id") for c in champions}
        ranked_flights = [winner]

        for champ in champions:
            if champ.get("id") != winner.get("id"):
                ranked_flights.append(champ)

        for flight in flights:
            if flight.get("id") not in champion_ids:
                ranked_flights.append(flight)

        # Add rank numbers
        for i, flight in enumerate(ranked_flights):
            flight['rank'] = i + 1

        return {
            "algorithm": "LISTEN-T",
            "tournament_rounds": num_rounds,
            "batch_size": batch_size,
            "tournament_log": tournament_log,
            "winner": winner,
            "champions": champions,
            "ranked_flights": ranked_flights,
            "preference_utterance": preference_utterance
        }
