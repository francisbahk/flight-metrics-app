"""
LILO Optimizer - Bayesian Optimization with Interactive Natural Language Feedback

Implements the LILO algorithm from https://arxiv.org/pdf/2510.17671
"""
import json
import os
from typing import List, Dict, Optional
import google.generativeai as genai
from .prompts import (
    get_question_generation_prompt,
    get_utility_estimation_prompt,
    get_feedback_summarization_prompt
)


class LILOOptimizer:
    """
    LILO (Bayesian Optimization with Interactive Natural Language Feedback) implementation.

    Workflow:
    1. Show user N candidate flights
    2. User ranks top-k and provides natural language feedback
    3. LLM extracts preferences and updates utility model
    4. Repeat for multiple rounds
    5. Return final ranking using learned utility
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize LILO optimizer.

        Args:
            gemini_api_key: Gemini API key (or uses GEMINI_API_KEY env var)
        """
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # LILO state
        self.all_flights = []
        self.round_history = []  # List of dicts: {round_num, flights_shown, rankings, feedback}
        self.feedback_summary = ""

    def format_flights_for_prompt(self, flights: List[Dict], indices: Optional[List[int]] = None) -> str:
        """
        Format flights as structured text for LLM prompts.

        Args:
            flights: List of flight dictionaries
            indices: Optional list of specific indices to include

        Returns:
            Formatted string describing flights
        """
        if indices is not None:
            flights = [flights[i] for i in indices]

        formatted = []
        for i, flight in enumerate(flights):
            arm_index = f"arm_{i}"
            formatted.append(
                f"Outcome {arm_index}:\n"
                f"  Airline: {flight['airline']}{flight.get('flight_number', '')}\n"
                f"  Route: {flight['origin']} → {flight['destination']}\n"
                f"  Price: ${flight['price']:.0f}\n"
                f"  Duration: {flight['duration_min']//60}h {flight['duration_min']%60}m\n"
                f"  Stops: {flight['stops']}\n"
                f"  Departure: {flight['departure_time']}\n"
                f"  Arrival: {flight['arrival_time']}\n"
            )

        return "\n".join(formatted)

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with error handling."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return ""

    def _extract_json_from_response(self, text: str) -> Dict:
        """Extract JSON from LLM response (handles ```json markers)."""
        # Remove markdown code blocks
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw text: {text[:500]}")
            return {}

    def _extract_jsonl_from_response(self, text: str) -> List[Dict]:
        """Extract JSONL from LLM response (Prompt 4 returns JSONL)."""
        # Remove markdown code blocks
        text = text.strip()
        if "```jsonl" in text:
            text = text.split("```jsonl")[1].split("```")[0]
        elif "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        results = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return results

    def generate_questions(self, flights_shown: List[Dict], user_rankings: List[int],
                          round_num: int, n_questions: int = 3) -> List[str]:
        """
        Prompt 2: Generate questions to ask user after they rank flights.

        Args:
            flights_shown: Flights shown this round
            user_rankings: Indices of flights user ranked (top-k)
            round_num: Current round number
            n_questions: Number of questions to generate

        Returns:
            List of questions to ask user
        """
        experiment_data = self.format_flights_for_prompt(flights_shown)

        # Build human feedback string from previous rounds
        human_feedback = ""
        if self.round_history:
            for r in self.round_history:
                human_feedback += f"Round {r['round_num']}: {r['feedback']}\n"

        selected_indices = ", ".join([f"arm_{i}" for i in user_rankings])

        prompt = get_question_generation_prompt(
            experiment_data=experiment_data,
            human_feedback=human_feedback or "No previous feedback yet.",
            selected_outcome_indices=selected_indices,
            n_questions=n_questions
        )

        response = self._call_gemini(prompt)
        questions_dict = self._extract_json_from_response(response)

        # Extract questions from dict {"q1": "...", "q2": "...", ...}
        questions = [questions_dict.get(f"q{i+1}", "") for i in range(n_questions)]
        return [q for q in questions if q]  # Filter empty

    def summarize_feedback(self, flights_shown: List[Dict]) -> str:
        """
        Prompt 5: Summarize all user feedback into optimization goals.

        Args:
            flights_shown: All flights shown so far

        Returns:
            Summary string
        """
        experiment_data = self.format_flights_for_prompt(flights_shown)

        # Collect all feedback
        human_feedback = ""
        for r in self.round_history:
            human_feedback += f"Round {r['round_num']}:\n"
            human_feedback += f"User ranked: {r['rankings']}\n"
            human_feedback += f"Feedback: {r['feedback']}\n\n"

        if not human_feedback:
            return ""

        prompt = get_feedback_summarization_prompt(
            experiment_data=experiment_data,
            human_feedback=human_feedback
        )

        response = self._call_gemini(prompt)
        summary_dict = self._extract_json_from_response(response)

        return summary_dict.get("summary", "")

    def estimate_utilities(self, flights: List[Dict]) -> List[float]:
        """
        Prompt 4: Estimate utility (p_accept) for each flight.

        Args:
            flights: All flights to estimate utilities for

        Returns:
            List of utility scores (0-1) for each flight
        """
        experiment_data = self.format_flights_for_prompt(flights)

        # Collect all feedback
        human_feedback = ""
        for r in self.round_history:
            human_feedback += f"Round {r['round_num']}:\n"
            human_feedback += f"User ranked: {r['rankings']}\n"
            human_feedback += f"Feedback: {r['feedback']}\n\n"

        if not human_feedback:
            human_feedback = "No feedback yet."

        # Include summary if available
        summary_text = f"\n## Feedback Summary:\n{self.feedback_summary}" if self.feedback_summary else ""

        prompt = get_utility_estimation_prompt(
            experiment_data=experiment_data,
            human_feedback=human_feedback,
            human_feedback_summary=summary_text
        )

        response = self._call_gemini(prompt)
        utility_list = self._extract_jsonl_from_response(response)

        # Extract p_accept scores in order
        utilities = []
        for i in range(len(flights)):
            arm_id = f"arm_{i}"
            # Find matching entry
            entry = next((u for u in utility_list if u.get("arm_index") == arm_id), None)
            if entry and "p_accept" in entry:
                utilities.append(float(entry["p_accept"]))
            else:
                utilities.append(0.5)  # Default neutral score

        return utilities

    def select_candidates(self, all_flights: List[Dict], round_num: int,
                         n_candidates: int = 15) -> List[Dict]:
        """
        Select N candidate flights to show user in this round.

        Round 1: Random/diverse selection
        Round 2+: Use utility estimates to select promising candidates

        Args:
            all_flights: All available flights
            round_num: Current round number
            n_candidates: Number of flights to show

        Returns:
            List of selected flight dictionaries
        """
        if round_num == 1:
            # Round 1: Show diverse set (top by price, duration, mix)
            cheapest = sorted(all_flights, key=lambda x: x['price'])[:5]
            fastest = sorted(all_flights, key=lambda x: x['duration_min'])[:5]
            # Random remaining
            import random
            remaining = [f for f in all_flights if f not in cheapest and f not in fastest]
            random.shuffle(remaining)
            candidates = cheapest + fastest + remaining[:n_candidates-10]
            return candidates[:n_candidates]
        else:
            # Round 2+: Use utility estimates
            utilities = self.estimate_utilities(all_flights)

            # Pair flights with utilities
            flights_with_utility = list(zip(all_flights, utilities))

            # Sort by utility (highest first)
            flights_with_utility.sort(key=lambda x: x[1], reverse=True)

            # Take top N
            return [f for f, u in flights_with_utility[:n_candidates]]

    def run_round(self, all_flights: List[Dict], round_num: int,
                 user_rankings: List[int], user_feedback: str,
                 n_candidates: int = 15) -> Dict:
        """
        Run one LILO round.

        Args:
            all_flights: All available flights
            round_num: Current round number
            user_rankings: Indices of flights user ranked (in order of preference)
            user_feedback: User's natural language feedback
            n_candidates: Number of flights to show

        Returns:
            Dict with round results
        """
        # Select candidate flights for this round
        flights_shown = self.select_candidates(all_flights, round_num, n_candidates)

        # Store round data
        round_data = {
            'round_num': round_num,
            'flights_shown': flights_shown,
            'rankings': user_rankings,
            'feedback': user_feedback
        }
        self.round_history.append(round_data)

        # Generate questions for next round (if not final round)
        questions = []
        if round_num < 2:  # 2 rounds total
            questions = self.generate_questions(flights_shown, user_rankings, round_num)

        # Summarize feedback after round 1
        if round_num == 1:
            self.feedback_summary = self.summarize_feedback(flights_shown)

        return {
            'flights_shown': flights_shown,
            'questions': questions,
            'feedback_summary': self.feedback_summary
        }

    def get_final_ranking(self, all_flights: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Get final ranking of all flights using learned utility function.

        Args:
            all_flights: All available flights
            top_k: Number of top flights to return

        Returns:
            List of top-k flights ranked by learned utility
        """
        # Estimate utilities for all flights
        utilities = self.estimate_utilities(all_flights)

        # Pair and sort
        flights_with_utility = list(zip(all_flights, utilities))
        flights_with_utility.sort(key=lambda x: x[1], reverse=True)

        return [f for f, u in flights_with_utility[:top_k]]