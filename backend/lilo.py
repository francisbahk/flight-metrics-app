"""
LILO (Language-based Interactive Learning with Optimization)
Iterative preference learning using Gemini LLM for utility estimation.
"""
from typing import List, Dict, Optional
import random
from backend.gemini_llm_client import GeminiLLMPreferenceClient


class LILOOptimizer:
    """
    LILO optimizer that learns user preferences through iterative feedback.

    Uses Gemini to:
    1. Generate preference questions (Prompt 1)
    2. Process feedback and estimate utilities (Prompt 2)
    3. Rank flights by learned utility function
    """

    def __init__(self):
        self.client = GeminiLLMPreferenceClient()
        self.all_flights = []
        self.feedback_history = []
        self.initial_answers = {}
        self.feedback_summary = ""

    def generate_initial_questions(self, user_prompt: str) -> List[str]:
        """
        Generate initial preference questions based on user's prompt.

        Args:
            user_prompt: User's natural language flight search query

        Returns:
            List of preference questions
        """
        prompt = f"""Based on this flight search query: "{user_prompt}"

Generate 5 specific questions to understand the user's preferences. Focus on:
1. Price vs. time trade-offs
2. Preferred departure/arrival times
3. Willingness to take connecting flights
4. Airline preferences
5. Other important factors (refundability, baggage, etc.)

Return ONLY the questions, one per line, numbered 1-5."""

        try:
            response = self.client.generate_response(
                prompt,
                temperature=0.7,
                max_new_tokens=500
            )

            # Parse questions from response
            questions = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering
                    question = line.lstrip('0123456789.-•) ').strip()
                    if question and '?' in question:
                        questions.append(question)

            # Fallback questions if parsing fails
            if len(questions) < 3:
                questions = [
                    "What is most important to you: price, travel time, or number of stops?",
                    "Do you prefer morning, afternoon, or evening departures?",
                    "Are you willing to pay more for a nonstop flight?",
                    "How important is the airline carrier to you?",
                    "Do you prefer to arrive as early as possible, or is arrival time flexible?"
                ]

            return questions[:5]

        except Exception as e:
            print(f"  ⚠️ Error generating questions: {str(e)}")
            # Return default questions
            return [
                "What is most important to you: price, travel time, or number of stops?",
                "Do you prefer morning, afternoon, or evening departures?",
                "Are you willing to pay more for a nonstop flight?",
                "How important is the airline carrier to you?",
                "Do you prefer to arrive as early as possible, or is arrival time flexible?"
            ]

    def generate_feedback_questions(self, flights_shown: List[Dict]) -> List[str]:
        """
        Generate feedback questions after showing flights to user.

        Args:
            flights_shown: List of flights that were shown to user

        Returns:
            List of feedback questions
        """
        return [
            "Which of these flights best matches your preferences? (Copy flight info to reference)",
            "What aspects of the shown flights are most appealing to you?",
            "Are there any dealbreakers in the flights you see?",
            "How would you rank price vs. convenience for this trip?"
        ]

    def select_candidates(
        self,
        all_flights: List[Dict],
        round_num: int,
        n_candidates: int = 15
    ) -> List[Dict]:
        """
        Select candidate flights for a round.

        Round 1: Random diverse flights
        Round 2+: Top flights by estimated utility

        Args:
            all_flights: All available flights
            round_num: Current round number (1-based)
            n_candidates: Number of flights to show

        Returns:
            List of selected flights
        """
        if round_num == 1:
            # Round 1: Random sampling for diversity
            if len(all_flights) <= n_candidates:
                return all_flights
            return random.sample(all_flights, n_candidates)
        else:
            # Round 2+: Utility-based ranking
            utilities = self.estimate_utilities(all_flights)

            # Sort by utility (descending)
            sorted_flights = sorted(
                zip(all_flights, utilities),
                key=lambda x: x[1],
                reverse=True
            )

            return [f for f, _ in sorted_flights[:n_candidates]]

    def estimate_utilities(self, flights: List[Dict]) -> List[float]:
        """
        Estimate utility scores for flights using Gemini based on feedback history.

        Args:
            flights: List of flights to score

        Returns:
            List of utility scores (higher is better)
        """
        if not self.feedback_history:
            # No feedback yet - return neutral scores
            return [0.5] * len(flights)

        # Build preference summary from all feedback
        feedback_text = "\n\n".join([
            f"Round {i+1} Feedback: {fb}"
            for i, fb in enumerate(self.feedback_history)
        ])

        if self.initial_answers:
            initial_text = "\n".join([
                f"Q{k}: {v}"
                for k, v in self.initial_answers.items()
            ])
            feedback_text = f"Initial Preferences:\n{initial_text}\n\n{feedback_text}"

        # Score each flight using Gemini
        utilities = []

        for flight in flights:
            flight_desc = self._format_flight_for_llm(flight)

            prompt = f"""Given these user preferences:

{feedback_text}

Rate this flight on a scale of 0-100 based on how well it matches the user's preferences:

{flight_desc}

Consider:
- Price preferences (expensive vs. cheap)
- Time preferences (departure/arrival times)
- Duration and stops
- Airline preferences
- Any dealbreakers mentioned

Return ONLY a number between 0-100. Higher means better match."""

            try:
                response = self.client.generate_response(
                    prompt,
                    temperature=0.3,
                    max_new_tokens=50
                )

                # Extract number from response
                import re
                numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
                if numbers:
                    score = float(numbers[0])
                    # Normalize to 0-1
                    utilities.append(score / 100.0)
                else:
                    utilities.append(0.5)  # Neutral score

            except Exception as e:
                print(f"  ⚠️ Error estimating utility: {str(e)}")
                utilities.append(0.5)

        return utilities

    def run_round(
        self,
        all_flights: List[Dict],
        round_num: int,
        user_rankings: List[int],
        user_feedback: str,
        n_candidates: int = 15
    ) -> Dict:
        """
        Run a LILO round: process feedback and select next flights.

        Args:
            all_flights: All available flights
            round_num: Current round number
            user_rankings: Indices of flights user ranked (in order)
            user_feedback: Natural language feedback
            n_candidates: Number of flights for next round

        Returns:
            Dict with 'flights_shown', 'questions', 'feedback_summary'
        """
        # Store feedback
        self.feedback_history.append(user_feedback)
        self.feedback_summary = f"Collected {len(self.feedback_history)} rounds of feedback"

        # Select flights for next round
        next_flights = self.select_candidates(all_flights, round_num + 1, n_candidates)

        # Generate questions for next round
        questions = self.generate_feedback_questions(next_flights)

        return {
            'flights_shown': next_flights,
            'questions': questions,
            'feedback_summary': self.feedback_summary
        }

    def get_final_ranking(self, all_flights: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Get final ranking based on all feedback.

        Args:
            all_flights: All available flights
            top_k: Number of top flights to return

        Returns:
            List of top-k flights ranked by utility
        """
        utilities = self.estimate_utilities(all_flights)

        # Sort by utility (descending)
        sorted_flights = sorted(
            zip(all_flights, utilities),
            key=lambda x: x[1],
            reverse=True
        )

        return [f for f, _ in sorted_flights[:top_k]]

    def _format_flight_for_llm(self, flight: Dict) -> str:
        """Format flight data for LLM prompt."""
        return f"""Flight: {flight.get('carrier', 'Unknown')} {flight.get('flight_number', '')}
Route: {flight.get('origin', '')} → {flight.get('destination', '')}
Departure: {flight.get('departure_time', 'N/A')}
Arrival: {flight.get('arrival_time', 'N/A')}
Duration: {flight.get('duration_minutes', 0)} minutes
Stops: {flight.get('stops', 0)}
Price: ${flight.get('price', 0)}
"""