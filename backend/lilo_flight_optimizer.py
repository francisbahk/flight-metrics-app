"""
LILO Flight Optimizer - Adapted from language_bo_code for flight preferences
Uses LLM-based utility approximation from the LILO paper (arxiv.org/pdf/2510.17671)
"""
import os
import json
import random
from typing import List, Dict, Optional
import google.generativeai as genai


class LILOFlightOptimizer:
    """
    LILO optimizer for flight preferences using language feedback.

    Implements the LILO algorithm:
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
        self.round_history = []  # List of dicts: {round_num, flights_shown, rankings, feedback, answers}
        self.feedback_summary = ""

    def format_flight_for_prompt(self, flight: Dict, arm_index: str) -> str:
        """
        Format a single flight for LLM prompt.

        Args:
            flight: Flight dictionary
            arm_index: Identifier like "arm_0", "arm_1", etc.

        Returns:
            Formatted string describing the flight
        """
        return f"""Outcome {arm_index}:
  Airline: {flight.get('airline', 'Unknown')}
  Route: {flight.get('origin', '')} → {flight.get('destination', '')}
  Price: ${flight.get('price', 0):.0f}
  Duration: {flight.get('duration_min', 0)//60}h {flight.get('duration_min', 0)%60}m
  Stops: {flight.get('stops', 0)}
  Departure: {flight.get('departure_time', 'N/A')}
  Arrival: {flight.get('arrival_time', 'N/A')}
"""

    def format_flights_for_prompt(self, flights: List[Dict]) -> str:
        """
        Format multiple flights as structured text for LLM prompts.

        Args:
            flights: List of flight dictionaries

        Returns:
            Formatted string describing all flights
        """
        formatted = []
        for i, flight in enumerate(flights):
            arm_index = f"arm_{i}"
            formatted.append(self.format_flight_for_prompt(flight, arm_index))

        return "\n".join(formatted)

    def _call_gemini(self, prompt: str, temperature: float = 0.7) -> str:
        """Call Gemini API with error handling."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=2048
                )
            )
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return ""

    def _extract_json_from_response(self, text: str) -> Dict:
        """Extract JSON from LLM response (handles ```json markers)."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            print(f"Raw text: {text[:500]}")
            return {}

    def _extract_jsonl_from_response(self, text: str) -> List[Dict]:
        """Extract JSONL from LLM response (utility estimation returns JSONL)."""
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
        Generate questions to ask user after they rank flights (Prompt 2 from LILO paper).

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

        # Prompt 2 from LILO paper (adapted for flights)
        prompt = f"""You are an expert in determining whether a human decision maker (DM)
is going to be satisfied with a set of flight options.

## Experimental outcomes:
So far, we have obtained the following flight options:
{experiment_data}

## Human feedback messages:
We have also received the following messages from the DM:
{human_feedback if human_feedback else "No previous feedback yet."}

## Your task:
The decision maker has indicated interest in these flights: {selected_indices}.

Given the above, your task is to predict preferences between flight options.

In order to better understand the decision maker's utility function, you want to ask them about their optimization goals or for feedback regarding specific flight options.

First, analyze the decision maker's goals and feedback messages to understand their overall preferences.

Then, provide a list of {n_questions} questions you would ask the decision maker to better understand their internal utility model.

Your questions can be either general or referring to specific outcomes. For instance, you may ask the decision maker:
- questions clarifying the optimization objective (price vs time vs convenience)
- to rank two (or more) flights
- how to improve certain flights
- for a rating regarding a specific flight
- about their priorities (price, duration, stops, airlines, times)

When referring to specific flights, always state the arm_index involved.

Your questions should help you predict pairwise preferences between any two flight options from the set provided above.

Return your final answer as a json file with the following format containing exactly {n_questions} most important questions:
```json
{{
"q1" : <question1>,
"q2" : <question2>,
"q3" : <question3>
}}
```"""

        response = self._call_gemini(prompt, temperature=0.7)
        questions_dict = self._extract_json_from_response(response)

        # Extract questions from dict {"q1": "...", "q2": "...", ...}
        questions = [questions_dict.get(f"q{i+1}", "") for i in range(n_questions)]
        return [q for q in questions if q]  # Filter empty

    def summarize_feedback(self, flights_shown: List[Dict]) -> str:
        """
        Summarize all user feedback into optimization goals (Prompt 5 from LILO paper).

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
            human_feedback += f"Feedback: {r['feedback']}\n"
            if 'answers' in r and r['answers']:
                human_feedback += f"Q&A: {r['answers']}\n"
            human_feedback += "\n"

        if not human_feedback:
            return ""

        # Prompt 5 from LILO paper
        prompt = f"""You are an expert in determining whether a human decision maker (DM)
is going to be satisfied with a set of flight options.

## Experimental outcomes:
So far, we have obtained the following flight options:
{experiment_data}

## Human feedback messages:
We have also received the following messages from the DM:
{human_feedback}

## Your task:
Given the above your task is to summarize the human feedback messages
into a clear description of the DM's optimization goals.

Make your summary as quantitative as possible so that it can be easily
used for utility estimation.

After analyzing the human feedback messages, return your final answer
as a json file with the following format:
```json
{{
"summary": <summary>
}}
```

Remember about the ```json header!"""

        response = self._call_gemini(prompt, temperature=0.5)
        summary_dict = self._extract_json_from_response(response)

        return summary_dict.get("summary", "")

    def estimate_utilities(self, flights: List[Dict]) -> List[float]:
        """
        Estimate utility (p_accept) for each flight (Prompt 4 from LILO paper).

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
            human_feedback += f"Feedback: {r['feedback']}\n"
            if 'answers' in r and r['answers']:
                human_feedback += f"Q&A: {r['answers']}\n"
            human_feedback += "\n"

        if not human_feedback:
            human_feedback = "No feedback yet."

        # Include summary if available
        summary_text = f"\n## Feedback Summary:\n{self.feedback_summary}" if self.feedback_summary else ""

        # Prompt 4 from LILO paper
        prompt = f"""You are an expert in determining whether a human decision maker (DM)
is going to be satisfied with a set of flight options.

## Experimental outcomes:
So far, we have obtained the following flight options:
{experiment_data}

## Human feedback messages:
We have also received the following messages from the DM:
{human_feedback}
{summary_text}

## Your task:
Given the above your task is to predict the probability of the
decision maker being satisfied with the flight options.

First, analyze the human feedback messages to understand the DM's preferences.

Then, provide your predictions for all flights in the set of all
experimental outcomes above.

Return your final answer as a jsonl file with the following format:
```jsonl
{{
"arm_index": "arm_0",
"reasoning": <reasoning>,
"p_accept": <probability>
}}
{{
"arm_index": "arm_1",
"reasoning": <reasoning>,
"p_accept": <probability>
}}
...
```

Where <reasoning> should be a short reasoning for your prediction and
<probability> should be your best estimate for the probability between
0 and 1 that the DM will be satisfied with the corresponding flight.

Provide your predictions for ALL flights in the set of experimental outcomes above. That is, for EACH flight from arm_0 to arm_{len(flights)-1}.

Do not generate any Python code. Just return your predictions as plain text."""

        response = self._call_gemini(prompt, temperature=0.3)
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

        Round 1: Diverse selection (cheapest, fastest, mixed)
        Round 2+: Use utility estimates to select promising candidates

        Args:
            all_flights: All available flights
            round_num: Current round number
            n_candidates: Number of flights to show

        Returns:
            List of selected flight dictionaries
        """
        if round_num == 1:
            # Round 1: Show diverse set
            cheapest = sorted(all_flights, key=lambda x: x.get('price', 999999))[:5]
            fastest = sorted(all_flights, key=lambda x: x.get('duration_min', 999999))[:5]
            # Random remaining
            remaining = [f for f in all_flights if f not in cheapest and f not in fastest]
            random.shuffle(remaining)
            candidates = cheapest + fastest + remaining[:max(0, n_candidates-10)]
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
                 user_answers: Optional[Dict] = None,
                 n_candidates: int = 15) -> Dict:
        """
        Run one LILO round.

        Args:
            all_flights: All available flights
            round_num: Current round number
            user_rankings: Indices of flights user ranked (in order of preference)
            user_feedback: User's natural language feedback
            user_answers: Optional Q&A answers from previous round
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
            'feedback': user_feedback,
            'answers': user_answers or {}
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
