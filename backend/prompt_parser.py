"""
LLM-based flight prompt parser using Gemini.
Extracts structured flight search parameters from natural language.
"""
import os
import json
from typing import Dict, Optional
from datetime import datetime
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    genai_client = None


def parse_flight_prompt_with_llm(prompt: str) -> Dict:
    """
    Parse natural language flight prompt using Gemini LLM.
    Returns a dictionary with origins, destinations, departure_dates, return_dates.
    """

    llm_prompt = f"""You are a flight search assistant. Extract structured information from this flight search query.

User Query:
{prompt}

FIRST — decide whether this query is asking about a flight search.
A valid flight query must mention at least an origin and/or destination (a city, airport, or region).
If the query is gibberish, random words, or completely unrelated to flights, return:
{{
    "is_flight_query": false,
    "rejection_message": "Please describe your flight — include where you're flying from, where you're going, and when. For example: 'Flight from New York to Los Angeles on March 15th'.",
    "origins": null,
    "destinations": null,
    "departure_dates": null,
    "return_dates": null
}}

Otherwise, extract and return ONLY a valid JSON object with these fields:
{{
    "is_flight_query": true,
    "rejection_message": null,
    "origins": ["AIRPORT_CODE"],
    "destinations": ["AIRPORT_CODE"],
    "departure_dates": ["YYYY-MM-DD"],
    "return_dates": ["YYYY-MM-DD"]
}}

CRITICAL INSTRUCTIONS:
1. Return ONLY the JSON object, no other text or explanation.
2. Use null for missing fields.
3. **DATE PARSING**:
   - Today is {datetime.now().strftime("%Y-%m-%d")}.
   - Always interpret ambiguous dates as the next future occurrence. Never return past dates.
   - If multiple dates are mentioned (e.g. "Friday or Saturday", "Jan 5 or 6"), list all of them.
4. **RETURN FLIGHTS**: Only set return_dates if the user explicitly mentions a return trip. Otherwise set it to [].
5. **AIRPORT CODES**:
   - If the user writes a 3-letter IATA airport code (e.g. JFK, LAX, ORD), use it exactly as written.
   - If the user writes a city or region name (e.g. "New York", "LA", "Chicago"), use your knowledge to return all relevant airports for that area.
   - Be careful with 3-letter words that are common English words (e.g. "get", "fly", "can", "the"). Use context to decide: if it appears as a destination or origin it is likely an airport code; if it appears mid-sentence as a verb or article it is not.
"""

    try:
        if not genai_client:
            raise ValueError("Gemini API key not configured")

        response = genai_client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=llm_prompt
        )

        response_text = response.text.strip()

        # Strip markdown code fences if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]

        parsed = json.loads(response_text)

        # Check if the LLM determined this is not a flight query
        if not parsed.get('is_flight_query', True):
            return {
                'parsed_successfully': False,
                'user_message': parsed.get(
                    'rejection_message',
                    "Please describe your flight — include where you're flying from, where you're going, and when."
                ),
                'original_prompt': prompt,
            }

        # Filter out common words that are NOT airport codes
        INVALID_CODES = {
            'GET', 'FLY', 'GO', 'RUN', 'SET', 'PUT', 'LET', 'MAY', 'CAN', 'USE',
            'AND', 'THE', 'FOR', 'ARE', 'WAS', 'SEE', 'VIA', 'HAS', 'HAD', 'BUT',
            'NOT', 'YOU', 'ALL', 'DAY', 'NEW', 'OLD', 'WAY', 'OUT', 'NOW', 'OUR',
            'TWO', 'HOW', 'ITS', 'SAY', 'SHE', 'HER', 'HIM', 'HIS', 'WHO', 'BOY',
            'DID', 'ITS', 'LET', 'OWN', 'SAW', 'TOO', 'WHY', 'TRY', 'ASK', 'MEN',
            'BIG', 'FEW', 'GOT', 'HAS', 'HER', 'HIM', 'HIS', 'HOW', 'MAN', 'NEW'
        }

        if parsed.get('origins'):
            parsed['origins'] = [code for code in parsed['origins'] if code and code.upper() not in INVALID_CODES]

        if parsed.get('destinations'):
            parsed['destinations'] = [code for code in parsed['destinations'] if code and code.upper() not in INVALID_CODES]

        departure_dates = parsed.get('departure_dates') or []
        return_dates = parsed.get('return_dates') or []

        return {
            'parsed_successfully': True,
            'origins': parsed.get('origins') or [],
            'destinations': parsed.get('destinations') or [],
            'departure_dates': departure_dates,
            'return_dates': return_dates,
            'return_date': return_dates[0] if return_dates else None,
            'original_prompt': prompt,
        }

    except Exception as e:
        print(f"⚠️ LLM parsing failed: {e}")
        raise RuntimeError(f"Failed to parse flight prompt: {e}") from e


def get_test_api_fallback(airport_code: str) -> tuple:
    """No-op — returns airport code as-is. If Amadeus has no flights, the chat bot handles it."""
    return (airport_code, None)


if __name__ == "__main__":
    test_prompt = "Cheapest flight from New York to LA on March 1st, direct preferred"
    result = parse_flight_prompt_with_llm(test_prompt)
    print(json.dumps(result, indent=2))
