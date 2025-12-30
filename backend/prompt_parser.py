"""
LLM-based flight prompt parser using Gemini.
Extracts structured flight search parameters from natural language.
"""
import os
import json
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini with new SDK
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    genai_client = None


def parse_flight_prompt_with_llm(prompt: str) -> Dict:
    """
    Parse natural language flight prompt using Gemini LLM.

    Args:
        prompt: User's natural language query about flights

    Returns:
        Dictionary with structured flight search parameters
    """

    llm_prompt = f"""You are a flight search assistant. Extract structured information from this flight search query.

User Query:
{prompt}

Extract and return ONLY a valid JSON object with these fields:
{{
    "origins": ["AIRPORT_CODE"],  // List of origin airport codes (IATA 3-letter)
    "destinations": ["AIRPORT_CODE"],  // List of destination airport codes
    "departure_dates": ["YYYY-MM-DD"],  // List of departure dates (e.g., ["2024-12-20", "2024-12-21"] for "Friday or Saturday")
    "return_dates": ["YYYY-MM-DD"],  // List of return dates ONLY if explicitly mentioned (e.g., ["2024-12-27", "2024-12-28"] for "Dec 27th or 28th")
    "preferences": {{
        "prefer_direct": true/false,  // Wants nonstop flights
        "prefer_cheap": true/false,  // Price sensitive
        "prefer_fast": true/false,  // Wants shortest duration
        "avoid_early_departures": true/false,  // Doesn't want flights before 7am
        "min_connection_time": 0,  // Minimum connection time in minutes (0 = no minimum)
        "max_layover_time": 10000,  // Maximum layover time in minutes (10000 = effectively no limit)
        "preferred_airlines": ["AIRLINE_CODE"],  // e.g. ["UA", "AA"]
        "avoid_airports": ["AIRPORT_CODE"],  // Airports to avoid e.g. ["JFK", "EWR"]
        "fly_america_act": true/false  // Must use US carriers
    }},
    "constraints": {{
        "latest_arrival": "HH:MM or null",  // Latest acceptable arrival time
        "earliest_departure": "HH:MM or null"  // Earliest acceptable departure
    }}
}}

CRITICAL INSTRUCTIONS:
1. Return ONLY the JSON object, no other text or explanation
2. Use null for missing fields
3. Infer preferences from context (e.g., "as cheap as possible" → prefer_cheap: true)
4. **DATE PARSING - VERY IMPORTANT**:
   - Today is {datetime.now().strftime("%Y-%m-%d")}
   - If user mentions a date WITHOUT a year (e.g., "December 22nd", "Christmas"), ALWAYS assume they mean the NEXT future occurrence of that date
   - NEVER return dates in the past - always interpret ambiguous dates as future dates
   - Examples: If today is 2025-11-26 and user says "December 22nd", return "2025-12-22" (NOT 2024-12-22)
5. **MULTIPLE DATES**: If user mentions multiple dates (e.g., "weekend of Jan 5", "Friday or Saturday", "Jan 5 or 6"), extract ALL dates as separate items in departure_dates list
6. **RETURN FLIGHTS**: ONLY set return_dates if user explicitly mentions returning, coming back, or round-trip. Extract ALL mentioned return dates in return_dates list (e.g., "return Dec 27 or 28 or 29" → ["2025-12-27", "2025-12-28", "2025-12-29"]). If only one-way is mentioned, set return_dates to empty list []

**MOST IMPORTANT - AIRPORT CODE EXTRACTION:**
You MUST use your knowledge of world airports to convert city/region names into proper IATA airport codes (3-letter codes).

**CRITICAL RULE - Explicit Airport Codes vs City Names:**
- If user specifies a specific 3-letter IATA airport code (e.g., "JFK", "LAX", "ORD"), return ONLY that code
- If user specifies a city/region name (e.g., "New York", "NYC", "Los Angeles"), return ALL major airports for that area
- Examples:
  - User says "JFK" → ["JFK"] (NOT ["JFK", "LGA", "EWR"])
  - User says "NYC" or "New York" → ["JFK", "LGA", "EWR"]
  - User says "LAX" → ["LAX"] (NOT all LA airports)
  - User says "Los Angeles" or "LA" → ["LAX", "BUR", "SNA"]
  - User says "ORD" → ["ORD"] (NOT ["ORD", "MDW"])
  - User says "Chicago" → ["ORD", "MDW"]

**CITY → AIRPORT CODE CONVERSION (only when user provides city name, not specific code):**
- "Houston" or "Houston Texas" or "Houston TX" → ["IAH", "HOU"]
- "New York" or "New York City" or "NYC" → ["JFK", "LGA", "EWR"]
- "Los Angeles" or "LA" → ["LAX", "BUR", "SNA"]
- "San Francisco" or "SF" or "Bay Area" → ["SFO", "OAK", "SJC"]
- "Chicago" → ["ORD", "MDW"]
- "Washington DC" or "DC" or "Washington" → ["DCA", "IAD", "BWI"]
- "Boston" → ["BOS"]
- "Seattle" → ["SEA"]
- "Miami" → ["MIA", "FLL"]
- "Dallas" → ["DFW", "DAL"]
- "Atlanta" → ["ATL"]
- "Denver" → ["DEN"]
- "Phoenix" → ["PHX"]
- "London" → ["LHR", "LGW", "STN"]
- "Paris" → ["CDG", "ORY"]
- "Ithaca" or "Ithaca NY" → ["ITH", "SYR"]

**DO NOT extract random words from the sentence as airport codes!**
**ONLY return valid 3-letter IATA airport codes that you KNOW are real airports.**
**CRITICAL: Common words that are NOT airport codes:**
- "get", "fly", "go", "come", "see", "via", "and", "the", "for", "can", "use", "put", "set", "may", "run", "let"
- If you see "get to [city]", extract [city] as the destination, NOT "get"
- Example: "I need to GET to Atlanta" → destinations: ["ATL"] (NOT ["GET"])

For small cities, include nearby major airports within 100 miles as alternatives.
"""

    try:
        if not genai_client:
            raise ValueError("Gemini API key not configured")

        # Use new SDK
        response = genai_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=llm_prompt
        )

        # Extract JSON from response
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]

        # Parse JSON
        parsed = json.loads(response_text)

        # Filter out common words that are NOT airport codes
        INVALID_CODES = {
            'GET', 'FLY', 'GO', 'RUN', 'SET', 'PUT', 'LET', 'MAY', 'CAN', 'USE',
            'AND', 'THE', 'FOR', 'ARE', 'WAS', 'SEE', 'VIA', 'HAS', 'HAD', 'BUT',
            'NOT', 'YOU', 'ALL', 'DAY', 'NEW', 'OLD', 'WAY', 'OUT', 'NOW', 'OUR',
            'TWO', 'HOW', 'ITS', 'SAY', 'SHE', 'HER', 'HIM', 'HIS', 'WHO', 'BOY',
            'DID', 'ITS', 'LET', 'OWN', 'SAW', 'TOO', 'WHY', 'TRY', 'ASK', 'MEN',
            'BIG', 'FEW', 'GOT', 'HAS', 'HER', 'HIM', 'HIS', 'HOW', 'MAN', 'NEW'
        }

        # Clean origins
        if 'origins' in parsed:
            parsed['origins'] = [code for code in parsed['origins'] if code.upper() not in INVALID_CODES]

        # Clean destinations
        if 'destinations' in parsed:
            parsed['destinations'] = [code for code in parsed['destinations'] if code.upper() not in INVALID_CODES]

        # Validate and clean up
        departure_dates = parsed.get('departure_dates', [])

        # Handle legacy single date format for backward compatibility
        if not departure_dates and parsed.get('departure_date'):
            departure_dates = [parsed.get('departure_date')]

        # Handle return dates (support both old single return_date and new multiple return_dates)
        return_dates = parsed.get('return_dates', [])
        if not return_dates and parsed.get('return_date'):
            # Backward compatibility: convert single return_date to list
            return_dates = [parsed.get('return_date')]

        result = {
            'parsed_successfully': True,
            'origins': parsed.get('origins', []),
            'destinations': parsed.get('destinations', []),
            'departure_dates': departure_dates,  # Now a list
            'return_dates': return_dates,  # Now a list
            'return_date': return_dates[0] if return_dates else None,  # Keep for backward compatibility
            'preferences': parsed.get('preferences', {}),
            'constraints': parsed.get('constraints', {}),
            'original_prompt': prompt
        }

        return result

    except Exception as e:
        print(f"⚠️ LLM parsing failed: {e}")
        raise RuntimeError(f"Failed to parse flight prompt: {e}") from e


def get_test_api_fallback(airport_code: str) -> tuple[str, Optional[str]]:
    """
    Convert small airports to major ones for Amadeus test API.

    Returns:
        (fallback_code, warning_message)
    """
    fallbacks = {
        'ITH': ('SYR', 'Ithaca (ITH) not in API, using Syracuse (SYR)'),
        'ELM': ('SYR', 'Elmira (ELM) not in API, using Syracuse (SYR)'),
        'BGM': ('SYR', 'Binghamton (BGM) not in API, using Syracuse (SYR)'),
    }

    if airport_code in fallbacks:
        return fallbacks[airport_code]
    return (airport_code, None)


# For testing
if __name__ == "__main__":
    test_prompt = """
    On October 20 I need to fly from Ithaca NY to Reston VA.
    I prefer direct flights and want to avoid early morning departures before 7am.
    I fly United and need to comply with Fly America Act.
    """

    result = parse_flight_prompt_with_llm(test_prompt)
    print(json.dumps(result, indent=2))
