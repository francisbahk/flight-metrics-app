"""
LLM-based flight prompt parser using Gemini.
Extracts structured flight search parameters from natural language.
"""
import os
import json
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


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

EXAMPLES of correct city → airport code conversion:
- "Houston" or "Houston Texas" or "Houston TX" → ["IAH", "HOU"]
- "New York" or "New York City" or "NYC" → ["JFK", "LGA", "EWR"]
- "Los Angeles" or "LA" → ["LAX"]
- "San Francisco" or "SF" → ["SFO"]
- "Chicago" → ["ORD", "MDW"]
- "Washington DC" or "DC" or "Washington" → ["DCA", "IAD"]
- "Boston" → ["BOS"]
- "Seattle" → ["SEA"]
- "Miami" → ["MIA"]
- "Dallas" → ["DFW", "DAL"]
- "Atlanta" → ["ATL"]
- "Denver" → ["DEN"]
- "Phoenix" → ["PHX"]
- "London" → ["LHR", "LGW"]
- "Paris" → ["CDG"]
- "Ithaca" or "Ithaca NY" → ["ITH", "SYR"]

**DO NOT extract random words from the sentence as airport codes!**
**ONLY return valid 3-letter IATA airport codes that you KNOW are real airports.**

For small cities, include nearby major airports within 100 miles as alternatives.
"""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(llm_prompt)

        # Extract JSON from response
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]

        # Parse JSON
        parsed = json.loads(response_text)

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
        print(f"LLM parsing failed: {e}")
        # Fallback to simple parsing
        return parse_flight_prompt_simple(prompt)


def parse_flight_prompt_simple(prompt: str) -> Dict:
    """
    Simple fallback parser using regex when LLM fails.
    """
    result = {
        'parsed_successfully': False,
        'origins': [],
        'destinations': [],
        'departure_dates': [],  # Changed to list
        'return_date': None,
        'preferences': {},
        'warnings': ['Using simple parser - LLM parsing failed'],
        'original_prompt': prompt
    }

    # Try to extract airport codes
    airport_pattern = r'\b[A-Z]{3}\b'
    airports = re.findall(airport_pattern, prompt.upper())

    if len(airports) >= 2:
        result['origins'] = [airports[0]]
        result['destinations'] = [airports[1]]

    # Extract date patterns
    date_patterns = [
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY
        r'(\d{1,2})-(\d{1,2})-(\d{4})',  # MM-DD-YYYY
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s*(\d{4})',  # Month DD, YYYY
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})',  # Month DD (without year)
    ]

    for pattern in date_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            try:
                # Parse based on pattern
                if len(match.groups()) == 3 and match.group(1).isdigit():
                    # Numeric date MM/DD/YYYY or MM-DD-YYYY
                    month, day, year = match.groups()
                    result['departure_dates'].append(f"{year}-{month.zfill(2)}-{day.zfill(2)}")
                elif len(match.groups()) == 3:
                    # Month name with year: "January 5, 2026"
                    month_name = match.group(1)
                    day = match.group(2)
                    year = match.group(3)
                    month_num = datetime.strptime(month_name, "%B").month
                    result['departure_dates'].append(f"{year}-{month_num:02d}-{int(day):02d}")
                else:
                    # Month name without year: "January 5"
                    month_name = match.group(1)
                    day = match.group(2)
                    year = datetime.now().year
                    month_num = datetime.strptime(month_name, "%B").month
                    result['departure_dates'].append(f"{year}-{month_num:02d}-{int(day):02d}")
                break
            except:
                pass

    # Extract preferences from keywords
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ['cheap', 'budget', 'affordable', 'inexpensive']):
        result['preferences']['prefer_cheap'] = True

    if any(word in prompt_lower for word in ['fast', 'quick', 'shortest']):
        result['preferences']['prefer_fast'] = True

    if any(word in prompt_lower for word in ['direct', 'nonstop', 'non-stop']):
        result['preferences']['prefer_direct'] = True

    return result


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
