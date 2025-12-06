"""
Convert Amadeus flight data to LISTEN CSV format and generate YAML configs.
"""
import csv
import yaml
from datetime import datetime
from typing import List, Dict
from pathlib import Path


def flights_to_listen_csv(flights: List[Dict], csv_path: str, user_preferences: Dict = None) -> str:
    """
    Convert Amadeus flights to LISTEN CSV format.

    Args:
        flights: List of flight dictionaries from Amadeus
        csv_path: Path where CSV should be saved
        user_preferences: User preferences dict (for dis_from_origin/dest calculation)

    Returns:
        Path to the created CSV file
    """
    # Fixed reference date (same as LISTEN examples use 1900-10-20)
    # We'll use epoch for simplicity
    reference_date = datetime(1970, 1, 1)

    # Get origin/destination from first flight
    if not flights:
        raise ValueError("No flights provided")

    origin = flights[0].get('origin', 'UNK')
    destination = flights[0].get('destination', 'UNK')

    # Get preferred airports from user preferences
    preferred_origins = user_preferences.get('origins', [origin]) if user_preferences else [origin]
    preferred_dests = user_preferences.get('destinations', [destination]) if user_preferences else [destination]

    # Open CSV for writing
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            '', 'Unnamed: 0', 'unique_id', 'is_best', 'name', 'origin', 'destination',
            'departure time', 'arrival time', 'duration', 'stops', 'price',
            'dis_from_origin', 'dis_from_dest', 'departure_dt', 'departure_seconds',
            'arrival_dt', 'arrival_seconds', 'duration_min'
        ])

        for idx, flight in enumerate(flights):
            # Extract flight data
            airline = flight.get('airline', 'Unknown')
            flight_num = flight.get('flight_number', '')
            origin_code = flight.get('origin', origin)
            dest_code = flight.get('destination', destination)

            # Parse datetime strings
            dep_str = flight.get('departure_time', '')
            arr_str = flight.get('arrival_time', '')

            try:
                # Parse ISO format: "2024-11-15T08:00:00"
                dep_dt = datetime.fromisoformat(dep_str.replace('Z', '+00:00'))
                arr_dt = datetime.fromisoformat(arr_str.replace('Z', '+00:00'))
            except:
                # Fallback: use current time
                dep_dt = datetime.now()
                arr_dt = datetime.now()

            # Format times like "10:28 AM on Mon, Oct 20"
            dep_formatted = dep_dt.strftime("%I:%M %p on %a, %b %d")
            arr_formatted = arr_dt.strftime("%I:%M %p on %a, %b %d")

            # Calculate seconds since reference
            dep_seconds = (dep_dt - reference_date).total_seconds()
            arr_seconds = (arr_dt - reference_date).total_seconds()

            # Duration
            duration_min = flight.get('duration_min', 0)
            hours = duration_min // 60
            mins = duration_min % 60
            duration_formatted = f"{hours} hr {mins} min"

            # Stops
            stops = flight.get('stops', 0)

            # Price
            price = flight.get('price', 0)

            # Distance from origin/dest (normalized 0-1, where 0 = match)
            # If origin matches preferred, dis = 0, else 1
            dis_from_origin = 0.0 if origin_code in preferred_origins else 1.0
            dis_from_dest = 0.0 if dest_code in preferred_dests else 1.0

            # Unique ID
            unique_id = f"{origin_code}_{dest_code}{idx}"

            # is_best - set to False for all initially
            is_best = False

            # Write row
            writer.writerow([
                idx,  # Row index
                idx,  # Unnamed: 0
                unique_id,
                is_best,
                airline,
                origin_code,
                dest_code,
                dep_formatted,
                arr_formatted,
                duration_formatted,
                float(stops),
                float(price),
                dis_from_origin,
                dis_from_dest,
                dep_dt.strftime("%Y-%m-%d %H:%M:%S"),
                dep_seconds,
                arr_dt.strftime("%Y-%m-%d %H:%M:%S"),
                arr_seconds,
                duration_min
            ])

    return csv_path


def generate_listen_config(
    user_prompt: str,
    csv_filename: str,
    config_path: str,
    tag: str = "dynamic_flight"
) -> str:
    """
    Generate LISTEN YAML config from user prompt.

    Args:
        user_prompt: Original user prompt (e.g., "cheap flight from ITH to DCA")
        csv_filename: Name of the CSV file (relative to LISTEN/input/)
        config_path: Path where YAML config should be saved
        tag: Unique tag for this scenario

    Returns:
        Path to the created config file
    """
    config = {
        'tag': tag,
        'data_csv': f'input/{csv_filename}',
        'rate_limit_delay': 6.0,  # Gemini API: 15 req/min free tier limit → use 10 req/min = 60/10 = 6 sec (safe buffer for retries)
        'metric_columns': [
            'name',
            'origin',
            'destination',
            'departure time',
            'arrival time',
            'duration',
            'stops',
            'price',
            'dis_from_origin',
            'dis_from_dest',
            'departure_seconds',
            'arrival_seconds',
            'duration_min'
        ],
        'metric_signs': {
            'name': 0,  # Not a metric
            'origin': 0,  # Not a metric
            'destination': 0,  # Not a metric
            'departure time': 0,  # Not a metric
            'arrival time': 0,  # Not a metric
            'duration': 0,  # Not a metric
            'stops': -1,  # Lower is better
            'price': -1,  # Lower is better
            'dis_from_origin': -1,  # Lower is better
            'dis_from_dest': -1,  # Lower is better
            'departure_seconds': 0,  # Time of day preference varies
            'arrival_seconds': 0,  # Time of day preference varies
            'duration_min': -1  # Lower is better
        },
        'non_metric_metrics': [
            'name',
            'origin',
            'destination',
            'departure time',
            'arrival time',
            'duration'
        ],
        'prompts': {
            'scenario_header': """You are an expert travel scheduling agent that specializes in air fare.
The data set is only for a one-way flight.
Use these definitions:
• name: name of airline operating the flight
• origin: origin airport
• destination: destination airport
• departure time: time of departure from origin airport
• arrival time: time of arrival at destination airport
• duration: how long the flight is
• stops: number of layover stops
• price: cost of the flight
• dis_from_origin: distance of origin airport from where the customer prefers (lower is better)
• dis_from_dest: distance of arrival airport from where the customer prefers (lower is better)
• departure_seconds: time of departure since a fixed date in seconds
• arrival_seconds: time of arrival since a fixed date in seconds
• duration_min: duration of total flight in minutes (lower is better)
""",
            'comparison_base': """{scenario_header}
Your task: Identify the SINGLE BEST flight from the options below.

User's request: {user_prompt}
""",
            'utility_base': """{scenario_header}
Your task: Define a utility function by assigning weights to these metrics, reflecting priorities for customer satisfaction. Use positive weights for metrics where higher is better and negative weights where lower is better.

User's request: {user_prompt}

Each metric is first mapped to a normalized [0,1] preference score.

Return your response in this EXACT JSON format (include ALL listed metric keys):
{{
  "weights": {{
  "stops": 0.0,
  "price":0.0,
  "dis_from_origin":0.0,
  "dis_from_dest":0.0,
  "departure_seconds":0.0,
  "arrival_seconds":0.0,
  "duration_min":0.0
  }},
  "formula": "utility = sum(weight_i * score_i) for all metrics (score_i ∈ [0,1])",
  "description": "Brief explanation of your weighting rationale"
}}

STRICT OUTPUT RULES:
- Output ONLY a valid JSON object. No code fences. No prose. No Python.
- Weights may be positive or negative. Use positive weights where higher is better and negative weights where lower is better.
- Do not include any additional keys beyond weights, formula, description.
- Ensure every weight key exactly matches the metric names listed above.
- End your output with the delimiter: <END_JSON>
""",
            'utility_refinement': """You are refining a utility function for flight itinerary selection (iteration {iteration}).

User's original request: {user_prompt}

CURRENT UTILITY FUNCTION (that produced the best flight below):
Weights: {weights}
Formula: {formula}
Description: {description}

BEST FLIGHT FOUND with the above utility function:
{best_solution}

ANALYSIS:
- Does the flight align with the user's request?
- Does the flight chosen factor in cost/time based on user desire?
- Is this the most optimal flight?

Your task: Based on this best flight, adjust the utility function weights if needed.
Consider:
{policy_guidance}

Return your response in this EXACT JSON format (include ALL listed metric keys):
{{
  "weights": {{
  "stops":  <adjusted_weight>,
  "price": <adjusted_weight>,
  "dis_from_origin": <adjusted_weight>,
  "dis_from_dest": <adjusted_weight>,
  "departure_seconds": <adjusted_weight>,
  "arrival_seconds": <adjusted_weight>,
  "duration_min": <adjusted_weight>
  }},
  "formula": "utility = sum(weight_i * score_i) for all metrics (score_i ∈ [0,1])",
  "description": "Brief explanation of your adjustments and reasoning based on the best schedule shown above"
}}

STRICT OUTPUT RULES:
- Output ONLY a valid JSON object. No code fences. No prose. No Python.
- Weights may be positive or negative. Use positive weights where higher is better and negative weights where lower is better.
- Do not include any additional keys beyond weights, formula, description.
- Ensure every weight key exactly matches the metric names in the initial prompt.
- If no change is needed, return the previous weights unchanged in the JSON.
- End your output with the delimiter: <END_JSON>
"""
        },
        'default_mode': 'User',
        'modes': {
            'User': {
                'prompt': user_prompt,
                'weights': {
                    # Neutral weights (0.0) - let LLM learn from user prompt
                    # LLM will assign positive/negative weights based on user preferences
                    'stops': 0.0,
                    'price': 0.0,
                    'dis_from_origin': 0.0,
                    'dis_from_dest': 0.0,
                    'departure_seconds': 0.0,
                    'arrival_seconds': 0.0,
                    'duration_min': 0.0
                }
            }
        }
    }

    # Write YAML config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


if __name__ == "__main__":
    # Test with sample data
    sample_flights = [
        {
            'airline': 'United',
            'flight_number': 'UA1234',
            'origin': 'ITH',
            'destination': 'IAD',
            'departure_time': '2024-11-15T10:28:00',
            'arrival_time': '2024-11-15T11:50:00',
            'duration_min': 82,
            'stops': 0,
            'price': 204
        }
    ]

    csv_path = '/tmp/test_flights.csv'
    config_path = '/tmp/test_config.yml'

    flights_to_listen_csv(sample_flights, csv_path)
    generate_listen_config("cheap flight from ITH to IAD", "test_flights.csv", config_path)

    print(f"Created CSV: {csv_path}")
    print(f"Created config: {config_path}")