"""
Wrapper to call LISTEN's main.py with flight data.
"""
import os
import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from backend.listen_data_converter import flights_to_listen_csv, generate_listen_config


def rank_flights_with_listen_main(
    flights: List[Dict],
    user_prompt: str,
    user_preferences: Dict = None,
    n_iterations: int = 5
) -> List[Dict]:
    """
    Rank flights using LISTEN's main.py framework.

    Args:
        flights: List of flight dictionaries from Amadeus
        user_prompt: User's original search query
        user_preferences: Dict with origins, destinations, preferences
        n_iterations: Number of LISTEN iterations (batches)

    Returns:
        List of all flights ranked by LISTEN-U (best to worst)
    """
    if not flights:
        return []

    print(f"🎯 Running LISTEN main.py with {len(flights)} flights...")

    # Create timestamp-based tag for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"flight_{timestamp}"

    # Paths
    listen_dir = Path(__file__).parent.parent / "LISTEN"
    input_dir = listen_dir / "input"
    config_dir = listen_dir / "configs"

    # Create directories if they don't exist
    input_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    # File names
    csv_filename = f"{tag}.csv"
    config_filename = f"{tag}.yml"

    csv_path = input_dir / csv_filename
    config_path = config_dir / config_filename

    # Step 1: Convert flights to LISTEN CSV format
    print(f"  ✓ Converting flights to CSV: {csv_filename}")
    flights_to_listen_csv(flights, str(csv_path), user_preferences)

    # Step 2: Generate LISTEN YAML config
    print(f"  ✓ Generating config: {config_filename}")
    generate_listen_config(user_prompt, csv_filename, str(config_path), tag)

    # Step 3: Run LISTEN main.py
    print(f"  ✓ Running LISTEN main.py with {n_iterations} iterations...")
    print(f"  ⏳ Expected runtime: ~{n_iterations * 5} seconds (Gemini rate limiting: 13 req/min)")

    # Use current Python interpreter (has all packages from requirements.txt)
    # LISTEN requires Python 3.10+ for union type syntax (float | None)
    # Streamlit Cloud deployment uses runtime.txt to ensure Python 3.11
    import sys

    python_executable = sys.executable
    print(f"  ✓ Using Python: {python_executable}")

    cmd = [
        python_executable,
        "main.py",
        "--scenario", tag,
        "--algo", "utility",  # Main LISTEN algorithm - learns utility function over iterations
        "--mode", "User",
        "--max-iters", str(n_iterations),  # 25 iterations for production-quality utility learning
        "--api-model", "gemini",
        "--seed", "42"
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(listen_dir),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout (25 iterations * ~5 sec/iter with rate limiting = ~2 min, giving buffer)
        )

        if result.returncode != 0:
            error_msg = f"LISTEN main.py failed with return code {result.returncode}"
            print(f"  ⚠️ {error_msg}")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            raise RuntimeError(f"{error_msg}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

        print(f"  ✓ LISTEN completed successfully!")

        # VERIFICATION: Print LISTEN output for debugging
        print(f"\n  📊 LISTEN OUTPUT (last 1000 chars):")
        print(f"  {result.stdout[-1000:]}")
        print()

    except subprocess.TimeoutExpired as e:
        error_msg = f"LISTEN timed out after 10 minutes"
        print(f"  ⚠️ {error_msg}")
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Error running LISTEN: {str(e)}"
        print(f"  ⚠️ {error_msg}")
        raise RuntimeError(error_msg) from e

    # Step 4: Parse LISTEN JSON output to get utility function and rank ALL flights
    print(f"  ✓ Parsing LISTEN results...")

    # LISTEN saves JSON output in outputs/{tag}/ directory
    output_dir = listen_dir / "outputs" / tag  # Note: "outputs" not "output"

    if output_dir.exists():
        # Find the JSON result file
        json_files = list(output_dir.glob("*.json"))

        if json_files:
            try:
                # Read the LISTEN output JSON
                with open(json_files[0]) as f:
                    listen_data = json.load(f)

                # Extract the final utility function weights
                final_utility = listen_data['optimization_results']['final_utility_function']
                weights = final_utility['weights']

                print(f"  ✓ Loaded LISTEN utility function:")
                print(f"    📊 Learned Weights from User Prompt:")
                print(f"       Price weight: {weights.get('price', 0):.4f}")
                print(f"       Duration weight: {weights.get('duration_min', 0):.4f}")
                print(f"       Stops weight: {weights.get('stops', 0):.4f}")
                print(f"    💡 Interpretation:")
                if weights.get('price', 0) > 0:
                    print(f"       User prefers EXPENSIVE flights (positive price weight)")
                elif weights.get('price', 0) < 0:
                    print(f"       User prefers CHEAP flights (negative price weight)")
                else:
                    print(f"       User is NEUTRAL about price")
                print(f"    📝 User's original prompt: {user_prompt[:100]}...")

                # Now calculate utility for ALL flights using the learned weights
                # First, we need to normalize each metric to [0, 1] like LISTEN does
                def normalize_metric(value, min_val, max_val, lower_is_better=False):
                    """Normalize to [0,1] range"""
                    if max_val == min_val:
                        return 0.5
                    normalized = (value - min_val) / (max_val - min_val)
                    if lower_is_better:
                        normalized = 1 - normalized  # Invert so 1 is best
                    return normalized

                # Calculate min/max for each metric across all flights
                prices = [f['price'] for f in flights]
                durations = [f['duration_min'] for f in flights]
                stops = [f['stops'] for f in flights]

                min_price, max_price = min(prices), max(prices)
                min_duration, max_duration = min(durations), max(durations)
                min_stops, max_stops = min(stops), max(stops)

                # Calculate utility for each flight
                flight_utilities = []
                for i, flight in enumerate(flights):
                    # Normalize metrics to [0,1] WITHOUT inverting
                    # LISTEN's weights already encode direction:
                    #   - Positive weight = higher is better (e.g., +0.4 for price means prefer expensive)
                    #   - Negative weight = lower is better (e.g., -0.3 for stops means prefer fewer stops)
                    price_score = normalize_metric(flight['price'], min_price, max_price, lower_is_better=False)
                    duration_score = normalize_metric(flight['duration_min'], min_duration, max_duration, lower_is_better=False)
                    stops_score = normalize_metric(flight['stops'], min_stops, max_stops, lower_is_better=False)

                    # Calculate utility = sum(weight_i * score_i)
                    utility = (
                        weights.get('price', 0) * price_score +
                        weights.get('duration_min', 0) * duration_score +
                        weights.get('stops', 0) * stops_score
                    )

                    flight_utilities.append((i, utility, flight))

                # Sort by utility (highest first = best)
                flight_utilities.sort(key=lambda x: x[1], reverse=True)

                # Extract sorted flights
                ranked_flights = [f[2] for f in flight_utilities]

                print(f"  ✓ Ranked {len(ranked_flights)} flights by LISTEN utility")
                print(f"  ✓ Top 3 utilities: {[f[1] for f in flight_utilities[:3]]}")
                print(f"  ✓ Top 3 prices: {[f[2]['price'] for f in flight_utilities[:3]]}")

                return ranked_flights

            except Exception as e:
                import traceback
                error_msg = f"Error parsing LISTEN JSON output: {str(e)}"
                print(f"  ⚠️ {error_msg}")
                print(f"  ⚠️ Traceback: {traceback.format_exc()}")
                raise RuntimeError(error_msg) from e

    # No fallback - if we get here, LISTEN didn't produce expected output
    error_msg = f"LISTEN did not produce expected JSON output in {output_dir}"
    print(f"  ⚠️ {error_msg}")
    print(f"  ⚠️ Output directory checked: {output_dir}")
    if output_dir.exists():
        print(f"  ⚠️ Directory contents: {list(output_dir.iterdir())}")
    raise RuntimeError(error_msg)


def cleanup_listen_files(tag: str):
    """
    Clean up temporary LISTEN files after ranking.

    Args:
        tag: The tag used for this LISTEN run
    """
    listen_dir = Path(__file__).parent.parent / "LISTEN"

    # Remove CSV
    csv_path = listen_dir / "input" / f"{tag}.csv"
    if csv_path.exists():
        csv_path.unlink()

    # Remove config
    config_path = listen_dir / "configs" / f"{tag}.yml"
    if config_path.exists():
        config_path.unlink()

    # Remove output directory
    output_dir = listen_dir / "output" / tag
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)


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
        },
        {
            'airline': 'Delta',
            'flight_number': 'DL5678',
            'origin': 'ITH',
            'destination': 'DCA',
            'departure_time': '2024-11-15T14:05:00',
            'arrival_time': '2024-11-15T18:00:00',
            'duration_min': 235,
            'stops': 1,
            'price': 214
        }
    ]

    user_prompt = "I need a cheap flight from Ithaca to DC"
    user_prefs = {
        'origins': ['ITH'],
        'destinations': ['IAD', 'DCA'],
        'preferences': {'prefer_cheap': True}
    }

    ranked = rank_flights_with_listen_main(sample_flights, user_prompt, user_prefs, n_iterations=3)

    print("\nRanked flights:")
    for i, flight in enumerate(ranked, 1):
        print(f"{i}. {flight['airline']}{flight['flight_number']} - ${flight['price']}")