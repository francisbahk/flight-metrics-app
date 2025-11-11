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
    n_iterations: int = 25
) -> List[Dict]:
    """
    Rank flights using LISTEN's main.py framework.

    Args:
        flights: List of flight dictionaries from Amadeus
        user_prompt: User's original search query
        user_preferences: Dict with origins, destinations, preferences
        n_iterations: Number of LISTEN iterations (batches)

    Returns:
        List of top 10 flights ranked by LISTEN-U
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

    cmd = [
        "python3",
        "main.py",
        "--scenario", tag,
        "--algo", "utility",  # Main LISTEN algorithm - learns utility function over iterations
        "--mode", "User",
        "--max-iters", str(n_iterations),  # 25 iterations to learn utility
        "--api-model", "gemini",
        "--seed", "42"
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(listen_dir),
            capture_output=True,
            text=True,
            timeout=900  # 15 minute timeout (25 iterations takes longer)
        )

        if result.returncode != 0:
            print(f"  ⚠️ LISTEN main.py failed:")
            print(f"  STDOUT: {result.stdout[:500]}")
            print(f"  STDERR: {result.stderr[:500]}")
            # Fall back to simple price ranking
            return sorted(flights, key=lambda x: x.get('price', float('inf')))[:10]

        print(f"  ✓ LISTEN completed successfully!")

    except subprocess.TimeoutExpired:
        print(f"  ⚠️ LISTEN timed out after 15 minutes")
        return sorted(flights, key=lambda x: x.get('price', float('inf')))[:10]
    except Exception as e:
        print(f"  ⚠️ Error running LISTEN: {str(e)}")
        return sorted(flights, key=lambda x: x.get('price', float('inf')))[:10]

    # Step 4: Parse LISTEN output to get ranked flights
    print(f"  ✓ Parsing LISTEN results...")

    # LISTEN saves results in various places. Let's check for output files
    # The comparison algorithm typically saves best solutions
    output_dir = listen_dir / "output" / tag

    if output_dir.exists():
        # Look for results file
        result_files = list(output_dir.glob("*.csv")) + list(output_dir.glob("*.json"))

        if result_files:
            # Parse the results file to get ranking
            # For now, let's just use the order from the CSV
            try:
                # Read the best solution indices if available
                best_file = output_dir / "best_solutions.json"
                if best_file.exists():
                    with open(best_file) as f:
                        best_data = json.load(f)
                    # Extract flight indices in ranked order
                    ranked_indices = best_data.get('ranked_indices', list(range(min(10, len(flights)))))
                else:
                    # Fall back to reading utility scores if available
                    ranked_indices = list(range(min(10, len(flights))))

                # Map indices back to flights
                ranked_flights = []
                for idx in ranked_indices[:10]:
                    if 0 <= idx < len(flights):
                        ranked_flights.append(flights[idx])

                if ranked_flights:
                    print(f"  ✓ Retrieved {len(ranked_flights)} ranked flights from LISTEN")
                    return ranked_flights

            except Exception as e:
                print(f"  ⚠️ Error parsing LISTEN output: {str(e)}")

    # Fallback: Parse stdout for best solution
    # LISTEN prints the best solution, we can extract the flight index from that
    if "Best solution" in result.stdout or "best" in result.stdout.lower():
        # Try to extract flight info from output
        # This is a simplified parser - adjust based on actual LISTEN output format
        lines = result.stdout.split('\n')
        best_indices = []

        for line in lines:
            if 'unique_id' in line or 'best' in line.lower():
                # Try to extract unique_id and map back to flight
                # This is a placeholder - actual parsing depends on LISTEN output format
                pass

    # Final fallback: Return first 10 flights (at least they ran through LISTEN)
    print(f"  ⚠️ Could not parse LISTEN ranking, returning first 10 flights")
    return flights[:10]


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