"""
Simple script to view saved flight ranking data with CSV exports.
"""
from backend.db import SessionLocal, Search, FlightCSV
import pandas as pd
import io

def view_all_data():
    """Display all saved searches and CSV exports."""
    db = SessionLocal()

    try:
        print("=" * 80)
        print("FLIGHT RANKING DATA (CSV Export Version)")
        print("=" * 80)

        # Get all searches (oldest first, so newest appears at bottom)
        searches = db.query(Search).order_by(Search.created_at.asc()).all()

        if not searches:
            print("\nNo data saved yet. Run the app and submit some rankings first!")
            return

        print(f"\nTotal searches: {len(searches)}\n")

        for search in searches:
            print("-" * 80)
            print(f"Search ID: {search.search_id}")
            print(f"Session ID: {search.session_id}")
            print(f"Date: {search.created_at}")
            print(f"\nUser Prompt:")
            print(f"  {search.user_prompt}")
            print(f"\nParsed Parameters:")
            print(f"  Origins: {search.parsed_origins}")
            print(f"  Destinations: {search.parsed_destinations}")
            print(f"  Departure Date: {search.departure_date}")
            print(f"  Preferences: {search.parsed_preferences}")

            # Get CSV exports for this search
            csvs = db.query(FlightCSV).filter(FlightCSV.search_id == search.search_id).all()

            if csvs:
                print(f"\n✓ CSV Exports: {len(csvs)}")
                for idx, csv in enumerate(csvs, 1):
                    print(f"\n  CSV #{idx} (ID: {csv.id}):")
                    print(f"    - Total flights: {csv.num_flights}")
                    print(f"    - Selected (top 5): {csv.num_selected}")
                    print(f"    - Created: {csv.created_at}")

                    # Parse CSV and show top 5
                    try:
                        df = pd.read_csv(io.StringIO(csv.csv_data), sep='\t')
                        selected = df[df['is_best'] == True].sort_values('rank')

                        if len(selected) > 0:
                            print(f"\n    Top 5 Selected Flights:")
                            for _, row in selected.iterrows():
                                print(f"      #{int(row['rank'])}: {row['name']} - {row['origin']}→{row['destination']}")
                                print(f"           ${row['price']:.0f}, {row['duration']}, {int(row['stops'])} stops")
                    except Exception as e:
                        print(f"    Error parsing CSV: {e}")
            else:
                print("\n  No CSV exports found")

            print()

    finally:
        db.close()


def view_latest():
    """Display just the most recent search."""
    db = SessionLocal()

    try:
        search = db.query(Search).order_by(Search.created_at.desc()).first()

        if not search:
            print("No data saved yet.")
            return

        print("=" * 80)
        print("LATEST SEARCH")
        print("=" * 80)
        print(f"\nSearch ID: {search.search_id}")
        print(f"Prompt: {search.user_prompt}")

        csvs = db.query(FlightCSV).filter(FlightCSV.search_id == search.search_id).all()

        if csvs:
            for idx, csv in enumerate(csvs, 1):
                print(f"\nCSV #{idx}: {csv.num_flights} flights, {csv.num_selected} selected")

                # Parse and show top 5
                try:
                    df = pd.read_csv(io.StringIO(csv.csv_data), sep='\t')
                    selected = df[df['is_best'] == True].sort_values('rank')

                    if len(selected) > 0:
                        print(f"\nTop 5 Rankings:")
                        for _, row in selected.iterrows():
                            print(f"  #{int(row['rank'])}: {row['name']} - {row['origin']}→{row['destination']} (${row['price']:.0f})")
                except Exception as e:
                    print(f"Error parsing CSV: {e}")
        print()

    finally:
        db.close()


def export_csv(search_id: int, output_dir: str = "."):
    """Export CSV data to files."""
    import os
    from datetime import datetime

    db = SessionLocal()
    exported_files = []

    try:
        csvs = db.query(FlightCSV).filter(FlightCSV.search_id == search_id).all()

        if not csvs:
            print(f"No CSV data found for search ID {search_id}")
            return exported_files

        for idx, csv in enumerate(csvs, 1):
            filename = f"search_{search_id}_csv_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w') as f:
                f.write(csv.csv_data)

            print(f"✓ Exported CSV #{idx} to: {filepath}")
            exported_files.append(filepath)

        return exported_files

    finally:
        db.close()


def open_csv(search_id: int, output_dir: str = "."):
    """Export and automatically open CSV files for a search."""
    import subprocess
    import platform

    # Export the CSV files
    filepaths = export_csv(search_id, output_dir)

    if not filepaths:
        return

    # Open each file with the default application
    for filepath in filepaths:
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', filepath])
            elif platform.system() == 'Windows':
                subprocess.run(['start', filepath], shell=True)
            else:  # Linux
                subprocess.run(['xdg-open', filepath])

            print(f"✓ Opened: {filepath}")
        except Exception as e:
            print(f"✗ Could not open {filepath}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "latest":
            view_latest()
        elif sys.argv[1] == "export" and len(sys.argv) > 2:
            search_id = int(sys.argv[2])
            output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
            export_csv(search_id, output_dir)
        elif sys.argv[1] == "open" and len(sys.argv) > 2:
            search_id = int(sys.argv[2])
            output_dir = sys.argv[3] if len(sys.argv) > 3 else "."
            open_csv(search_id, output_dir)
        else:
            print("Usage:")
            print("  python view_data.py              # View all searches")
            print("  python view_data.py latest       # View latest search")
            print("  python view_data.py export <id>  # Export CSV for search ID")
            print("  python view_data.py open <id>    # Export and open CSV for search ID")
    else:
        view_all_data()