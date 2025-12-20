"""
Simple script to view saved flight ranking data with CSV exports.
"""
from backend.db import SessionLocal, Search, FlightCSV, SurveyResponse, CrossValidation
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
            print(f"Completion Token: {search.completion_token or '(not provided)'}")
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
        print(f"Completion Token: {search.completion_token or '(not provided)'}")
        print(f"Prompt: {search.user_prompt}")

        csvs = db.query(FlightCSV).filter(FlightCSV.search_id == search.search_id).all()

        if csvs:
            for idx, csv in enumerate(csvs, 1):
                print(f"\nCSV #{idx}: {csv.num_flights} flights, {csv.num_selected} selected")

                # Parse and show top 5
                try:
                    df = pd.read_csv(io.StringIO(csv.csv_data), sep='\t')
                    selected = df[df['is_best'] == 1].sort_values('rank')  # Changed to 1 (was True)

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


def view_survey_responses():
    """Display all survey responses."""
    db = SessionLocal()

    try:
        print("=" * 80)
        print("SURVEY RESPONSES")
        print("=" * 80)

        # Get all survey responses (oldest first, so newest appears at bottom)
        surveys = db.query(SurveyResponse).order_by(SurveyResponse.created_at.asc()).all()

        if not surveys:
            print("\nNo survey responses yet. Wait for users to complete the survey!")
            return

        print(f"\nTotal responses: {len(surveys)}\n")

        for survey in surveys:
            print("-" * 80)
            print(f"Response ID: {survey.id}")
            print(f"Session ID: {survey.session_id}")
            print(f"Completion Token: {survey.completion_token or '(not provided)'}")
            print(f"Date: {survey.created_at}")

            print(f"\n📊 Satisfaction & Usability:")
            print(f"  Q1. Satisfaction: {survey.satisfaction}/5" if survey.satisfaction else "  Q1. Satisfaction: (not answered)")
            print(f"  Q2. Ease of use: {survey.ease_of_use}/5" if survey.ease_of_use else "  Q2. Ease of use: (not answered)")
            print(f"  Q3. Technical issues: {survey.encountered_issues}")
            if survey.issues_description:
                print(f"      Description: {survey.issues_description}")

            print(f"\n🎯 Feature Value:")
            print(f"  Q4. Search method: {survey.search_method}")
            print(f"  Q5. Understood AI ranking: {survey.understood_ranking}/5" if survey.understood_ranking else "  Q5. Understood AI ranking: (not answered)")
            print(f"  Q6. Helpful features: {survey.helpful_features}")
            print(f"  Q7. Flights matched expectations: {survey.flights_matched}/5" if survey.flights_matched else "  Q7. Flights matched expectations: (not answered)")

            print(f"\n⚠️ Friction & Missing:")
            if survey.confusing_frustrating:
                print(f"  Q8. Confusing/frustrating: {survey.confusing_frustrating}")
            else:
                print(f"  Q8. Confusing/frustrating: (no response)")

            if survey.missing_features:
                print(f"  Q9. Missing features: {survey.missing_features}")
            else:
                print(f"  Q9. Missing features: (no response)")

            print(f"\n🔄 Future Usage:")
            print(f"  Q10. Would use again: {survey.would_use_again}")
            if survey.would_use_again_reason:
                print(f"       Reason: {survey.would_use_again_reason}")

            print(f"\n📈 Comparison:")
            print(f"  Q11. Compared to others: {survey.compared_to_others}/5" if survey.compared_to_others else "  Q11. Compared to others: (not answered)")

            if survey.additional_comments:
                print(f"\n💬 Additional Comments:")
                print(f"  {survey.additional_comments}")

            print()

    finally:
        db.close()


def view_survey_summary():
    """Display summary statistics of survey responses."""
    db = SessionLocal()

    try:
        surveys = db.query(SurveyResponse).all()

        if not surveys:
            print("No survey responses yet.")
            return

        print("=" * 80)
        print("SURVEY SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nTotal Responses: {len(surveys)}\n")

        # Calculate averages for Likert scales
        satisfactions = [s.satisfaction for s in surveys if s.satisfaction is not None]
        ease_of_uses = [s.ease_of_use for s in surveys if s.ease_of_use is not None]
        understandings = [s.understood_ranking for s in surveys if s.understood_ranking is not None]
        matches = [s.flights_matched for s in surveys if s.flights_matched is not None]
        comparisons = [s.compared_to_others for s in surveys if s.compared_to_others is not None]

        print("📊 Average Ratings (1-5 scale):")
        if satisfactions:
            print(f"  Satisfaction: {sum(satisfactions)/len(satisfactions):.2f} (n={len(satisfactions)})")
        if ease_of_uses:
            print(f"  Ease of use: {sum(ease_of_uses)/len(ease_of_uses):.2f} (n={len(ease_of_uses)})")
        if understandings:
            print(f"  Understood AI ranking: {sum(understandings)/len(understandings):.2f} (n={len(understandings)})")
        if matches:
            print(f"  Flights matched expectations: {sum(matches)/len(matches):.2f} (n={len(matches)})")
        if comparisons:
            print(f"  Compared to others: {sum(comparisons)/len(comparisons):.2f} (n={len(comparisons)})")

        # Count technical issues
        issues_yes = len([s for s in surveys if s.encountered_issues == "Yes"])
        issues_no = len([s for s in surveys if s.encountered_issues == "No"])
        print(f"\n⚠️ Technical Issues:")
        print(f"  Yes: {issues_yes} ({100*issues_yes/len(surveys):.1f}%)")
        print(f"  No: {issues_no} ({100*issues_no/len(surveys):.1f}%)")

        # Search method breakdown
        from collections import Counter
        methods = Counter([s.search_method for s in surveys if s.search_method])
        print(f"\n🔍 Search Method Used:")
        for method, count in methods.items():
            print(f"  {method}: {count} ({100*count/len(surveys):.1f}%)")

        # Would use again breakdown
        use_again = Counter([s.would_use_again for s in surveys if s.would_use_again])
        print(f"\n🔄 Would Use Again:")
        for response, count in use_again.items():
            print(f"  {response}: {count} ({100*count/len(surveys):.1f}%)")

        # Most helpful features
        all_features = []
        for s in surveys:
            if s.helpful_features:
                all_features.extend(s.helpful_features)

        if all_features:
            feature_counts = Counter(all_features)
            print(f"\n⭐ Most Helpful Features:")
            for feature, count in feature_counts.most_common():
                print(f"  {feature}: {count} ({100*count/len(surveys):.1f}%)")

        print()

    finally:
        db.close()


def view_cross_validations():
    """Display all cross-validation responses."""
    db = SessionLocal()

    try:
        print("=" * 80)
        print("CROSS-VALIDATION DATA")
        print("=" * 80)

        # Get all cross-validations (oldest first, so newest appears at bottom)
        cross_vals = db.query(CrossValidation).order_by(CrossValidation.created_at.asc()).all()

        if not cross_vals:
            print("\nNo cross-validations yet. Users need to complete searches first!")
            return

        print(f"\nTotal cross-validations: {len(cross_vals)}\n")

        for cv in cross_vals:
            print("-" * 80)
            print(f"Cross-Validation ID: {cv.id}")
            print(f"Date: {cv.created_at}")
            print(f"\nReviewer: {cv.reviewer_session_id}")
            print(f"Reviewer Token: {cv.reviewer_token or '(not provided)'}")
            print(f"\nReviewed User: {cv.reviewed_session_id}")
            print(f"Reviewed Search ID: {cv.reviewed_search_id}")

            print(f"\n📝 Original User's Prompt:")
            print(f"  {cv.reviewed_prompt}")

            print(f"\n✈️ Total Flights Shown: {len(cv.reviewed_flights_json)}")
            print(f"✅ Reviewer Selected: {len(cv.selected_flight_ids)} flights")

            if cv.selected_flights_data:
                print(f"\n🏆 Reviewer's Top 5 Selections:")
                for idx, flight in enumerate(cv.selected_flights_data, 1):
                    airline = flight.get('airline', 'Unknown')
                    flight_num = flight.get('flight_number', '')
                    origin = flight.get('origin', '')
                    destination = flight.get('destination', '')
                    price = flight.get('price', 0)
                    print(f"  {idx}. {airline} {flight_num} | {origin} → {destination} | ${price:.0f}")

            print()

    finally:
        db.close()


def view_cross_validation_summary():
    """Display summary statistics of cross-validations."""
    db = SessionLocal()

    try:
        cross_vals = db.query(CrossValidation).all()

        if not cross_vals:
            print("No cross-validations yet.")
            return

        print("=" * 80)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 80)
        print(f"\nTotal Cross-Validations: {len(cross_vals)}\n")

        # Count unique reviewers and reviewed users
        unique_reviewers = len(set(cv.reviewer_session_id for cv in cross_vals))
        unique_reviewed = len(set(cv.reviewed_session_id for cv in cross_vals))

        print(f"📊 Participation:")
        print(f"  Unique reviewers: {unique_reviewers}")
        print(f"  Unique users reviewed: {unique_reviewed}")
        print(f"  Avg validations per reviewer: {len(cross_vals) / unique_reviewers:.2f}")

        # Calculate average number of flights shown
        avg_flights_shown = sum(len(cv.reviewed_flights_json) for cv in cross_vals) / len(cross_vals)
        print(f"\n✈️ Flight Data:")
        print(f"  Avg flights shown per validation: {avg_flights_shown:.1f}")
        print(f"  Total validation selections: {len(cross_vals) * 5}")

        print()

    finally:
        db.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "latest":
            view_latest()
        elif sys.argv[1] == "survey":
            view_survey_responses()
        elif sys.argv[1] == "survey-summary":
            view_survey_summary()
        elif sys.argv[1] == "cross-validation":
            view_cross_validations()
        elif sys.argv[1] == "cross-validation-summary":
            view_cross_validation_summary()
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
            print("  python view_data.py                       # View all searches")
            print("  python view_data.py latest                # View latest search")
            print("  python view_data.py survey                # View all survey responses")
            print("  python view_data.py survey-summary        # View survey statistics")
            print("  python view_data.py cross-validation      # View all cross-validations")
            print("  python view_data.py cross-validation-summary  # View CV statistics")
            print("  python view_data.py export <id>           # Export CSV for search ID")
            print("  python view_data.py open <id>             # Export and open CSV for search ID")
    else:
        view_all_data()