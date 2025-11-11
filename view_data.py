"""
Simple script to view saved flight ranking data.
"""
from backend.db import SessionLocal, Search, FlightShown, UserRanking
from sqlalchemy import func

def view_all_data():
    """Display all saved searches and rankings."""
    db = SessionLocal()

    try:
        print("=" * 80)
        print("FLIGHT RANKING DATA")
        print("=" * 80)

        # Get all searches
        searches = db.query(Search).order_by(Search.created_at.desc()).all()

        if not searches:
            print("\nNo data saved yet. Run the app and submit some rankings first!")
            return

        print(f"\nTotal searches: {len(searches)}\n")

        for search in searches:
            print("-" * 80)
            print(f"Search ID: {search.search_id}")
            print(f"Session ID: {search.session_id}")
            print(f"Date: {search.created_at}")
            print(f"\nUser Prompt: {search.user_prompt}")
            print(f"\nOrigins: {search.parsed_origins}")
            print(f"Destinations: {search.parsed_destinations}")
            print(f"Departure Date: {search.departure_date}")

            # Count flights shown
            flights_count = db.query(func.count(FlightShown.id))\
                .filter(FlightShown.search_id == search.search_id)\
                .scalar()
            print(f"\nFlights shown: {flights_count}")

            # Get user rankings
            rankings = db.query(UserRanking, FlightShown)\
                .join(FlightShown, UserRanking.flight_id == FlightShown.id)\
                .filter(UserRanking.search_id == search.search_id)\
                .order_by(UserRanking.user_rank)\
                .all()

            if rankings:
                print(f"\nUser's Top {len(rankings)} Rankings:")
                for ranking, flight in rankings:
                    flight_info = flight.flight_data

                    # Handle different price formats
                    if isinstance(flight_info.get('price'), dict):
                        price = flight_info.get('price', {}).get('total', 'N/A')
                    else:
                        price = flight_info.get('price', 'N/A')

                    # Handle duration (convert minutes to hours:minutes)
                    duration = 'N/A'
                    if 'duration_min' in flight_info:
                        duration_min = flight_info['duration_min']
                        hours = int(duration_min // 60)
                        minutes = int(duration_min % 60)
                        duration = f"{hours}h {minutes}m"
                    elif 'itineraries' in flight_info and flight_info['itineraries']:
                        duration = flight_info['itineraries'][0].get('duration', 'N/A')
                    elif 'duration' in flight_info:
                        duration = flight_info.get('duration', 'N/A')

                    print(f"  #{ranking.user_rank}: {flight.algorithm} (rank {flight.algorithm_rank})")
                    print(f"       Price: ${price}, Duration: {duration}")
            else:
                print("\nNo rankings submitted yet.")

            print()

        print("=" * 80)
        print("\nALGORITHM PERFORMANCE SUMMARY")
        print("=" * 80)

        # Get algorithm stats
        stats = db.query(
            FlightShown.algorithm,
            func.count(UserRanking.id).label('times_selected'),
            func.avg(UserRanking.user_rank).label('avg_rank')
        ).join(UserRanking)\
         .group_by(FlightShown.algorithm)\
         .order_by(func.count(UserRanking.id).desc())\
         .all()

        if stats:
            print(f"\n{'Algorithm':<15} {'Times Selected':<18} {'Avg Rank':<10}")
            print("-" * 50)
            for algo, count, avg_rank in stats:
                print(f"{algo:<15} {count:<18} {avg_rank:.2f}")
        else:
            print("\nNo ranking data available yet.")

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

        rankings = db.query(UserRanking, FlightShown)\
            .join(FlightShown, UserRanking.flight_id == FlightShown.id)\
            .filter(UserRanking.search_id == search.search_id)\
            .order_by(UserRanking.user_rank)\
            .all()

        if rankings:
            print(f"\nUser Rankings:")
            for ranking, flight in rankings:
                print(f"  #{ranking.user_rank}: {flight.algorithm} (algorithm rank {flight.algorithm_rank})")
        print()

    finally:
        db.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "latest":
        view_latest()
    else:
        view_all_data()