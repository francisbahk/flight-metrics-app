"""
Seed script to add dummy prompts for cross-validation queues.
Run this once to initialize the DEMO and DATA cross-validation queues.
"""

from backend.db import SessionLocal, Search, LILOSession
from datetime import datetime
import uuid

# Dummy prompt for cross-validation
SEED_PROMPT = """I want to fly from JFK to LAX on January 30th. I prefer cheap flights but if a flight is longer than 12 hours I'd prefer to pay a bit more money to take a shorter flight. I also want to arrive in the afternoon since I want to have time to eat dinner"""

# Dummy flights for cross-validation (realistic sample data)
SEED_FLIGHTS = [
    {
        "id": "seed_flight_1",
        "airline": "Delta Air Lines",
        "flight_number": "DL123",
        "origin": "JFK",
        "destination": "LAX",
        "departure_time": "2025-01-30T08:00:00",
        "arrival_time": "2025-01-30T11:30:00",
        "duration_min": 330,
        "stops": 0,
        "price": 289
    },
    {
        "id": "seed_flight_2",
        "airline": "American Airlines",
        "flight_number": "AA456",
        "origin": "JFK",
        "destination": "LAX",
        "departure_time": "2025-01-30T10:00:00",
        "arrival_time": "2025-01-30T14:00:00",
        "duration_min": 360,
        "stops": 0,
        "price": 245
    },
    {
        "id": "seed_flight_3",
        "airline": "United Airlines",
        "flight_number": "UA789",
        "origin": "JFK",
        "destination": "LAX",
        "departure_time": "2025-01-30T06:00:00",
        "arrival_time": "2025-01-30T12:00:00",
        "duration_min": 480,
        "stops": 1,
        "price": 199
    },
    {
        "id": "seed_flight_4",
        "airline": "JetBlue Airways",
        "flight_number": "B6321",
        "origin": "JFK",
        "destination": "LAX",
        "departure_time": "2025-01-30T14:00:00",
        "arrival_time": "2025-01-30T17:30:00",
        "duration_min": 330,
        "stops": 0,
        "price": 319
    },
    {
        "id": "seed_flight_5",
        "airline": "Spirit Airlines",
        "flight_number": "NK555",
        "origin": "JFK",
        "destination": "LAX",
        "departure_time": "2025-01-30T05:00:00",
        "arrival_time": "2025-01-30T14:00:00",
        "duration_min": 660,
        "stops": 2,
        "price": 149
    }
]


def seed_cross_validation_data():
    """Add seed data for DEMO and DATA cross-validation queues."""
    db = SessionLocal()

    try:
        # Check if seed data already exists
        existing_demo = db.query(Search).filter(
            Search.completion_token == "DEMO",
            Search.session_id.like("seed_%")
        ).first()

        existing_data = db.query(Search).filter(
            Search.completion_token == "DATA",
            Search.session_id.like("seed_%")
        ).first()

        if existing_demo and existing_data:
            print("✓ Seed data already exists for both DEMO and DATA")
            return

        # Add DEMO seed if not exists
        if not existing_demo:
            demo_session_id = f"seed_demo_{uuid.uuid4().hex[:8]}"
            demo_search = Search(
                session_id=demo_session_id,
                completion_token="DEMO",
                user_prompt=SEED_PROMPT,
                parsed_origins=["JFK"],
                parsed_destinations=["LAX"],
                parsed_preferences={"price": "cheap", "duration": "prefer shorter"},
                departure_date="2025-01-30",
                listen_ranked_flights_json=SEED_FLIGHTS,
                created_at=datetime.utcnow()
            )
            db.add(demo_search)
            db.flush()  # Get the search_id

            # Add completed LILO session for DEMO
            demo_lilo = LILOSession(
                session_id=demo_session_id,
                search_id=demo_search.search_id,
                completion_token="DEMO",
                num_iterations=2,
                questions_per_round=3,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()  # Mark as completed
            )
            db.add(demo_lilo)
            print(f"✓ Added DEMO seed data (session: {demo_session_id})")

        # Add DATA seed if not exists
        if not existing_data:
            data_session_id = f"seed_data_{uuid.uuid4().hex[:8]}"
            data_search = Search(
                session_id=data_session_id,
                completion_token="DATA",
                user_prompt=SEED_PROMPT,
                parsed_origins=["JFK"],
                parsed_destinations=["LAX"],
                parsed_preferences={"price": "cheap", "duration": "prefer shorter"},
                departure_date="2025-01-30",
                listen_ranked_flights_json=SEED_FLIGHTS,
                created_at=datetime.utcnow()
            )
            db.add(data_search)
            db.flush()

            # Add completed LILO session for DATA
            data_lilo = LILOSession(
                session_id=data_session_id,
                search_id=data_search.search_id,
                completion_token="DATA",
                num_iterations=2,
                questions_per_round=3,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
            db.add(data_lilo)
            print(f"✓ Added DATA seed data (session: {data_session_id})")

        db.commit()
        print("✓ Cross-validation seed data committed successfully")

    except Exception as e:
        db.rollback()
        print(f"✗ Error seeding data: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_cross_validation_data()
