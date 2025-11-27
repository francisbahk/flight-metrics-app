"""
Database functions for storing flight searches and user rankings.
Uses SQLAlchemy ORM with MySQL.
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, DateTime, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()

# Helper function to get config from Streamlit secrets or environment
def get_config(key, default=''):
    """Get config from Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except (ImportError, FileNotFoundError, AttributeError):
        return os.getenv(key, default)

# Database connection
DB_TYPE = get_config('DB_TYPE', 'sqlite')  # 'sqlite' or 'mysql'

if DB_TYPE == 'mysql':
    MYSQL_HOST = get_config('MYSQL_HOST', 'localhost')
    MYSQL_PORT = get_config('MYSQL_PORT', '3306')
    MYSQL_DATABASE = get_config('MYSQL_DATABASE', 'flight_rankings')
    MYSQL_USER = get_config('MYSQL_USER', 'root')
    MYSQL_PASSWORD = get_config('MYSQL_PASSWORD', '')
    DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
else:
    # Use SQLite (file-based, no installation required)
    DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'flight_rankings.db')
    DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # Disable connection pooling for simplicity
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Models
class Search(Base):
    """Stores each search query."""
    __tablename__ = 'searches'

    search_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    user_prompt = Column(Text, nullable=False)
    parsed_origins = Column(JSON)
    parsed_destinations = Column(JSON)
    parsed_preferences = Column(JSON)
    departure_date = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    flights_shown = relationship("FlightShown", back_populates="search", cascade="all, delete-orphan")
    rankings = relationship("UserRanking", back_populates="search", cascade="all, delete-orphan")


class FlightShown(Base):
    """Stores all flights shown to user."""
    __tablename__ = 'flights_shown'

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(Integer, ForeignKey('searches.search_id', ondelete='CASCADE'), nullable=False, index=True)
    flight_data = Column(JSON, nullable=False)  # Full flight object
    algorithm = Column(String(50), nullable=False)  # 'Cheapest', 'Fastest', 'LISTEN-U'
    algorithm_rank = Column(Integer, nullable=False)  # Position in that algorithm (1-10)
    display_position = Column(Integer, nullable=False)  # Position shown to user (1-30)

    # Relationships
    search = relationship("Search", back_populates="flights_shown")
    rankings = relationship("UserRanking", back_populates="flight", cascade="all, delete-orphan")


class UserRanking(Base):
    """Stores user's top 5 ranked flights."""
    __tablename__ = 'user_rankings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(Integer, ForeignKey('searches.search_id', ondelete='CASCADE'), nullable=False, index=True)
    flight_id = Column(Integer, ForeignKey('flights_shown.id', ondelete='CASCADE'), nullable=False)
    user_rank = Column(Integer, nullable=False)  # User's ranking (1-5)
    submitted_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    search = relationship("Search", back_populates="rankings")
    flight = relationship("FlightShown", back_populates="rankings")


class LILOSession(Base):
    """Stores LILO refinement sessions."""
    __tablename__ = 'lilo_sessions'

    session_id = Column(String(255), primary_key=True)
    search_id = Column(Integer, ForeignKey('searches.search_id', ondelete='CASCADE'), nullable=True, index=True)
    user_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    num_rounds = Column(Integer, default=0)
    final_utility_scores = Column(JSON, nullable=True)  # Final utilities for all flights
    feedback_summary = Column(Text, nullable=True)  # Summarized preferences

    # Relationships
    rounds = relationship("LILORound", back_populates="session", cascade="all, delete-orphan")


class LILORound(Base):
    """Stores each round of LILO refinement."""
    __tablename__ = 'lilo_rounds'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey('lilo_sessions.session_id', ondelete='CASCADE'), nullable=False, index=True)
    round_number = Column(Integer, nullable=False)
    flights_shown = Column(JSON, nullable=False)  # List of flight IDs/indices shown this round
    user_rankings = Column(JSON, nullable=False)  # User's top-k ranking for this round
    user_feedback = Column(Text, nullable=False)  # Natural language feedback
    generated_questions = Column(JSON, nullable=True)  # Questions asked by LLM
    extracted_preferences = Column(JSON, nullable=True)  # LLM-extracted preference signals
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("LILOSession", back_populates="rounds")


class EvaluationSession(Base):
    """Stores human vs LLM evaluation experiments."""
    __tablename__ = 'evaluation_sessions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    search_id = Column(Integer, ForeignKey('searches.search_id', ondelete='CASCADE'), nullable=True)

    # Person A (ground truth provider)
    person_a_user_id = Column(String(255), nullable=True)
    person_a_prompt = Column(Text, nullable=False)
    person_a_rankings = Column(JSON, nullable=False)  # Ground truth top-k

    # Person B (human recommender)
    person_b_user_id = Column(String(255), nullable=True)
    person_b_rankings = Column(JSON, nullable=True)  # Person B's guesses

    # Algorithm rankings
    listen_u_rankings = Column(JSON, nullable=True)  # LISTEN-U recommendations
    lilo_rankings = Column(JSON, nullable=True)  # LILO recommendations
    cheapest_rankings = Column(JSON, nullable=True)  # Cheapest algorithm
    fastest_rankings = Column(JSON, nullable=True)  # Fastest algorithm

    # Comparison results
    team_draft_results = Column(JSON, nullable=True)  # Who won each comparison
    metrics = Column(JSON, nullable=True)  # NDCG, Precision@k, etc.

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class InteractionEvent(Base):
    """Stores user interaction events for offline analysis."""
    __tablename__ = 'interaction_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    search_id = Column(Integer, ForeignKey('searches.search_id', ondelete='CASCADE'), nullable=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # 'flight_click', 'flight_view', etc.
    event_data = Column(JSON, nullable=False)  # Flexible: flight_id, timestamp, etc.
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class SequentialEvaluation(Base):
    """Stores sequential evaluation workflow: Manual → LISTEN → LILO."""
    __tablename__ = 'sequential_evaluations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), nullable=True)

    # Manual method (baseline)
    manual_search_results = Column(JSON, nullable=True)  # All flights from Amadeus
    manual_rankings = Column(JSON, nullable=True)  # Top 5 rankings
    manual_completed_at = Column(DateTime, nullable=True)

    # LISTEN method
    listen_prompt = Column(Text, nullable=True)  # Natural language prompt
    listen_search_results = Column(JSON, nullable=True)  # All flights from Amadeus
    listen_ranked_flights = Column(JSON, nullable=True)  # LISTEN-U ranked flights
    listen_rankings = Column(JSON, nullable=True)  # Top 5 rankings
    listen_completed_at = Column(DateTime, nullable=True)

    # LILO method (3 iterations)
    lilo_prompt = Column(Text, nullable=True)  # Natural language prompt
    lilo_search_results = Column(JSON, nullable=True)  # All flights from Amadeus
    lilo_initial_answers = Column(JSON, nullable=True)  # Answers to Prompt 1 questions
    lilo_iteration1_flights = Column(JSON, nullable=True)  # 15 random flights
    lilo_iteration1_feedback = Column(Text, nullable=True)  # User feedback iteration 1
    lilo_iteration2_flights = Column(JSON, nullable=True)  # 15 utility-ranked flights
    lilo_iteration2_feedback = Column(Text, nullable=True)  # User feedback iteration 2
    lilo_iteration3_flights = Column(JSON, nullable=True)  # 15 final utility-ranked flights
    lilo_rankings = Column(JSON, nullable=True)  # Top 5 final rankings
    lilo_completed_at = Column(DateTime, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Analysis/comparison results
    comparison_metrics = Column(JSON, nullable=True)  # Overlap, NDCG, etc.


class FlightCSV(Base):
    """Stores CSV exports for each user session."""
    __tablename__ = 'flight_csvs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    search_id = Column(Integer, ForeignKey('searches.search_id', ondelete='CASCADE'), nullable=True, index=True)
    csv_data = Column(Text, nullable=False)  # The actual CSV content
    num_flights = Column(Integer, nullable=False)  # Total number of flights
    num_selected = Column(Integer, nullable=False)  # Number of selected flights (should be 5)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# Database functions
def init_db():
    """
    Create all tables in the database.
    Call this once to set up the schema.
    """
    Base.metadata.create_all(bind=engine)
    db_name = "SQLite" if DB_TYPE == 'sqlite' else MYSQL_DATABASE
    print(f"✓ Database tables created in {db_name}")


def save_search_and_rankings(
    session_id: str,
    user_prompt: str,
    parsed_params: Dict,
    interleaved_results: List[Dict],
    user_shortlist: List[Dict]
) -> int:
    """
    Save a complete search session with results and user rankings.

    Args:
        session_id: Unique session identifier
        user_prompt: Original natural language query
        parsed_params: Parsed search parameters from LLM
        interleaved_results: All 30 flights shown (with algorithm labels)
        user_shortlist: User's top 5 selected flights (in ranked order)

    Returns:
        search_id of the created search record
    """
    db = SessionLocal()

    try:
        # Create search record
        search = Search(
            session_id=session_id,
            user_prompt=user_prompt,
            parsed_origins=parsed_params.get('origins'),
            parsed_destinations=parsed_params.get('destinations'),
            parsed_preferences=parsed_params.get('preferences'),
            departure_date=parsed_params.get('departure_date')
        )
        db.add(search)
        db.flush()  # Get search_id

        search_id = search.search_id

        # Save all flights shown
        flight_id_map = {}  # Map (algorithm, rank) -> flight_id

        for position, item in enumerate(interleaved_results, 1):
            flight_shown = FlightShown(
                search_id=search_id,
                flight_data=item['flight'],
                algorithm=item['algorithm'],
                algorithm_rank=item['rank'],
                display_position=position
            )
            db.add(flight_shown)
            db.flush()

            # Store mapping for later lookup
            key = (item['algorithm'], item['rank'])
            flight_id_map[key] = flight_shown.id

        # Save user rankings
        for user_rank, shortlist_item in enumerate(user_shortlist, 1):
            # Find the corresponding flight_id
            # shortlist_item has 'key', 'flight', 'algorithm', 'rank'
            algo = shortlist_item['algorithm']
            algo_rank = shortlist_item['rank']
            key = (algo, algo_rank)

            flight_id = flight_id_map.get(key)

            if flight_id:
                ranking = UserRanking(
                    search_id=search_id,
                    flight_id=flight_id,
                    user_rank=user_rank
                )
                db.add(ranking)
            else:
                print(f"Warning: Could not find flight_id for {key}")

        db.commit()
        print(f"✓ Saved search {search_id} with {len(user_shortlist)} rankings")
        return search_id

    except Exception as e:
        db.rollback()
        print(f"✗ Error saving to database: {str(e)}")
        raise

    finally:
        db.close()


def get_algorithm_stats() -> Dict:
    """
    Get statistics on algorithm performance.

    Returns:
        Dict with algorithm stats
    """
    db = SessionLocal()

    try:
        from sqlalchemy import func

        # Count how many times each algorithm appears in user's top 5
        stats = db.query(
            FlightShown.algorithm,
            func.count(UserRanking.id).label('times_selected'),
            func.avg(UserRanking.user_rank).label('avg_rank')
        ).join(UserRanking).group_by(FlightShown.algorithm).all()

        result = {}
        for algo, count, avg_rank in stats:
            result[algo] = {
                'times_selected': count,
                'avg_rank': float(avg_rank) if avg_rank else None
            }

        return result

    finally:
        db.close()


def save_search_and_csv(
    session_id: str,
    user_prompt: str,
    parsed_params: Dict,
    all_flights: List[Dict],
    selected_flights: List[Dict],
    csv_data: str
) -> int:
    """
    Save a search session with CSV export (new simplified version).

    Args:
        session_id: Unique session identifier
        user_prompt: Original natural language query
        parsed_params: Parsed search parameters from LLM
        all_flights: All flights returned by Amadeus
        selected_flights: User's top 5 selected flights (in ranked order)
        csv_data: Generated CSV content

    Returns:
        search_id of the created search record
    """
    db = SessionLocal()

    try:
        # Create search record
        search = Search(
            session_id=session_id,
            user_prompt=user_prompt,
            parsed_origins=parsed_params.get('origins'),
            parsed_destinations=parsed_params.get('destinations'),
            parsed_preferences=parsed_params.get('preferences'),
            departure_date=parsed_params.get('departure_date')
        )
        db.add(search)
        db.flush()  # Get search_id

        search_id = search.search_id

        # Save CSV data
        csv_record = FlightCSV(
            session_id=session_id,
            search_id=search_id,
            csv_data=csv_data,
            num_flights=len(all_flights),
            num_selected=len(selected_flights)
        )
        db.add(csv_record)

        db.commit()
        print(f"✓ Saved search {search_id} with CSV export ({len(all_flights)} flights, {len(selected_flights)} selected)")
        return search_id

    except Exception as e:
        db.rollback()
        print(f"✗ Error saving to database: {str(e)}")
        raise

    finally:
        db.close()


def test_connection() -> bool:
    """
    Test database connection.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        print(f"✓ Database connection successful: {DATABASE_URL}")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {str(e)}")
        return False


# For testing
if __name__ == "__main__":
    print("Testing database connection...")
    if test_connection():
        print("\nCreating tables...")
        init_db()
        print("\nDatabase setup complete!")
    else:
        print("\n✗ Database setup failed. Check your .env configuration:")
        print(f"  MYSQL_HOST={MYSQL_HOST}")
        print(f"  MYSQL_PORT={MYSQL_PORT}")
        print(f"  MYSQL_DATABASE={MYSQL_DATABASE}")
        print(f"  MYSQL_USER={MYSQL_USER}")
        print(f"  MYSQL_PASSWORD={'***' if MYSQL_PASSWORD else '(empty)'}")