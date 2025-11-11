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

# Database connection
DB_TYPE = os.getenv('DB_TYPE', 'sqlite')  # 'sqlite' or 'mysql'

if DB_TYPE == 'mysql':
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = os.getenv('MYSQL_PORT', '3306')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'flight_rankings')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
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