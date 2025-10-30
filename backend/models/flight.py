"""
SQLAlchemy ORM models for flight metrics application.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.sql import func
from database import Base


class Flight(Base):
    """Flight offers from Amadeus API with computed metrics."""

    __tablename__ = "flights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=True)
    origin = Column(String(3), nullable=False)
    destination = Column(String(3), nullable=False)
    departure_time = Column(DateTime, nullable=False)
    arrival_time = Column(DateTime, nullable=False)
    duration = Column(String(20), nullable=False)
    stops = Column(Integer, nullable=False, default=0)
    price = Column(Float, nullable=False)
    dis_from_origin = Column(Float, nullable=True)
    dis_from_dest = Column(Float, nullable=True)
    departure_seconds = Column(Integer, nullable=True)
    arrival_seconds = Column(Integer, nullable=True)
    duration_min = Column(Float, nullable=True)
    date_retrieved = Column(DateTime, default=func.now())
    raw_data = Column(JSON, nullable=True)

    def to_dict(self):
        """Convert model to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "origin": self.origin,
            "destination": self.destination,
            "departure_time": self.departure_time.isoformat() if self.departure_time else None,
            "arrival_time": self.arrival_time.isoformat() if self.arrival_time else None,
            "duration": self.duration,
            "stops": self.stops,
            "price": self.price,
            "dis_from_origin": self.dis_from_origin,
            "dis_from_dest": self.dis_from_dest,
            "departure_seconds": self.departure_seconds,
            "arrival_seconds": self.arrival_seconds,
            "duration_min": self.duration_min,
            "date_retrieved": self.date_retrieved.isoformat() if self.date_retrieved else None,
            "raw_data": self.raw_data,
        }


class ListenRanking(Base):
    """LISTEN evaluation rankings submitted by users."""

    __tablename__ = "listen_rankings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    prompt = Column(Text, nullable=False)
    flight_ids = Column(JSON, nullable=False)  # Original list of flight IDs
    user_ranking = Column(JSON, nullable=False)  # User's ranked list
    mode = Column(String(50), default="listen")
    timestamp = Column(DateTime, default=func.now())
    notes = Column(Text, nullable=True)

    def to_dict(self):
        """Convert model to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "prompt": self.prompt,
            "flight_ids": self.flight_ids,
            "user_ranking": self.user_ranking,
            "mode": self.mode,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "notes": self.notes,
        }


class TeamDraftResult(Base):
    """Team Draft evaluation results with interleaved rankings."""

    __tablename__ = "team_draft_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    prompt = Column(Text, nullable=False)
    algorithm_a = Column(String(100), nullable=False)
    algorithm_b = Column(String(100), nullable=False)
    algorithm_a_ranking = Column(JSON, nullable=False)
    algorithm_b_ranking = Column(JSON, nullable=False)
    interleaved_list = Column(JSON, nullable=False)
    user_preferences = Column(JSON, nullable=False)  # List of user choices
    a_score = Column(Integer, default=0)
    b_score = Column(Integer, default=0)
    timestamp = Column(DateTime, default=func.now())
    notes = Column(Text, nullable=True)

    def to_dict(self):
        """Convert model to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "prompt": self.prompt,
            "algorithm_a": self.algorithm_a,
            "algorithm_b": self.algorithm_b,
            "algorithm_a_ranking": self.algorithm_a_ranking,
            "algorithm_b_ranking": self.algorithm_b_ranking,
            "interleaved_list": self.interleaved_list,
            "user_preferences": self.user_preferences,
            "a_score": self.a_score,
            "b_score": self.b_score,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "notes": self.notes,
        }


class Rating(Base):
    """Individual flight ratings submitted by users."""

    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    flight_id = Column(Integer, nullable=False)
    rating = Column(Integer, nullable=False)
    prompt_giver = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=func.now())

    def to_dict(self):
        """Convert model to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "flight_id": self.flight_id,
            "rating": self.rating,
            "prompt_giver": self.prompt_giver,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
