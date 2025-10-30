"""
Database configuration and connection management using SQLAlchemy.
"""
import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters from environment
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB", "flights")

# Construct database URL
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"

# Create SQLAlchemy engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Enable connection health checks
    pool_size=10,  # Maximum number of connections to keep open
    max_overflow=20,  # Maximum number of connections that can be created beyond pool_size
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False  # Set to True for SQL query logging
)

# Create SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session.
    Use with FastAPI Depends() to inject database sessions into routes.

    Usage:
        @app.get("/example")
        def example_route(db: Session = Depends(get_db)):
            # Use db session here
            pass
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection():
    """
    Test database connection.
    Returns True if connection is successful, raises exception otherwise.
    """
    try:
        # Create a test session
        db = SessionLocal()
        # Execute a simple query
        db.execute(text("SELECT 1"))
        db.close()
        print(f"✓ Database connection successful: {MYSQL_DB}@{MYSQL_HOST}:{MYSQL_PORT}")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {str(e)}")
        raise


def init_db():
    """
    Initialize database tables.
    Creates all tables defined in models if they don't exist.
    """
    from models.flight import Flight, ListenRanking, TeamDraftResult, Rating
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables initialized")
