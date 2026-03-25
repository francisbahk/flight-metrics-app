"""
Database — 3 clean tables: routes, participants, rankings.
Uses SQLAlchemy ORM with MySQL (or SQLite for local dev).
"""
import os
import json
from datetime import datetime
from urllib.parse import quote_plus
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# CONFIG
# ============================================================================
def get_config(key, default=''):
    """Get config from Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


DB_TYPE = get_config('DB_TYPE', 'sqlite')

if DB_TYPE == 'mysql':
    MYSQL_HOST     = get_config('MYSQL_HOST', 'localhost')
    MYSQL_PORT     = get_config('MYSQL_PORT', '3306')
    MYSQL_DATABASE = get_config('MYSQL_DATABASE', 'flight_rankings')
    MYSQL_USER     = get_config('MYSQL_USER', 'root')
    MYSQL_PASSWORD = get_config('MYSQL_PASSWORD', '')
    _pw = quote_plus(MYSQL_PASSWORD)
    DATABASE_URL = (
        f"mysql+pymysql://{MYSQL_USER}:{_pw}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    )
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=False,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=10,
        connect_args={},
    )
else:
    _db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'flight_rankings.db')
    DATABASE_URL = f"sqlite:///{_db_path}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ============================================================================
# MODELS
# ============================================================================
class Route(Base):
    """One row per route+date (lightweight catalog — actual flights stay in file)."""
    __tablename__ = 'routes'

    route_id     = Column(String(64), primary_key=True)   # e.g. "JFK-LAX-20260301"
    origin       = Column(String(8),  nullable=False)
    destination  = Column(String(8),  nullable=False)
    date         = Column(String(16), nullable=False)
    flight_count = Column(Integer,    default=0)


class Participant(Base):
    """One row per Prolific participant."""
    __tablename__ = 'participants'

    prolific_id       = Column(String(128), primary_key=True)
    session_id        = Column(String(64))
    study_id          = Column(String(128))
    route_id          = Column(String(64))
    prompt            = Column(Text)
    # AI-filtered flights shown to this participant
    all_flights_json  = Column(MEDIUMTEXT().with_variant(Text, 'sqlite'))
    ranking_confirmed = Column(Boolean, default=False)
    created_at        = Column(DateTime, default=datetime.utcnow)
    updated_at        = Column(DateTime, default=datetime.utcnow)


class Ranking(Base):
    """One row per ranked flight per participant (5 rows when complete)."""
    __tablename__ = 'rankings'

    row_number  = Column(Integer, primary_key=True, autoincrement=True)
    prolific_id = Column(String(128), nullable=False, index=True)
    rank        = Column(Integer, nullable=False)           # 1 = top choice
    flight_key  = Column(String(256), nullable=False)       # "{id}_{departure_time}"
    flight_json = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


class ScreeningData(Base):
    """Screening question answers per participant."""
    __tablename__ = 'screening_data'

    prolific_id  = Column(String(128), primary_key=True)
    answers_json = Column(Text, nullable=False)   # JSON dict of q1–q4 answers
    screened_out = Column(Boolean, default=False)
    created_at   = Column(DateTime, default=datetime.utcnow)


class PostSurvey(Base):
    """Post-study freeform feedback per participant."""
    __tablename__ = 'post_survey'

    prolific_id = Column(String(128), primary_key=True)
    feedback    = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


class PromptAttempt(Base):
    """Every prompt version submitted by a participant, in order."""
    __tablename__ = 'prompt_attempts'

    id                = Column(Integer, primary_key=True, autoincrement=True)
    prolific_id       = Column(String(128), nullable=False, index=True)
    attempt_num       = Column(Integer, nullable=False)
    prompt_text       = Column(Text, nullable=False)
    passed            = Column(Boolean, nullable=True)
    llm_feedback      = Column(Text, nullable=True)
    chat_history_json = Column(Text, nullable=True)   # full chat conversation as JSON
    is_edit           = Column(Boolean, nullable=True, default=False)  # 1 if prompt came from a Make Edits action
    edit_source       = Column(String(32), nullable=True)              # "ranking" or "confirmation"
    created_at        = Column(DateTime, default=datetime.utcnow)


class SeedPrompt(Base):
    """A prompt from a past participant, used as a seed for cross-validation."""
    __tablename__ = 'seed_prompts'

    id           = Column(Integer, primary_key=True, autoincrement=True)
    slot_number  = Column(Integer, nullable=False)
    prolific_id  = Column(String(128), nullable=False)   # reference participant
    prompt_text  = Column(Text, nullable=False)
    flights_json = Column(Text, nullable=False)           # all flights shown for this prompt
    rerank_count = Column(Integer, default=0)
    created_at   = Column(DateTime, default=datetime.utcnow)


class CVRanking(Base):
    """Rankings submitted by participants during cross-validation."""
    __tablename__ = 'cv_rankings'

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    reviewer_prolific_id = Column(String(128), nullable=False, index=True)
    seed_prompt_id       = Column(Integer, nullable=False, index=True)
    rank                 = Column(Integer, nullable=False)
    flight_key           = Column(String(256), nullable=False)
    flight_json          = Column(Text, nullable=False)
    created_at           = Column(DateTime, default=datetime.utcnow)




# ============================================================================
# INIT
# ============================================================================
def init_db():
    """Create all tables and apply any missing column migrations."""
    Base.metadata.create_all(engine)
    _sa = __import__('sqlalchemy')
    migrations = [
        ("participants",    "ALTER TABLE participants ADD COLUMN study_id VARCHAR(128)"),
        ("prompt_attempts", "ALTER TABLE prompt_attempts ADD COLUMN llm_feedback TEXT"),
        ("prompt_attempts", "ALTER TABLE prompt_attempts ADD COLUMN chat_history_json MEDIUMTEXT"),
        ("prompt_attempts", "ALTER TABLE prompt_attempts ADD COLUMN is_edit TINYINT(1) DEFAULT 0"),
        ("prompt_attempts", "ALTER TABLE prompt_attempts ADD COLUMN edit_source VARCHAR(32)"),
    ]
    for table, sql in migrations:
        try:
            with engine.connect() as conn:
                conn.execute(_sa.text(sql))
                conn.commit()
                print(f"[DB] Migrated: {sql}")
        except Exception:
            pass  # Column already exists
    print("[DB] Tables ready.")


# ============================================================================
# ROUTE FUNCTIONS
# ============================================================================
def seed_route(route_id: str, origin: str, destination: str, date: str, flight_count: int):
    """Insert route metadata row. No-op if route already exists."""
    db = SessionLocal()
    try:
        if not db.query(Route).filter_by(route_id=route_id).first():
            db.add(Route(
                route_id=route_id,
                origin=origin,
                destination=destination,
                date=date,
                flight_count=flight_count,
            ))
            db.commit()
            print(f"[DB] Seeded route {route_id} ({flight_count} flights).")
    except Exception as e:
        db.rollback()
        print(f"[DB] seed_route error: {e}")
    finally:
        db.close()


# ============================================================================
# PARTICIPANT FUNCTIONS
# ============================================================================
def save_participant_progress(
    prolific_id: str,
    session_id: str,
    prompt: str = None,
    route_id: str = None,
    all_flights: list = None,
    study_id: str = None,
):
    """Upsert participant row. Creates on first call, updates on subsequent."""
    db = SessionLocal()
    try:
        p = db.query(Participant).filter_by(prolific_id=prolific_id).first()
        if p:
            p.session_id = session_id
            p.updated_at = datetime.utcnow()
            if prompt is not None:
                p.prompt = prompt
            if route_id is not None:
                p.route_id = route_id
            if all_flights is not None:
                p.all_flights_json = json.dumps(all_flights)
            if study_id is not None:
                p.study_id = study_id
        else:
            db.add(Participant(
                prolific_id=prolific_id,
                session_id=session_id,
                study_id=study_id,
                route_id=route_id,
                prompt=prompt,
                all_flights_json=json.dumps(all_flights) if all_flights is not None else None,
                ranking_confirmed=False,
            ))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB] save_participant_progress error: {e}")
    finally:
        db.close()


def get_participant(prolific_id: str) -> dict:
    """Return participant data dict (with all_flights list parsed), or {} if not found."""
    db = SessionLocal()
    try:
        p = db.query(Participant).filter_by(prolific_id=prolific_id).first()
        if not p:
            return {}
        return {
            'prolific_id':        p.prolific_id,
            'session_id':         p.session_id,
            'route_id':           p.route_id,
            'prompt':             p.prompt,
            'all_flights':        json.loads(p.all_flights_json) if p.all_flights_json else [],
            'ranking_confirmed':  bool(p.ranking_confirmed),
        }
    finally:
        db.close()


# ============================================================================
# PROMPT ATTEMPT FUNCTIONS
# ============================================================================
def save_prompt_attempt(prolific_id: str, prompt_text: str, passed: bool = None,
                        is_edit: bool = False, edit_source: str = None) -> int:
    """
    Append a new prompt attempt row for this participant.
    Returns the attempt_num assigned (1-based).
    is_edit=True when the prompt came from a Make Edits action; edit_source is "ranking" or "confirmation".
    """
    db = SessionLocal()
    try:
        last = (
            db.query(PromptAttempt)
            .filter_by(prolific_id=prolific_id)
            .order_by(PromptAttempt.attempt_num.desc())
            .first()
        )
        attempt_num = (last.attempt_num + 1) if last else 1
        db.add(PromptAttempt(
            prolific_id=prolific_id,
            attempt_num=attempt_num,
            prompt_text=prompt_text,
            passed=passed,
            is_edit=is_edit,
            edit_source=edit_source,
        ))
        db.commit()
        print(f"[DB] Saved prompt attempt #{attempt_num} for {prolific_id}.")
        return attempt_num
    except Exception as e:
        db.rollback()
        print(f"[DB] save_prompt_attempt error: {e}")
        return -1
    finally:
        db.close()


def update_prompt_attempt_result(prolific_id: str, attempt_num: int, passed: bool, llm_feedback: str = None):
    """Update the passed field (and optional llm_feedback) on an existing prompt attempt row."""
    db = SessionLocal()
    try:
        row = (
            db.query(PromptAttempt)
            .filter_by(prolific_id=prolific_id, attempt_num=attempt_num)
            .first()
        )
        if row:
            row.passed = passed
            if llm_feedback is not None:
                row.llm_feedback = llm_feedback
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB] update_prompt_attempt_result error: {e}")
    finally:
        db.close()


def save_chat_history(prolific_id: str, attempt_num: int, chat_history: list):
    """Save the full chat conversation JSON to the prompt_attempts row."""
    db = SessionLocal()
    try:
        row = db.query(PromptAttempt).filter_by(prolific_id=prolific_id, attempt_num=attempt_num).first()
        if row:
            row.chat_history_json = json.dumps(chat_history)
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB] save_chat_history error: {e}")
    finally:
        db.close()


# ============================================================================
# RANKING FUNCTIONS
# ============================================================================
def save_rankings(prolific_id: str, selected_flights: list, prompt: str) -> bool:
    """
    Save participant's ranked flights (replaces any previous rankings).
    Also updates participant prompt and marks ranking_confirmed = True.
    Returns True on success.
    """
    db = SessionLocal()
    try:
        # Replace any previous rankings
        db.query(Ranking).filter_by(prolific_id=prolific_id).delete()
        for rank, flight in enumerate(selected_flights, start=1):
            flight_key = f"{flight['id']}_{flight['departure_time']}"
            db.add(Ranking(
                prolific_id=prolific_id,
                rank=rank,
                flight_key=flight_key,
                flight_json=json.dumps(flight),
            ))
        # Mark participant confirmed
        p = db.query(Participant).filter_by(prolific_id=prolific_id).first()
        if p:
            p.prompt = prompt
            p.ranking_confirmed = True
            p.updated_at = datetime.utcnow()
        db.commit()
        print(f"[DB] Saved {len(selected_flights)} rankings for {prolific_id}.")
        return True
    except Exception as e:
        db.rollback()
        print(f"[DB] save_rankings error: {e}")
        return False
    finally:
        db.close()


# ============================================================================
# COMPATIBILITY STUBS (no-ops kept so dead code paths don't crash)
# ============================================================================
def mark_token_used(token: str):
    """No-op — token system removed. Kept for import compatibility."""
    pass


def save_screening_data(prolific_id: str, answers: dict, screened_out: bool) -> bool:
    """Save or replace screening question answers for a participant."""
    db = SessionLocal()
    try:
        row = db.query(ScreeningData).filter_by(prolific_id=prolific_id).first()
        if row:
            row.answers_json = json.dumps(answers)
            row.screened_out = screened_out
        else:
            db.add(ScreeningData(
                prolific_id=prolific_id,
                answers_json=json.dumps(answers),
                screened_out=screened_out,
            ))
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"[DB] save_screening_data error: {e}")
        return False
    finally:
        db.close()


def save_post_survey(prolific_id: str, feedback: str) -> bool:
    """Save or replace post-study freeform feedback for a participant."""
    db = SessionLocal()
    try:
        row = db.query(PostSurvey).filter_by(prolific_id=prolific_id).first()
        if row:
            row.feedback = feedback
        else:
            db.add(PostSurvey(prolific_id=prolific_id, feedback=feedback))
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"[DB] save_post_survey error: {e}")
        return False
    finally:
        db.close()


def get_rankings(prolific_id: str) -> list:
    """Return list of ranked flight dicts (in rank order) for a participant."""
    db = SessionLocal()
    try:
        rows = (
            db.query(Ranking)
            .filter_by(prolific_id=prolific_id)
            .order_by(Ranking.rank)
            .all()
        )
        return [json.loads(r.flight_json) for r in rows]
    finally:
        db.close()


# ============================================================================
# CROSS-VALIDATION FUNCTIONS
# ============================================================================
def get_next_seed_prompt(reviewer_prolific_id: str):
    """Return the next seed prompt to assign to this reviewer.

    Sequential allocation: lowest-id seed prompt that still has rerank_count < 10.
    Returns None if all seed prompts are fully saturated.
    """
    db = SessionLocal()
    try:
        seed = (
            db.query(SeedPrompt)
            .filter(SeedPrompt.rerank_count < 10)
            .order_by(SeedPrompt.id)
            .first()
        )
        if not seed:
            return None
        return {
            'id': seed.id,
            'slot_number': seed.slot_number,
            'prompt_text': seed.prompt_text,
            'flights_json': seed.flights_json,
        }
    finally:
        db.close()


def save_cv_rankings(reviewer_prolific_id: str, seed_prompt_id: int, selected_flights: list) -> bool:
    """Save cross-validation rankings and increment the seed prompt's rerank_count."""
    db = SessionLocal()
    try:
        for i, flight in enumerate(selected_flights):
            flight_key = f"{flight['id']}_{flight['departure_time']}"
            db.add(CVRanking(
                reviewer_prolific_id=reviewer_prolific_id,
                seed_prompt_id=seed_prompt_id,
                rank=i + 1,
                flight_key=flight_key,
                flight_json=json.dumps(flight),
            ))
        seed = db.query(SeedPrompt).filter_by(id=seed_prompt_id).first()
        if seed:
            seed.rerank_count += 1
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"[DB] save_cv_rankings error: {e}")
        return False
    finally:
        db.close()


def load_seed_prompt(slot_number: int, prolific_id: str, prompt_text: str, flights_json: str) -> bool:
    """Insert a single seed prompt row. No-op if prolific_id already loaded."""
    db = SessionLocal()
    try:
        existing = db.query(SeedPrompt).filter_by(prolific_id=prolific_id).first()
        if existing:
            return False  # already loaded
        db.add(SeedPrompt(
            slot_number=slot_number,
            prolific_id=prolific_id,
            prompt_text=prompt_text,
            flights_json=flights_json,
        ))
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"[DB] load_seed_prompt error: {e}")
        return False
    finally:
        db.close()
