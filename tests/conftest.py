import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.db as db_module


@pytest.fixture()
def test_db(monkeypatch):
    """
    Patch backend.db.SessionLocal with an in-memory SQLite session for each test.
    StaticPool ensures all connections share the same in-memory database.
    """
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    db_module.Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSession)
    yield TestSession
    db_module.Base.metadata.drop_all(test_engine)
    test_engine.dispose()
