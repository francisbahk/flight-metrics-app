"""
Edge case tests for pilot study data integrity.
Covers: duration parsing, DB saves, session progress upsert,
        token validation, and cross-validation group logic.
"""
from datetime import datetime

import pytest

from backend.db import (
    AccessToken,
    SessionProgress,
    save_search_and_csv,
    save_session_progress,
    validate_access_token,
)
from backend.utils.parse_duration import parse_duration_to_minutes
from pilot_tokens import get_rerank_targets, get_token_group, is_pilot_token

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_FLIGHT = {
    "id": "F1",
    "airline": "UA",
    "departure_time": "2025-06-01T08:00:00",
    "arrival_time": "2025-06-01T11:00:00",
    "price": 300.0,
    "duration": "PT3H",
    "duration_min": 180,
    "stops": 0,
    "origin": "JFK",
    "destination": "LAX",
}

_BASE_PARAMS = {
    "origins": ["JFK"],
    "destinations": ["LAX"],
    "preferences": {},
    "departure_date": "2025-06-01",
}


# ---------------------------------------------------------------------------
# Edge case 1: parse_duration_to_minutes
# ---------------------------------------------------------------------------

def test_parse_duration_minutes_only():
    assert parse_duration_to_minutes("PT45M") == 45.0


def test_parse_duration_hours_only():
    assert parse_duration_to_minutes("PT3H") == 180.0


def test_parse_duration_empty_string():
    assert parse_duration_to_minutes("") == 0.0


# ---------------------------------------------------------------------------
# Edge case 2: save_search_and_csv — ranked flight absent from all_flights
# ---------------------------------------------------------------------------

def test_save_search_ranked_flight_not_in_all_flights(test_db, capsys):
    missing_flight = {
        **_BASE_FLIGHT,
        "id": "MISSING",
        "airline": "DL",
        "price": 250.0,
    }

    search_id = save_search_and_csv(
        session_id="session-missing",
        user_prompt="flight from JFK to LAX",
        parsed_params=_BASE_PARAMS,
        all_flights=[_BASE_FLIGHT],
        selected_flights=[missing_flight],
        csv_data="origin,destination\nJFK,LAX",
        token="GA01",
    )

    assert search_id is not None
    captured = capsys.readouterr()
    assert "Warning" in captured.out


# ---------------------------------------------------------------------------
# Edge case 3: save_search_and_csv — empty selected_flights
# ---------------------------------------------------------------------------

def test_save_search_empty_rankings(test_db):
    search_id = save_search_and_csv(
        session_id="session-empty",
        user_prompt="cheap flight to LA",
        parsed_params=_BASE_PARAMS,
        all_flights=[_BASE_FLIGHT],
        selected_flights=[],
        csv_data="origin,destination\nJFK,LAX",
        token=None,
    )

    assert search_id is not None


# ---------------------------------------------------------------------------
# Edge case 4: save_session_progress — upsert (no duplicate rows)
# ---------------------------------------------------------------------------

def test_session_progress_upsert(test_db):
    save_session_progress("TOKEN-UPSERT", {
        "session_id": "session-abc",
        "current_phase": "search",
    })
    save_session_progress("TOKEN-UPSERT", {
        "session_id": "session-abc",
        "current_phase": "flight_selection",
    })

    db = test_db()
    rows = db.query(SessionProgress).filter(
        SessionProgress.access_token == "TOKEN-UPSERT"
    ).all()
    db.close()

    assert len(rows) == 1
    assert rows[0].current_phase == "flight_selection"


# ---------------------------------------------------------------------------
# Edge case 5: token validation
# ---------------------------------------------------------------------------

def test_validate_token_not_found(test_db):
    result = validate_access_token("NONEXISTENT")
    assert result["valid"] is False
    assert result["is_used"] is False


def test_validate_token_already_used(test_db):
    db = test_db()
    db.add(AccessToken(token="USED-TOKEN", is_used=1, used_at=datetime.utcnow()))
    db.commit()
    db.close()

    result = validate_access_token("USED-TOKEN")
    assert result["valid"] is False
    assert result["is_used"] is True


def test_validate_token_valid_and_unused(test_db):
    db = test_db()
    db.add(AccessToken(token="FRESH-TOKEN", is_used=0))
    db.commit()
    db.close()

    result = validate_access_token("FRESH-TOKEN")
    assert result["valid"] is True
    assert result["is_used"] is False


# ---------------------------------------------------------------------------
# Edge case 6: cross-validation group logic (pilot_tokens.py)
# ---------------------------------------------------------------------------

def test_group_a_has_no_reranks():
    assert get_rerank_targets("GA01") == []


def test_group_b_has_four_reranks_and_skips_self():
    targets = get_rerank_targets("GB01")
    assert len(targets) == 4
    assert "GA01" not in targets  # GB01 skips GA01


def test_unknown_token_returns_empty():
    assert get_rerank_targets("XXXX") == []
    assert get_token_group("XXXX") is None
    assert is_pilot_token("XXXX") is False
