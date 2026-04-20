"""
Tests for backend.prolific_bonus and the supporting DB helpers.

Run with:
    cd flight-metrics-app
    pytest tests/test_prolific_bonus.py -v
"""
import json
from unittest.mock import patch, MagicMock

import pytest

import backend.db as db_module
from backend.db import (
    Participant,
    Ranking,
    SeedPrompt,
    BonusAward,
    save_rankings,
    save_participant_progress,
    load_seed_prompt,
    record_bonus_award,
    has_bonus_award,
    get_ranking_flight_keys,
    get_participant_study_id,
    get_seed_source_prolific_id,
)
from backend.prolific_bonus import (
    count_overlap,
    create_bonus_payment,
    maybe_award_cv_overlap_bonus,
    PROLIFIC_BONUS_URL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _flight(fid: str, dep: str = "2026-05-01T10:00:00") -> dict:
    """Minimal flight dict compatible with save_rankings / CV flight_key."""
    return {
        "id": fid,
        "departure_time": dep,
        "arrival_time": "2026-05-01T14:00:00",
        "origin": "JFK",
        "destination": "LAX",
        "airline": "AA",
        "flight_number": f"AA{fid}",
        "price": 350,
        "duration_min": 240,
        "stops": 0,
    }


@pytest.fixture
def seeded(test_db):
    """Seed: source participant + 20-flight ranking + seed prompt."""
    source_pid = "SOURCE_PID"
    reviewer_pid = "REVIEWER_PID"

    save_participant_progress(
        prolific_id=source_pid,
        session_id="s1",
        prompt="I want a cheap morning nonstop",
        study_id="STUDY_SOURCE",
        all_flights=[_flight(f"F{i}") for i in range(30)],
    )
    source_flights = [_flight(f"F{i}") for i in range(20)]
    assert save_rankings(source_pid, source_flights, "I want a cheap morning nonstop")

    save_participant_progress(
        prolific_id=reviewer_pid,
        session_id="s2",
        prompt="placeholder",
        study_id="STUDY_REVIEWER",
    )

    seed_flights = [_flight(f"F{i}") for i in range(30)]
    assert load_seed_prompt(
        slot_number=1,
        prolific_id=source_pid,
        prompt_text="seed prompt",
        flights_json=json.dumps(seed_flights),
        overwrite=True,
    )

    db = test_db()
    seed = db.query(SeedPrompt).filter_by(prolific_id=source_pid).first()
    seed_id = seed.id
    db.close()

    return {
        "source_pid": source_pid,
        "reviewer_pid": reviewer_pid,
        "seed_id": seed_id,
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def test_get_ranking_flight_keys_returns_20(seeded):
    keys = get_ranking_flight_keys(seeded["source_pid"])
    assert len(keys) == 20
    assert "F0_2026-05-01T10:00:00" in keys


def test_get_participant_study_id(seeded):
    assert get_participant_study_id(seeded["reviewer_pid"]) == "STUDY_REVIEWER"
    assert get_participant_study_id("UNKNOWN_PID") == ""


def test_get_seed_source_prolific_id(seeded):
    assert get_seed_source_prolific_id(seeded["seed_id"]) == seeded["source_pid"]
    assert get_seed_source_prolific_id(9999) == ""


def test_record_bonus_award_is_idempotent(seeded):
    ok = record_bonus_award(
        reviewer_prolific_id=seeded["reviewer_pid"],
        source_prolific_id=seeded["source_pid"],
        seed_prompt_id=seeded["seed_id"],
        overlap_count=12,
        amount_usd="2.00",
        status="created",
    )
    assert ok
    # Second insert for same reviewer is a no-op.
    again = record_bonus_award(
        reviewer_prolific_id=seeded["reviewer_pid"],
        source_prolific_id=seeded["source_pid"],
        seed_prompt_id=seeded["seed_id"],
        overlap_count=99,
        amount_usd="2.00",
        status="created",
    )
    assert not again
    assert has_bonus_award(seeded["reviewer_pid"])


# ---------------------------------------------------------------------------
# count_overlap
# ---------------------------------------------------------------------------

def test_count_overlap_exact(seeded):
    # Reviewer picks 10 of the source's 20 flights.
    reviewer_cv = [_flight(f"F{i}") for i in range(10)]
    assert count_overlap(reviewer_cv, seeded["source_pid"]) == 10


def test_count_overlap_partial(seeded):
    reviewer_cv = [_flight(f"F{i}") for i in range(5)] + [_flight(f"Z{i}") for i in range(5)]
    assert count_overlap(reviewer_cv, seeded["source_pid"]) == 5


def test_count_overlap_zero_when_source_missing(seeded):
    reviewer_cv = [_flight("F0"), _flight("F1")]
    assert count_overlap(reviewer_cv, "NOT_A_USER") == 0


def test_count_overlap_zero_when_reviewer_empty(seeded):
    assert count_overlap([], seeded["source_pid"]) == 0


# ---------------------------------------------------------------------------
# create_bonus_payment
# ---------------------------------------------------------------------------

def test_create_bonus_payment_posts_to_prolific_with_csv():
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"id": "BULK_BONUS_123"}

    with patch("backend.prolific_bonus.requests.post", return_value=mock_resp) as post:
        result = create_bonus_payment(
            study_id="STUDY_X",
            participant_id="REVIEWER_PID",
            amount_usd="2.00",
            api_token="TOK_abc",
        )

    assert result == {"id": "BULK_BONUS_123"}
    args, kwargs = post.call_args
    assert args[0] == PROLIFIC_BONUS_URL
    assert kwargs["headers"]["Authorization"] == "Token TOK_abc"
    assert kwargs["json"]["study_id"] == "STUDY_X"
    assert kwargs["json"]["csv_bonuses"] == "REVIEWER_PID,2.00"


def test_create_bonus_payment_raises_on_4xx():
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Bad participant id"

    with patch("backend.prolific_bonus.requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Prolific bonus API returned 400"):
            create_bonus_payment(
                study_id="STUDY_X",
                participant_id="BAD",
                amount_usd="2.00",
                api_token="TOK_abc",
            )


# ---------------------------------------------------------------------------
# maybe_award_cv_overlap_bonus
# ---------------------------------------------------------------------------

def _config_fn(overrides):
    def _get(key, default=""):
        return overrides.get(key, default)
    return _get


def test_skipped_when_overlap_below_threshold(seeded):
    reviewer_cv = [_flight(f"F{i}") for i in range(5)]  # only 5 overlap
    with patch("backend.prolific_bonus.get_config",
               side_effect=_config_fn({
                   "PROLIFIC_BONUS_ENABLED": "true",
                   "PROLIFIC_BONUS_THRESHOLD": "10",
                   "PROLIFIC_BONUS_AMOUNT_USD": "2.00",
                   "PROLIFIC_API_TOKEN": "TOK",
               })), \
         patch("backend.prolific_bonus.requests.post") as post:
        result = maybe_award_cv_overlap_bonus(
            reviewer_prolific_id=seeded["reviewer_pid"],
            seed_prompt_id=seeded["seed_id"],
            reviewer_cv_flights=reviewer_cv,
        )

    assert result["status"] == "skipped"
    assert result["overlap"] == 5
    post.assert_not_called()
    assert has_bonus_award(seeded["reviewer_pid"])


def test_creates_bonus_when_overlap_meets_threshold(seeded):
    reviewer_cv = [_flight(f"F{i}") for i in range(10)]
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"id": "BULK_42"}

    with patch("backend.prolific_bonus.get_config",
               side_effect=_config_fn({
                   "PROLIFIC_BONUS_ENABLED": "true",
                   "PROLIFIC_BONUS_THRESHOLD": "10",
                   "PROLIFIC_BONUS_AMOUNT_USD": "2.00",
                   "PROLIFIC_API_TOKEN": "TOK_abc",
               })), \
         patch("backend.prolific_bonus.requests.post", return_value=mock_resp) as post:
        result = maybe_award_cv_overlap_bonus(
            reviewer_prolific_id=seeded["reviewer_pid"],
            seed_prompt_id=seeded["seed_id"],
            reviewer_cv_flights=reviewer_cv,
        )

    assert result["status"] == "created"
    assert result["overlap"] == 10
    assert result["bonus_id"] == "BULK_42"
    _, kwargs = post.call_args
    assert kwargs["json"]["study_id"] == "STUDY_REVIEWER"
    assert kwargs["json"]["csv_bonuses"] == f"{seeded['reviewer_pid']},2.00"


def test_pending_when_bonus_disabled(seeded):
    reviewer_cv = [_flight(f"F{i}") for i in range(15)]
    with patch("backend.prolific_bonus.get_config",
               side_effect=_config_fn({
                   "PROLIFIC_BONUS_ENABLED": "false",
                   "PROLIFIC_BONUS_THRESHOLD": "10",
                   "PROLIFIC_BONUS_AMOUNT_USD": "2.00",
                   "PROLIFIC_API_TOKEN": "TOK",
               })), \
         patch("backend.prolific_bonus.requests.post") as post:
        result = maybe_award_cv_overlap_bonus(
            reviewer_prolific_id=seeded["reviewer_pid"],
            seed_prompt_id=seeded["seed_id"],
            reviewer_cv_flights=reviewer_cv,
        )

    assert result["status"] == "pending"
    assert result["reason"] == "bonus_enabled=false"
    post.assert_not_called()


def test_pending_when_no_api_token(seeded):
    reviewer_cv = [_flight(f"F{i}") for i in range(15)]
    with patch("backend.prolific_bonus.get_config",
               side_effect=_config_fn({
                   "PROLIFIC_BONUS_ENABLED": "true",
                   "PROLIFIC_BONUS_THRESHOLD": "10",
                   "PROLIFIC_BONUS_AMOUNT_USD": "2.00",
                   "PROLIFIC_API_TOKEN": "",
               })), \
         patch("backend.prolific_bonus.requests.post") as post:
        result = maybe_award_cv_overlap_bonus(
            reviewer_prolific_id=seeded["reviewer_pid"],
            seed_prompt_id=seeded["seed_id"],
            reviewer_cv_flights=reviewer_cv,
        )

    assert result["status"] == "pending"
    assert result["reason"] == "missing_api_token"
    post.assert_not_called()


def test_idempotent_second_call_no_ops(seeded):
    reviewer_cv = [_flight(f"F{i}") for i in range(12)]
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"id": "B1"}

    with patch("backend.prolific_bonus.get_config",
               side_effect=_config_fn({
                   "PROLIFIC_BONUS_ENABLED": "true",
                   "PROLIFIC_BONUS_THRESHOLD": "10",
                   "PROLIFIC_BONUS_AMOUNT_USD": "2.00",
                   "PROLIFIC_API_TOKEN": "TOK",
               })), \
         patch("backend.prolific_bonus.requests.post", return_value=mock_resp) as post:
        first = maybe_award_cv_overlap_bonus(
            reviewer_prolific_id=seeded["reviewer_pid"],
            seed_prompt_id=seeded["seed_id"],
            reviewer_cv_flights=reviewer_cv,
        )
        second = maybe_award_cv_overlap_bonus(
            reviewer_prolific_id=seeded["reviewer_pid"],
            seed_prompt_id=seeded["seed_id"],
            reviewer_cv_flights=reviewer_cv,
        )

    assert first["status"] == "created"
    assert second["status"] == "already_awarded"
    assert post.call_count == 1  # no second API call


def test_failed_status_when_api_errors(seeded):
    reviewer_cv = [_flight(f"F{i}") for i in range(15)]
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "server error"

    with patch("backend.prolific_bonus.get_config",
               side_effect=_config_fn({
                   "PROLIFIC_BONUS_ENABLED": "true",
                   "PROLIFIC_BONUS_THRESHOLD": "10",
                   "PROLIFIC_BONUS_AMOUNT_USD": "2.00",
                   "PROLIFIC_API_TOKEN": "TOK",
               })), \
         patch("backend.prolific_bonus.requests.post", return_value=mock_resp):
        result = maybe_award_cv_overlap_bonus(
            reviewer_prolific_id=seeded["reviewer_pid"],
            seed_prompt_id=seeded["seed_id"],
            reviewer_cv_flights=reviewer_cv,
        )

    assert result["status"] == "failed"
    assert "500" in result["error"]
    assert has_bonus_award(seeded["reviewer_pid"])  # row still recorded
