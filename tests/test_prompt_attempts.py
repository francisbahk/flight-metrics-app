"""
Tests for prompt attempt tracking (save_prompt_attempt, update_prompt_attempt_result)
and the Groq validation helper (validate_prompt_with_groq).

Run with:
    cd flight-metrics-app
    pytest tests/test_prompt_attempts.py -v
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from backend.db import (
    PromptAttempt,
    save_prompt_attempt,
    update_prompt_attempt_result,
)
from backend.prompt_validation import validate_prompt_with_groq


# ---------------------------------------------------------------------------
# DB: save_prompt_attempt
# ---------------------------------------------------------------------------

def test_first_attempt_gets_num_1(test_db):
    n = save_prompt_attempt("P001", "I want cheap flights")
    assert n == 1


def test_second_attempt_increments(test_db):
    save_prompt_attempt("P001", "first prompt")
    n = save_prompt_attempt("P001", "revised prompt with more detail")
    assert n == 2


def test_attempts_are_independent_per_participant(test_db):
    save_prompt_attempt("P001", "P001 first")
    save_prompt_attempt("P002", "P002 first")
    n = save_prompt_attempt("P001", "P001 second")
    assert n == 2  # P002's row should not affect P001's counter


def test_prompt_text_is_stored_verbatim(test_db):
    text = "I prefer morning departures, direct flights only, budget under $400."
    save_prompt_attempt("P003", text)

    db = test_db()
    row = db.query(PromptAttempt).filter_by(prolific_id="P003").first()
    db.close()
    assert row.prompt_text == text


def test_passed_defaults_to_none(test_db):
    save_prompt_attempt("P004", "some prompt")

    db = test_db()
    row = db.query(PromptAttempt).filter_by(prolific_id="P004").first()
    db.close()
    assert row.passed is None


def test_passed_can_be_set_on_save(test_db):
    save_prompt_attempt("P005", "detailed prompt", passed=True)

    db = test_db()
    row = db.query(PromptAttempt).filter_by(prolific_id="P005").first()
    db.close()
    assert row.passed is True


# ---------------------------------------------------------------------------
# DB: update_prompt_attempt_result
# ---------------------------------------------------------------------------

def test_update_sets_passed_true(test_db):
    n = save_prompt_attempt("P010", "my prompt")
    update_prompt_attempt_result("P010", n, passed=True)

    db = test_db()
    row = db.query(PromptAttempt).filter_by(prolific_id="P010", attempt_num=n).first()
    db.close()
    assert row.passed is True


def test_update_sets_passed_false(test_db):
    n = save_prompt_attempt("P011", "short")
    update_prompt_attempt_result("P011", n, passed=False)

    db = test_db()
    row = db.query(PromptAttempt).filter_by(prolific_id="P011", attempt_num=n).first()
    db.close()
    assert row.passed is False


def test_update_nonexistent_row_is_silent(test_db):
    # Should not raise even if the row doesn't exist
    update_prompt_attempt_result("NOBODY", 99, passed=True)


def test_all_attempts_preserved_after_update(test_db):
    """Updating attempt #2 must not delete attempt #1."""
    save_prompt_attempt("P020", "first try")
    n2 = save_prompt_attempt("P020", "second try")
    update_prompt_attempt_result("P020", n2, passed=False)

    db = test_db()
    rows = db.query(PromptAttempt).filter_by(prolific_id="P020").order_by(PromptAttempt.attempt_num).all()
    db.close()

    assert len(rows) == 2
    assert rows[0].attempt_num == 1
    assert rows[0].passed is None     # first attempt untouched
    assert rows[1].attempt_num == 2
    assert rows[1].passed is False


# ---------------------------------------------------------------------------
# Groq validator: validate_prompt_with_groq
# ---------------------------------------------------------------------------

def _make_groq_response(detailed: bool, feedback: str = ""):
    """Build a fake Groq API response object."""
    content = json.dumps({"detailed": detailed, "feedback": feedback})
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_validate_returns_true_when_detailed():
    fake_response = _make_groq_response(detailed=True)

    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch("backend.prompt_validation.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = fake_response
            ok, feedback = validate_prompt_with_groq(
                "I prefer morning direct flights on United, price under $400."
            )

    assert ok is True
    assert feedback == ""


def test_validate_returns_false_with_feedback_when_not_detailed():
    fake_response = _make_groq_response(detailed=False, feedback="Please mention airline preference.")

    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch("backend.prompt_validation.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = fake_response
            ok, feedback = validate_prompt_with_groq("asdfasdf")

    assert ok is False
    assert "airline" in feedback


def test_validate_returns_true_when_no_api_key():
    """Without GROQ_API_KEY, validation should pass (fail open)."""
    with patch.dict("os.environ", {}, clear=True):
        ok, feedback = validate_prompt_with_groq("short")

    assert ok is True
    assert feedback == ""
