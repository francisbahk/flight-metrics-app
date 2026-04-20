"""
Prolific bonus payments.

Creates a bonus payment for a cross-validation reviewer when their top-20
rerank overlaps the source participant's original top-20 ranking by at least
a configured threshold.

Docs: https://docs.prolific.com/api-reference/bonuses/create-bonus-payments

Runtime wiring (see frontend/pages/results.py): after a CV submission is saved
to the database, maybe_award_cv_overlap_bonus is called in a background thread
so the UI is not blocked by Prolific API latency.

Environment / Streamlit secrets (read via backend.db.get_config):
  PROLIFIC_API_TOKEN          API token for https://api.prolific.com
  PROLIFIC_BONUS_ENABLED      "true" to actually hit the Prolific API (default: false)
  PROLIFIC_BONUS_THRESHOLD    minimum overlap count to award a bonus (default: 10)
  PROLIFIC_BONUS_AMOUNT_USD   bonus amount per reviewer, USD decimal (default: 2.00)
"""
from __future__ import annotations

import json
from typing import Optional

import requests

from backend.db import (
    get_config,
    get_participant_study_id,
    get_ranking_flight_keys,
    get_seed_source_prolific_id,
    has_bonus_award,
    record_bonus_award,
)


PROLIFIC_BONUS_URL = "https://api.prolific.com/api/v1/submissions/bonus-payments/"


def _config_flag(key: str, default: bool = False) -> bool:
    raw = get_config(key, 'true' if default else 'false')
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


def _config_int(key: str, default: int) -> int:
    try:
        return int(get_config(key, str(default)))
    except (TypeError, ValueError):
        return default


def _config_amount(key: str, default: str) -> str:
    raw = str(get_config(key, default)).strip()
    try:
        return f"{float(raw):.2f}"
    except ValueError:
        return default


def count_overlap(reviewer_cv_flights: list, source_prolific_id: str) -> int:
    """Count flights that appear in both the reviewer's CV top list and the
    source participant's original top ranking, keyed by id+departure_time."""
    if not source_prolific_id or not reviewer_cv_flights:
        return 0
    source_keys = get_ranking_flight_keys(source_prolific_id)
    if not source_keys:
        return 0
    reviewer_keys = {
        f"{f['id']}_{f['departure_time']}" for f in reviewer_cv_flights
    }
    return len(reviewer_keys & source_keys)


def create_bonus_payment(
    study_id: str,
    participant_id: str,
    amount_usd: str,
    api_token: str,
    timeout: float = 15.0,
) -> dict:
    """POST to Prolific's create-bonus-payments endpoint.

    Returns the parsed response body on 2xx, raises on failure.
    Note: this only *creates* the bonus. Actually paying it requires a
    separate POST to .../bulk-bonus-payments/<id>/pay/ (not done here).
    """
    payload = {
        "study_id": study_id,
        "csv_bonuses": f"{participant_id},{amount_usd}",
    }
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(PROLIFIC_BONUS_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Prolific bonus API returned {resp.status_code}: {resp.text[:500]}"
        )
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text}


def maybe_award_cv_overlap_bonus(
    reviewer_prolific_id: str,
    seed_prompt_id: int,
    reviewer_cv_flights: list,
    source_prolific_id: Optional[str] = None,
    study_id: Optional[str] = None,
) -> dict:
    """End-to-end: compute overlap, decide, record, and (if enabled) call Prolific.

    Idempotent per reviewer — subsequent calls for the same reviewer_prolific_id
    no-op because BonusAward.reviewer_prolific_id is unique.

    Returns a summary dict: {status, overlap, threshold, amount, bonus_id, error}.
    """
    threshold = _config_int('PROLIFIC_BONUS_THRESHOLD', 10)
    amount_usd = _config_amount('PROLIFIC_BONUS_AMOUNT_USD', '2.00')
    api_enabled = _config_flag('PROLIFIC_BONUS_ENABLED', default=False)

    if has_bonus_award(reviewer_prolific_id):
        return {'status': 'already_awarded', 'overlap': None}

    if source_prolific_id is None:
        source_prolific_id = get_seed_source_prolific_id(seed_prompt_id)

    overlap = count_overlap(reviewer_cv_flights, source_prolific_id or '')

    if overlap < threshold:
        record_bonus_award(
            reviewer_prolific_id=reviewer_prolific_id,
            source_prolific_id=source_prolific_id,
            seed_prompt_id=seed_prompt_id,
            overlap_count=overlap,
            amount_usd=amount_usd,
            status='skipped',
            error_message=f'overlap {overlap} < threshold {threshold}',
        )
        return {'status': 'skipped', 'overlap': overlap, 'threshold': threshold}

    if study_id is None:
        study_id = get_participant_study_id(reviewer_prolific_id)

    api_token = get_config('PROLIFIC_API_TOKEN', '')

    if not api_enabled or not api_token or not study_id:
        record_bonus_award(
            reviewer_prolific_id=reviewer_prolific_id,
            source_prolific_id=source_prolific_id,
            seed_prompt_id=seed_prompt_id,
            overlap_count=overlap,
            amount_usd=amount_usd,
            status='pending',
            study_id=study_id or None,
            error_message=(
                'bonus_enabled=false' if not api_enabled else
                'missing_api_token' if not api_token else
                'missing_study_id'
            ),
        )
        return {
            'status': 'pending',
            'overlap': overlap,
            'threshold': threshold,
            'amount': amount_usd,
            'reason': (
                'bonus_enabled=false' if not api_enabled else
                'missing_api_token' if not api_token else
                'missing_study_id'
            ),
        }

    try:
        result = create_bonus_payment(
            study_id=study_id,
            participant_id=reviewer_prolific_id,
            amount_usd=amount_usd,
            api_token=api_token,
        )
        bonus_id = result.get('id') if isinstance(result, dict) else None
        record_bonus_award(
            reviewer_prolific_id=reviewer_prolific_id,
            source_prolific_id=source_prolific_id,
            seed_prompt_id=seed_prompt_id,
            overlap_count=overlap,
            amount_usd=amount_usd,
            status='created',
            study_id=study_id,
            prolific_bonus_id=bonus_id,
        )
        print(f"[BONUS] Created $2 bonus for {reviewer_prolific_id} "
              f"(overlap={overlap}, bonus_id={bonus_id})")
        return {
            'status': 'created',
            'overlap': overlap,
            'threshold': threshold,
            'amount': amount_usd,
            'bonus_id': bonus_id,
        }
    except Exception as e:
        err = str(e)
        record_bonus_award(
            reviewer_prolific_id=reviewer_prolific_id,
            source_prolific_id=source_prolific_id,
            seed_prompt_id=seed_prompt_id,
            overlap_count=overlap,
            amount_usd=amount_usd,
            status='failed',
            study_id=study_id,
            error_message=err,
        )
        print(f"[BONUS] Failed to create bonus for {reviewer_prolific_id}: {err}")
        return {
            'status': 'failed',
            'overlap': overlap,
            'threshold': threshold,
            'amount': amount_usd,
            'error': err,
        }
