"""
Flight Evaluation Web App
Entry point / router. All rendering is delegated to frontend/ modules.
"""
import os
import sys
import uuid
import json as _json
import threading

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from phases import is_phase_url, get_phase_group, make_effective_token, is_phase_token

# ============================================================================
# CONFIGURATION
# ============================================================================
def _get_auto_backup_enabled():
    try:
        if hasattr(st, 'secrets') and 'AUTO_BACKUP_ENABLED' in st.secrets:
            return str(st.secrets['AUTO_BACKUP_ENABLED']).lower() == 'true'
    except Exception:
        pass
    return os.getenv('AUTO_BACKUP_ENABLED', 'false').lower() == 'true'


AUTO_BACKUP_ENABLED = _get_auto_backup_enabled()


def trigger_backup_on_completion(token: str):
    """Trigger S3 backup in a background thread after session completion."""
    if not AUTO_BACKUP_ENABLED:
        print(f"[BACKUP] Auto-backup disabled. Session {token} completed.")
        return

    def run_backup():
        try:
            from backup_db import run_backup as do_backup
            print(f"[BACKUP] Starting backup after session {token} completed...")
            success = do_backup(on_completion=True)
            print(f"[BACKUP] {'Completed' if success else 'Failed'} for session {token}")
        except Exception as e:
            print(f"[BACKUP] Error running backup: {e}")

    threading.Thread(target=run_backup, daemon=True).start()


# ============================================================================
# QUERY PARAM HELPERS
# ============================================================================
def get_query_param(key, default=None):
    try:
        return st.query_params.get(key, default)
    except AttributeError:
        params = st.experimental_get_query_params()
        values = params.get(key, [default])
        return values[0] if values else default


# ============================================================================
# STATIC FLIGHT DATABASE (pilot study)
# ============================================================================
STATIC_FLIGHTS = []
DAYS_OF_WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

try:
    _static_path = os.path.join(os.path.dirname(__file__), 'static_flights_filtered.json')
    with open(_static_path) as _f:
        STATIC_FLIGHTS = _json.load(_f)

    _route_day_counts = {}
    for _f in STATIC_FLIGHTS:
        _rdk = (_f['origin'], _f['destination'], _f['day_of_week'])
        _route_day_counts[_rdk] = _route_day_counts.get(_rdk, 0) + 1

    STATIC_ROUTE_DAY_OPTIONS = [
        f"{_o} \u2192 {_d} ({_dow})"
        for (_o, _d, _dow) in sorted(
            _route_day_counts,
            key=lambda x: (x[0], x[1], DAYS_OF_WEEK.index(x[2]))
        )
        if _route_day_counts[(_o, _d, _dow)] >= 50
    ]
    print(f"Loaded {len(STATIC_FLIGHTS)} static flights, "
          f"{len(STATIC_ROUTE_DAY_OPTIONS)} route+day options")
except FileNotFoundError:
    STATIC_ROUTE_DAY_OPTIONS = []
    print("static_flights_filtered.json not found. Run fetch_static_flights.py first.")


# ============================================================================
# ONE-TIME STARTUP (DB init + seed)
# ============================================================================
try:
    from backend.db import init_db
    init_db()
except Exception as e:
    print(f"Database initialization: {str(e)}")

try:
    from seed_cross_validation import seed_cross_validation_data
    seed_cross_validation_data()
except Exception as e:
    print(f"Cross-validation seed: {str(e)}")


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Flight Ranker",
    page_icon="✈️",
    layout="wide"
)

# ============================================================================
# PHASE GATE — check immediately after set_page_config, before anything renders
# Uses an inline set so this works even if phases.py has an import issue.
# ============================================================================
_PHASE_IDS = {'PHASEONE', 'PHASETWO'}
_raw_id = (get_query_param('id', '') or '').upper()

if _raw_id in _PHASE_IDS and not st.session_state.get('prolific_id'):
    from frontend.pages.prolific_gate import render_prolific_id_gate
    render_prolific_id_gate(_raw_id)
    st.stop()

# Inject global CSS (only reached after gate check passes)
from frontend.styles import GLOBAL_CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
_defaults = {
    'session_id': str(uuid.uuid4()),
    'token': None,
    'token_valid': False,
    'token_message': '',
    'token_group': None,
    'rerank_targets': [],
    'current_rerank_index': 0,
    'completed_reranks': [],
    'all_reranks_completed': False,
    'all_flights': [],
    'selected_flights': [],
    'parsed_params': None,
    'search_mode': 'ai',
    'csv_generated': False,
    'single_sort_version': 0,
    'sort_price_dir': 'asc',
    'sort_duration_dir': 'asc',
    'sort_price_dir_single': 'asc',
    'sort_duration_dir_single': 'asc',
    'outbound_submitted': False,
    'checkbox_version': 0,
    'csv_data_outbound': None,
    'review_confirmed': False,
    'cross_validation_completed': False,
    'survey_completed': False,
    'completion_page_dismissed': False,
    'filter_airlines': [],
    'filter_connections': [],
    'filter_price_range': None,
    'filter_duration_range': None,
    'filter_departure_time_range': None,
    'filter_arrival_time_range': None,
    'filter_origins': [],
    'filter_destinations': [],
    'filter_reset_counter': 0,
    'airline_names': {},
    'origin_search_results': [],
    'origin_iata_map': {},
    'dest_search_results': [],
    'dest_iata_map': {},
}
for key, value in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================================
# TOKEN / PHASE HANDLING
# ============================================================================
raw_url_id = get_query_param('id') or ''

if is_phase_url(raw_url_id):
    # ----------------------------------------------------------------
    # PHASE-BASED FLOW  (?id=PHASEONE  or  ?id=PHASETWO)
    # Prolific ID gate: show form before anything else if ID not yet set.
    # ----------------------------------------------------------------
    phase_url = raw_url_id.upper()

    if not st.session_state.get('prolific_id'):
        # Inject page config CSS so the gate page looks clean
        from frontend.styles import GLOBAL_CSS
        st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
        from frontend.pages.prolific_gate import render_prolific_id_gate
        render_prolific_id_gate(phase_url)
        st.stop()

    # Prolific ID already collected — build effective token
    prolific_id = st.session_state.prolific_id
    effective_token = make_effective_token(phase_url, prolific_id)

    st.session_state.token = effective_token
    st.session_state.token_valid = True
    st.session_state.token_message = 'Phase participant'
    st.session_state.phase_url = phase_url

    # Group/rerank config from phases.py (no pre-assigned rerank_targets for phases)
    st.session_state.token_group = get_phase_group(phase_url)
    st.session_state.rerank_targets = []   # Dynamic matching at runtime for Phase 2

    print(f"[PHASE] {phase_url} participant - Prolific ID: {prolific_id[:6]}*** "
          f"- Group: {st.session_state.token_group}")

    # Session restore (keyed off effective token)
    if 'session_restored' not in st.session_state:
        try:
            from backend.db import get_session_progress
            existing_progress = get_session_progress(effective_token)
            if existing_progress:
                st.session_state.session_id = existing_progress['session_id']
                if existing_progress.get('all_flights'):
                    st.session_state.all_flights = existing_progress['all_flights']
                if existing_progress.get('selected_flights'):
                    st.session_state.selected_flights = existing_progress['selected_flights']
                if existing_progress.get('search_id'):
                    st.session_state.search_id = existing_progress['search_id']
                if existing_progress.get('flight_selection_confirmed'):
                    st.session_state.review_confirmed = True
                if existing_progress.get('all_reranks_completed'):
                    st.session_state.all_reranks_completed = True
                    st.session_state.cross_validation_completed = True
                print(f"[PHASE] Restored session progress for {effective_token[:12]}***")
        except Exception as e:
            print(f"[PHASE] Session restore skipped: {e}")
        st.session_state.session_restored = True

elif raw_url_id:
    # ----------------------------------------------------------------
    # LEGACY TOKEN FLOW  (?id=DEMO / ?id=DATA / ?id=GA01 / etc.)
    # ----------------------------------------------------------------
    token_from_url = raw_url_id
    from backend.db import validate_token
    token_status = validate_token(token_from_url)
    st.session_state.token = token_from_url
    st.session_state.token_valid = token_status['valid']
    st.session_state.token_message = token_status['message']

    from pilot_tokens import is_pilot_token, get_token_group, get_rerank_targets
    if is_pilot_token(token_from_url):
        st.session_state.token_group = get_token_group(token_from_url)
        st.session_state.rerank_targets = get_rerank_targets(token_from_url)
        print(f"[PILOT] Token {token_from_url} - Group {st.session_state.token_group}, "
              f"Targets: {st.session_state.rerank_targets}")
    else:
        st.session_state.token_group = None
        st.session_state.rerank_targets = []

    # Restore progress on page refresh
    if 'session_restored' not in st.session_state:
        from backend.db import get_session_progress
        existing_progress = get_session_progress(token_from_url)
        if existing_progress:
            st.session_state.session_id = existing_progress['session_id']
            if existing_progress.get('user_prompt'):
                st.session_state.user_prompt = existing_progress['user_prompt']
            if existing_progress.get('all_flights'):
                st.session_state.all_flights = existing_progress['all_flights']
            if existing_progress.get('selected_flights'):
                st.session_state.selected_flights = existing_progress['selected_flights']
            if existing_progress.get('search_id'):
                st.session_state.search_id = existing_progress['search_id']
            if existing_progress.get('flight_selection_confirmed'):
                st.session_state.review_confirmed = True
            if existing_progress.get('current_rerank_index'):
                st.session_state.current_rerank_index = existing_progress['current_rerank_index']
            if existing_progress.get('completed_reranks'):
                st.session_state.completed_reranks = existing_progress['completed_reranks']
            if existing_progress.get('all_reranks_completed'):
                st.session_state.all_reranks_completed = True
                st.session_state.cross_validation_completed = True
            print(f"[PILOT] Restored session progress for {token_from_url}")
        st.session_state.session_restored = True


# ============================================================================
# FLIGHT CLIENT
# ============================================================================
from backend.flight_search import FlightSearchClient

def get_flight_client():
    load_dotenv(override=True)
    return FlightSearchClient()

flight_client = get_flight_client()


# ============================================================================
# RENDER: HEADER + TOKEN GATE (always)
# ============================================================================
from frontend.components.header import render_header, render_token_gate

render_header()
render_token_gate()


# ============================================================================
# RENDER: SEARCH FORM (always)
# ============================================================================
from frontend.pages.search import render_search_section

render_search_section(STATIC_ROUTE_DAY_OPTIONS, flight_client, STATIC_FLIGHTS)


# ============================================================================
# RENDER: RESULTS (conditional on flights being loaded)
# ============================================================================
if st.session_state.all_flights:
    if st.session_state.outbound_submitted:
        from frontend.pages.results import render_completion_section
        render_completion_section()
    else:
        from frontend.pages.ranking import render_ranking_section
        render_ranking_section()


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("Built for flight ranking research \u2022 Data collected for algorithm evaluation \u2022 Contact: listen.cornell@gmail.com")
