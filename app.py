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

# Import modules
from backend.flight_search import FlightSearchClient
from backend.prompt_parser import parse_flight_prompt_with_llm, get_test_api_fallback
from backend.utils.parse_duration import parse_duration_to_minutes


load_dotenv()


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
        # New API (Streamlit 1.30+)
        return st.query_params.get(key, default)
    except AttributeError:
        # Old API (Streamlit < 1.30)
        try:
            params = st.experimental_get_query_params()
            values = params.get(key, [default])
            return values[0] if values else default
        except Exception:
            return default

def clear_query_params():
    """Clear query parameters, compatible with both old and new Streamlit versions."""
    try:
        # New API (Streamlit 1.30+)
        st.query_params.clear()
    except AttributeError:
        # Old API (Streamlit < 1.30)
        st.experimental_set_query_params()

# Helper function to format price display
def format_price(price):
    """Format price for display, showing 'N/A' for 0 or None."""
    if price is None or price == 0:
        return "N/A"
    return f"${price:.0f}"

# Initialize database (create tables if they don't exist)
try:
    from backend.db import init_db
    init_db()
except Exception as e:
    # Don't crash the app if database setup fails, just log it
    print(f"Database initialization: {str(e)}")

# Seed cross-validation data for DEMO and DATA tokens (runs once)
try:
    from seed_cross_validation import seed_cross_validation_data
    seed_cross_validation_data()
except Exception as e:
    print(f"Cross-validation seed: {str(e)}")

AIRLINE_NAMES = {}

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
# ============================================================================
# Auto-fill from Prolific URL parameter (set study URL as
# https://listen-cornell3.streamlit.app/?PROLIFIC_PID={{%PROLIFIC_PID%}})
if not st.session_state.get('prolific_id'):
    _auto_pid = get_query_param('PROLIFIC_PID', '') or get_query_param('prolific_pid', '')
    if _auto_pid:
        st.session_state.prolific_id = _auto_pid.strip()

if not st.session_state.get('prolific_id'):
    from frontend.pages.prolific_gate import render_prolific_id_gate
    render_prolific_id_gate()
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
# SESSION TOKEN (keyed off Prolific ID)
# ============================================================================
prolific_id = st.session_state.prolific_id
effective_token = f"PHASEONE_{prolific_id}"

st.session_state.token = effective_token
st.session_state.token_valid = True
st.session_state.token_group = 'A'
st.session_state.rerank_targets = []

# Restore progress on page refresh
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
    except Exception as e:
        print(f"[SESSION] Restore skipped: {e}")
    st.session_state.session_restored = True


if 'all_flights' not in st.session_state:  # Changed from 'flights' to 'all_flights'
    st.session_state.all_flights = []
if 'selected_flights' not in st.session_state:  # Changed from 'shortlist'
    st.session_state.selected_flights = []
if 'parsed_params' not in st.session_state:
    st.session_state.parsed_params = None
if 'search_mode' not in st.session_state:
    st.session_state.search_mode = "ai"
if 'origin_search_results' not in st.session_state:
    st.session_state.origin_search_results = []
if 'origin_iata_map' not in st.session_state:
    st.session_state.origin_iata_map = {}
if 'dest_search_results' not in st.session_state:
    st.session_state.dest_search_results = []
if 'dest_iata_map' not in st.session_state:
    st.session_state.dest_iata_map = {}
if 'csv_generated' not in st.session_state:
    st.session_state.csv_generated = False
# New: for return flights
if 'all_return_flights' not in st.session_state:
    st.session_state.all_return_flights = []
if 'selected_return_flights' not in st.session_state:
    st.session_state.selected_return_flights = []
if 'has_return' not in st.session_state:
    st.session_state.has_return = False
# Version counters for dynamic rank updates
if 'outbound_sort_version' not in st.session_state:
    st.session_state.outbound_sort_version = 0
if 'return_sort_version' not in st.session_state:
    st.session_state.return_sort_version = 0
if 'single_sort_version' not in st.session_state:
    st.session_state.single_sort_version = 0
# Sort direction tracking (for arrows)
if 'sort_price_dir' not in st.session_state:
    st.session_state.sort_price_dir = 'asc'  # 'asc' or 'desc'
if 'sort_duration_dir' not in st.session_state:
    st.session_state.sort_duration_dir = 'asc'
if 'sort_price_dir_ret' not in st.session_state:
    st.session_state.sort_price_dir_ret = 'asc'
if 'sort_duration_dir_ret' not in st.session_state:
    st.session_state.sort_duration_dir_ret = 'asc'
if 'sort_price_dir_single' not in st.session_state:
    st.session_state.sort_price_dir_single = 'asc'
if 'sort_duration_dir_single' not in st.session_state:
    st.session_state.sort_duration_dir_single = 'asc'
# Submission tracking
if 'outbound_submitted' not in st.session_state:
    st.session_state.outbound_submitted = False
if 'return_submitted' not in st.session_state:
    st.session_state.return_submitted = False
# Checkbox version counter (incremented when X button removes a flight)
if 'checkbox_version' not in st.session_state:
    st.session_state.checkbox_version = 0
if 'csv_data_outbound' not in st.session_state:
    st.session_state.csv_data_outbound = None
if 'csv_data_return' not in st.session_state:
    st.session_state.csv_data_return = None
if 'review_confirmed' not in st.session_state:
    st.session_state.review_confirmed = False
if 'cross_validation_completed' not in st.session_state:
    st.session_state.cross_validation_completed = False
# Pilot study: Sequential re-ranking state
if 'token_group' not in st.session_state:
    st.session_state.token_group = None
if 'rerank_targets' not in st.session_state:
    st.session_state.rerank_targets = []
if 'current_rerank_index' not in st.session_state:
    st.session_state.current_rerank_index = 0
if 'completed_reranks' not in st.session_state:
    st.session_state.completed_reranks = []
if 'all_reranks_completed' not in st.session_state:
    st.session_state.all_reranks_completed = False
if 'survey_completed' not in st.session_state:
    st.session_state.survey_completed = False
if 'completion_page_dismissed' not in st.session_state:
    st.session_state.completion_page_dismissed = False
# Filter state
if 'filter_airlines' not in st.session_state:
    st.session_state.filter_airlines = []
if 'filter_connections' not in st.session_state:
    st.session_state.filter_connections = []
if 'filter_price_range' not in st.session_state:
    st.session_state.filter_price_range = None
if 'filter_duration_range' not in st.session_state:
    st.session_state.filter_duration_range = None
if 'filter_departure_time_range' not in st.session_state:
    st.session_state.filter_departure_time_range = None
if 'filter_arrival_time_range' not in st.session_state:
    st.session_state.filter_arrival_time_range = None
if 'filter_origins' not in st.session_state:
    st.session_state.filter_origins = []
if 'filter_destinations' not in st.session_state:
    st.session_state.filter_destinations = []
# Counter to force filter widgets to reset
if 'filter_reset_counter' not in st.session_state:
    st.session_state.filter_reset_counter = 0

def get_flight_client():
    load_dotenv(override=True)
    return FlightSearchClient()

flight_client = get_flight_client()


# ============================================================================
# RENDER: HEADER + TOKEN GATE (always)
# ============================================================================
from frontend.components.header import render_header

render_header()


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
