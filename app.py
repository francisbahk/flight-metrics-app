"""
Flight Evaluation Web App - Clean Suno-style Interface
Collect user feedback on algorithm-ranked flights for research.
"""
import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import uuid
import pandas as pd
import io
from datetime import datetime
from dotenv import load_dotenv
from streamlit_sortables import sort_items

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

# Import modules
# from backend.amadeus_client import AmadeusClient  # Legacy - replaced with unified client
from backend.flight_search import FlightSearchClient
from backend.prompt_parser import parse_flight_prompt_with_llm, get_test_api_fallback
from backend.utils.parse_duration import parse_duration_to_minutes

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
# Enable automatic S3 backup on session completion (set via env var or Streamlit secrets)
def get_auto_backup_enabled():
    """Check AUTO_BACKUP_ENABLED from Streamlit secrets first, then environment."""
    try:
        if hasattr(st, 'secrets') and 'AUTO_BACKUP_ENABLED' in st.secrets:
            return str(st.secrets['AUTO_BACKUP_ENABLED']).lower() == 'true'
    except Exception:
        pass
    return os.getenv('AUTO_BACKUP_ENABLED', 'false').lower() == 'true'

AUTO_BACKUP_ENABLED = get_auto_backup_enabled()


def trigger_backup_on_completion(token: str):
    """
    Trigger S3 backup after session completion.
    Runs in background thread to not block UI.
    Only runs if AUTO_BACKUP_ENABLED=true in environment.
    """
    if not AUTO_BACKUP_ENABLED:
        print(f"[BACKUP] Auto-backup disabled. Session {token} completed.")
        return

    import threading

    def run_backup():
        try:
            from backup_db import run_backup as do_backup
            print(f"[BACKUP] Starting backup after session {token} completed...")
            success = do_backup(on_completion=True)
            if success:
                print(f"[BACKUP] Backup completed successfully for session {token}")
            else:
                print(f"[BACKUP] Backup failed for session {token}")
        except Exception as e:
            print(f"[BACKUP] Error running backup: {e}")

    # Run in background thread
    backup_thread = threading.Thread(target=run_backup, daemon=True)
    backup_thread.start()

# ============================================================================
# BACKWARDS-COMPATIBLE QUERY PARAMS (Streamlit version compatibility)
# ============================================================================
def get_query_param(key, default=None):
    """Get query parameter, compatible with both old and new Streamlit versions."""
    try:
        # New API (Streamlit 1.30+)
        return st.query_params.get(key, default)
    except AttributeError:
        # Old API (Streamlit < 1.30)
        params = st.experimental_get_query_params()
        values = params.get(key, [default])
        return values[0] if values else default

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

# Airline code to name mapping (common IATA codes)
AIRLINE_NAMES = {
    'AA': 'American Airlines',
    'DL': 'Delta Air Lines',
    'UA': 'United Airlines',
    'WN': 'Southwest Airlines',
    'B6': 'JetBlue Airways',
    'AS': 'Alaska Airlines',
    'NK': 'Spirit Airlines',
    'F9': 'Frontier Airlines',
    'G4': 'Allegiant Air',
    'SY': 'Sun Country Airlines',
    'AC': 'Air Canada',
    'AM': 'Aeromexico',
    'BA': 'British Airways',
    'LH': 'Lufthansa',
    'AF': 'Air France',
    'KL': 'KLM',
    'IB': 'Iberia',
    'AY': 'Finnair',
    'SK': 'SAS',
    'TP': 'TAP Air Portugal',
    'LX': 'Swiss International Air Lines',
    'OS': 'Austrian Airlines',
    'SN': 'Brussels Airlines',
    'AZ': 'ITA Airways',
    'EI': 'Aer Lingus',
    'FR': 'Ryanair',
    'U2': 'easyJet',
    'EW': 'Eurowings',
    'VY': 'Vueling',
    'I2': 'Iberia Express',
    'UX': 'Air Europa',
    'TO': 'Transavia',
    'W6': 'Wizz Air',
    'QR': 'Qatar Airways',
    'EK': 'Emirates',
    'EY': 'Etihad Airways',
    'TK': 'Turkish Airlines',
    'SQ': 'Singapore Airlines',
    'CX': 'Cathay Pacific',
    'JL': 'Japan Airlines',
    'NH': 'All Nippon Airways',
    'KE': 'Korean Air',
    'OZ': 'Asiana Airlines',
    'TG': 'Thai Airways',
    'MH': 'Malaysia Airlines',
    'BR': 'EVA Air',
    'CI': 'China Airlines',
    'CA': 'Air China',
    'MU': 'China Eastern',
    'CZ': 'China Southern',
    'QF': 'Qantas',
    'NZ': 'Air New Zealand',
    'LA': 'LATAM Airlines',
    'AR': 'Aerolineas Argentinas',
    'CM': 'Copa Airlines',
    'AV': 'Avianca',
    'SA': 'South African Airways',
    'ET': 'Ethiopian Airlines',
    'MS': 'EgyptAir',
    'WY': 'Oman Air',
    'GF': 'Gulf Air',
}

# ============================================================================
# STATIC FLIGHT DATABASE (pilot study)
# Pre-fetched once via fetch_static_flights.py for March 1-7 2026 (Sun-Sat).
# ============================================================================
import json as _json
STATIC_FLIGHTS = []
STATIC_ORIGINS = []
STATIC_DESTINATIONS = []
DAYS_OF_WEEK = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
try:
    _static_path = os.path.join(os.path.dirname(__file__), 'static_flights.json')
    with open(_static_path) as _f:
        STATIC_FLIGHTS = _json.load(_f)
    STATIC_ORIGINS = sorted(set(f['origin'] for f in STATIC_FLIGHTS))
    STATIC_DESTINATIONS = sorted(set(f['destination'] for f in STATIC_FLIGHTS))
    print(f"✓ Loaded {len(STATIC_FLIGHTS)} static flights "
          f"({len(STATIC_ORIGINS)} origins, {len(STATIC_DESTINATIONS)} destinations)")
except FileNotFoundError:
    print("⚠ static_flights.json not found. Run fetch_static_flights.py first.")


def build_manual_parsed(origins, destinations, departure_date, return_date=None):
    """
    Build the same parsed dict structure that parse_flight_prompt_with_llm returns,
    from manual form inputs.
    """
    # origins/destinations are plain IATA codes (e.g. ["JFK", "EWR"])
    origin_codes = [c.strip().upper() for c in origins if c.strip()]
    dest_codes = [c.strip().upper() for c in destinations if c.strip()]

    dep_dates = [departure_date.strftime("%Y-%m-%d")]
    ret_dates = [return_date.strftime("%Y-%m-%d")] if return_date else []

    return {
        "parsed_successfully": True,
        "origins": origin_codes,
        "destinations": dest_codes,
        "departure_dates": dep_dates,
        "return_dates": ret_dates,
        "return_date": ret_dates[0] if ret_dates else None,
        "preferences": {
            "prefer_direct": False,
            "prefer_cheap": False,
            "prefer_fast": False,
            "avoid_early_departures": False,
            "min_connection_time": 0,
            "max_layover_time": 10000,
            "preferred_airlines": [],
            "avoid_airports": [],
            "fly_america_act": False,
        },
        "constraints": {
            "latest_arrival": None,
            "earliest_departure": None,
        },
        "original_prompt": f"Manual search: {', '.join(origin_codes)} to {', '.join(dest_codes)} on {dep_dates[0]}",
    }

def get_airline_name(code):
    """Convert airline IATA code to full name."""
    # First try session state (Amadeus API lookup)
    if 'airline_names' in st.session_state:
        if code in st.session_state.airline_names:
            return st.session_state.airline_names[code]

    # Fallback to hardcoded mapping
    return AIRLINE_NAMES.get(code, code)  # Return code if not found


def detect_codeshares(flights):
    """
    Detect which flights are codeshares based on identical:
    - departure_time
    - arrival_time
    - origin
    - destination

    Returns dict mapping flight index -> is_codeshare boolean
    """
    codeshare_map = {}

    for i, flight in enumerate(flights):
        is_codeshare = False

        # Check if any OTHER flight has same times + route
        for j, other_flight in enumerate(flights):
            if i == j:  # Skip self-comparison
                continue

            if (flight['departure_time'] == other_flight['departure_time'] and
                flight['arrival_time'] == other_flight['arrival_time'] and
                flight['origin'] == other_flight['origin'] and
                flight['destination'] == other_flight['destination']):
                is_codeshare = True
                break

        codeshare_map[i] = is_codeshare

    return codeshare_map


def apply_filters(flights, airlines=None, connections=None, price_range=None, duration_range=None,
                  departure_range=None, arrival_range=None, origins=None, destinations=None):
    """
    Filter flights based on user-selected criteria.

    Args:
        flights: List of flight dicts
        airlines: List of airline codes to include (None = all)
        connections: List of connection counts to include (None = all)
        price_range: Tuple of (min_price, max_price) (None = all)
        duration_range: Tuple of (min_duration_min, max_duration_min) (None = all)
        departure_range: Tuple of (min_hour, max_hour) in 24h format (None = all)
        arrival_range: Tuple of (min_hour, max_hour) in 24h format (None = all)
        origins: List of origin airport codes to include (None = all)
        destinations: List of destination airport codes to include (None = all)

    Returns:
        Filtered list of flights
    """
    filtered = flights

    # Filter by airline
    if airlines and len(airlines) > 0:
        filtered = [f for f in filtered if f['airline'] in airlines]

    # Filter by connections
    if connections is not None and len(connections) > 0:
        filtered = [f for f in filtered if f['stops'] in connections]

    # Filter by origin
    if origins and len(origins) > 0:
        filtered = [f for f in filtered if f['origin'] in origins]

    # Filter by destination
    if destinations and len(destinations) > 0:
        filtered = [f for f in filtered if f['destination'] in destinations]

    # Filter by price
    if price_range:
        min_price, max_price = price_range
        filtered = [f for f in filtered if min_price <= f['price'] <= max_price]

    # Filter by duration
    if duration_range:
        min_dur, max_dur = duration_range
        filtered = [f for f in filtered if min_dur <= f['duration_min'] <= max_dur]

    # Filter by departure time
    if departure_range:
        from datetime import datetime
        min_hour, max_hour = departure_range
        filtered_by_dept = []
        for f in filtered:
            dept_dt = datetime.fromisoformat(f['departure_time'].replace('Z', '+00:00'))
            hour = dept_dt.hour + dept_dt.minute / 60.0  # Convert to decimal hours
            if min_hour <= hour <= max_hour:
                filtered_by_dept.append(f)
        filtered = filtered_by_dept

    # Filter by arrival time
    if arrival_range:
        from datetime import datetime
        min_hour, max_hour = arrival_range
        filtered_by_arr = []
        for f in filtered:
            arr_dt = datetime.fromisoformat(f['arrival_time'].replace('Z', '+00:00'))
            hour = arr_dt.hour + arr_dt.minute / 60.0  # Convert to decimal hours
            if min_hour <= hour <= max_hour:
                filtered_by_arr.append(f)
        filtered = filtered_by_arr

    return filtered


# CSV Generation Function
def generate_flight_csv(all_flights, selected_flights, k=5):
    """
    Generate CSV with flight data and rankings.

    Args:
        all_flights: List of all flight dicts shown to user
        selected_flights: List of flight dicts user selected (in ranked order)
        k: Number of top flights selected (default 5)

    Returns:
        CSV string
    """
    reference_date = datetime.now()
    N = len(all_flights)
    unranked_value = ((k + 1) + N) / 2

    csv_rows = []

    # Create a mapping using unique composite key (ID + departure time + price)
    # This handles duplicate flight IDs (codeshares, same flight different dates)
    def get_flight_key(f):
        return f"{f['id']}_{f.get('departure_time', '')}_{f.get('price', 0)}"

    selected_keys = {get_flight_key(f): (idx + 1) for idx, f in enumerate(selected_flights)}

    for idx, flight in enumerate(all_flights):
        flight_key = get_flight_key(flight)
        # Determine if this flight was ranked
        is_best = flight_key in selected_keys
        rank = selected_keys.get(flight_key, unranked_value)

        # Generate unique_id
        unique_id = f"{flight['origin']}_{flight['destination']}{idx + 1}"

        # Get airline name (convert code to full name)
        name = get_airline_name(flight['airline'])

        # Parse departure and arrival times
        try:
            dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
            arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))

            # Format for display
            dept_time_str = dept_dt.strftime("%I:%M %p on %a, %b %d")
            arr_time_str = arr_dt.strftime("%I:%M %p on %a, %b %d")

            # Calculate seconds from reference date
            dept_seconds = (dept_dt.date() - reference_date.date()).days * 86400 + \
                          (dept_dt.hour * 3600 + dept_dt.minute * 60 + dept_dt.second)
            arr_seconds = (arr_dt.date() - reference_date.date()).days * 86400 + \
                         (arr_dt.hour * 3600 + arr_dt.minute * 60 + arr_dt.second)

            # Format departure_dt and arrival_dt with year 1900 (as per example)
            dept_dt_formatted = f"1900-{dept_dt.strftime('%m-%d %H:%M:%S')}"
            arr_dt_formatted = f"1900-{arr_dt.strftime('%m-%d %H:%M:%S')}"

        except Exception as e:
            dept_time_str = flight['departure_time']
            arr_time_str = flight['arrival_time']
            dept_seconds = 0
            arr_seconds = 0
            dept_dt_formatted = ""
            arr_dt_formatted = ""

        # Build row - ensure all text fields have no tabs or newlines
        def clean_text(text):
            """Remove tabs and newlines from text to prevent CSV column misalignment"""
            if isinstance(text, str):
                return text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            return text

        row = {
            'unique_id': clean_text(unique_id),
            'is_best': 1 if is_best else 0,  # Use 1/0 instead of True/False
            'rank': rank,
            'name': clean_text(name),
            'origin': clean_text(flight['origin']),
            'destination': clean_text(flight['destination']),
            'departure_time': clean_text(dept_time_str),
            'arrival_time': clean_text(arr_time_str),
            'duration': clean_text(f"{flight['duration_min']//60} hr {flight['duration_min']%60} min"),
            'stops': float(flight['stops']),
            'price': float(flight['price']),
            'dis_from_origin': 0.0,
            'dis_from_dest': 0.0,
            'departure_dt': clean_text(dept_dt_formatted),
            'departure_seconds': dept_seconds,
            'arrival_dt': clean_text(arr_dt_formatted),
            'arrival_seconds': arr_seconds,
            'duration_min': flight['duration_min']
        }
        csv_rows.append(row)

    # Convert to DataFrame with explicit column order
    df = pd.DataFrame(csv_rows)

    # Ensure columns are in the correct order
    columns_order = [
        'unique_id', 'is_best', 'rank', 'name', 'origin', 'destination',
        'departure_time', 'arrival_time', 'duration', 'stops', 'price',
        'dis_from_origin', 'dis_from_dest', 'departure_dt', 'departure_seconds',
        'arrival_dt', 'arrival_seconds', 'duration_min'
    ]
    df = df[columns_order]

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, sep='\t')  # Tab-separated as per example
    return csv_buffer.getvalue()

# Page config
st.set_page_config(
    page_title="Flight Ranker",
    page_icon="✈️",
    layout="wide"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# TOKEN-BASED AUTH: Read and validate token from URL
if 'token' not in st.session_state:
    st.session_state.token = None
if 'token_valid' not in st.session_state:
    st.session_state.token_valid = False
if 'token_message' not in st.session_state:
    st.session_state.token_message = ''

# Get token from URL parameter (?id=TOKEN)
# Re-validate on every page load to detect if token was used
token_from_url = get_query_param('id')
if token_from_url:
    # Validate token (checks database to see if it's been used)
    from backend.db import validate_token
    token_status = validate_token(token_from_url)

    st.session_state.token = token_from_url
    st.session_state.token_valid = token_status['valid']
    st.session_state.token_message = token_status['message']

    # Pilot study: Detect token group and re-rank assignments
    from pilot_tokens import is_pilot_token, get_token_group, get_rerank_targets
    if is_pilot_token(token_from_url):
        st.session_state.token_group = get_token_group(token_from_url)
        st.session_state.rerank_targets = get_rerank_targets(token_from_url)
        print(f"[PILOT] Token {token_from_url} - Group {st.session_state.token_group}, Targets: {st.session_state.rerank_targets}")
    else:
        st.session_state.token_group = None
        st.session_state.rerank_targets = []

    # Session persistence: Try to restore progress on page refresh
    if 'session_restored' not in st.session_state:
        from backend.db import get_session_progress
        existing_progress = get_session_progress(token_from_url)

        if existing_progress:
            # Restore session state from database
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

            st.session_state.session_restored = True
            print(f"[PILOT] Restored session progress for {token_from_url}")
        else:
            st.session_state.session_restored = True  # Mark as checked, even if no progress found

# COMMENTED OUT - No longer collecting name/email
# if 'user_name' not in st.session_state:
#     st.session_state.user_name = ''
# if 'user_email' not in st.session_state:
#     st.session_state.user_email = ''
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

# Initialize Flight Search Client (supports Amadeus and SerpAPI)
def get_flight_client():
    # Ensure .env is loaded fresh
    load_dotenv(override=True)
    return FlightSearchClient()

# Don't cache initially to avoid credential issues
flight_client = get_flight_client()

# Custom CSS for Suno-like clean design
st.markdown("""
<style>
    /* Prevent auto-scroll on interactions */
    html {
        scroll-behavior: auto !important;
        overflow-anchor: none !important;
    }
    body {
        overflow-anchor: none !important;
    }
    .main {
        overflow-anchor: none !important;
    }

    /* Prevent checkboxes from triggering scroll */
    .stCheckbox {
        overflow-anchor: none !important;
    }
    .stCheckbox input[type="checkbox"] {
        overflow-anchor: none !important;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .flight-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .shortlist-area {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 1.5rem;
        min-height: 400px;
    }
    /* Compact flight list */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="column"] {
        padding: 0.2rem !important;
    }
    .stMarkdown p {
        margin-bottom: 0.2rem !important;
        line-height: 1.3 !important;
    }
    /* Make checkboxes 15% larger */
    input[type="checkbox"] {
        transform: scale(1.15);
        cursor: pointer;
    }
</style>
<script>
    // Prevent scrolling when checkboxes are clicked
    (function() {
        // Store current scroll position before any checkbox interaction
        let scrollPosition = 0;

        // Function to attach scroll prevention to checkboxes
        function attachScrollPrevention() {
            const checkboxes = document.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(function(checkbox) {
                if (!checkbox.hasAttribute('data-scroll-prevention')) {
                    checkbox.setAttribute('data-scroll-prevention', 'true');

                    // Save scroll position before click
                    checkbox.addEventListener('mousedown', function() {
                        scrollPosition = window.scrollY;
                    });

                    // Restore scroll position after click
                    checkbox.addEventListener('change', function() {
                        setTimeout(function() {
                            window.scrollTo(0, scrollPosition);
                        }, 0);
                    });

                    // Also prevent on click to handle first-time clicks
                    checkbox.addEventListener('click', function() {
                        setTimeout(function() {
                            window.scrollTo(0, scrollPosition);
                        }, 10);
                    });
                }
            });
        }

        // Run immediately to catch early checkboxes
        attachScrollPrevention();

        // Run after DOM is loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', attachScrollPrevention);
        }

        // Observe for dynamically added checkboxes
        const observer = new MutationObserver(attachScrollPrevention);
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    })();
</script>
""", unsafe_allow_html=True)

# Header with animated title flip using components.html for full control
components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: 'Source Sans Pro', sans-serif;
    }

    .main-title {
        text-align: center;
        width: 100%;
        padding: 20px 0;
        font-size: 2.5rem;
        font-weight: 600;
        transition: all 0.8s ease-in-out;
    }

    .title-wrapper {
        display: inline-block;
    }

    #changing-word {
        display: inline-block;
        transition: all 2s ease-in-out;
        font-style: italic; /* Always italic */
    }

    #ai-prefix {
        display: inline-block;
        transition: opacity 1.5s ease-in-out, transform 1.5s ease-in-out;
        font-style: italic;
    }

    #flight-word {
        display: inline-block;
    }

    .subtitle {
        text-align: center;
        width: 100%;
        margin-top: 10px;
        font-size: 1.1rem;
        color: #555;
    }

    .subtitle span {
        display: inline-block;
        transition: all 0.8s ease-in-out;
        font-style: italic; /* Always italic */
    }
</style>
</head>
<body>

<div class="main-title">
    <div class="title-wrapper">
        <span>✈️ <span id="ai-prefix" style="opacity: 0; transform: translateX(-20px);"></span> <span id="flight-word">Flight</span> <span id="changing-word">Ranker</span></span>
    </div>
</div>

<div class="subtitle" id="subtitle-content">
    Share your flight preferences to <span id="word1">help</span> <span id="word2">build</span> <span id="word3">better</span> personalized <span id="word4">ranking</span> <span id="word5">systems</span>
</div>

<script>
(function() {
    const CYCLE_DURATION = 30000; // 30 seconds for slower animations
    const START_DELAY = 1000; // Start animation after 1 second
    const startTime = Date.now() + START_DELAY; // Offset start time

    var titleAnimating = false;
    var subtitleAnimating = false;
    var lastPhase = 0;

    function smoothFlip(element, newText, duration) {
        element.style.transform = 'rotateX(90deg)';
        element.style.opacity = '0';

        setTimeout(function() {
            element.textContent = newText;
            element.style.transform = 'rotateX(0deg)';
            element.style.opacity = '1';
        }, duration / 2);
    }

    function animateTitle() {
        const changingWord = document.getElementById('changing-word');
        const aiPrefix = document.getElementById('ai-prefix');
        if (!changingWord) return;

        const now = Date.now();
        if (now < startTime) return; // Don't animate until after delay

        const cyclePosition = ((now - startTime) % CYCLE_DURATION) / CYCLE_DURATION;
        const currentPhase = Math.floor(cyclePosition * 100);

        // Phase 1 (0-25%): Show "Flight Ranker"
        if (cyclePosition < 0.25) {
            if (lastPhase >= 25 && changingWord.textContent !== 'Ranker') {
                titleAnimating = true;
                // Slide out AI-Selected and flip word simultaneously
                aiPrefix.style.opacity = '0';
                aiPrefix.style.transform = 'translateX(-20px)';
                changingWord.style.transform = 'rotateX(90deg)';
                changingWord.style.opacity = '0';
                setTimeout(function() {
                    changingWord.textContent = 'Ranker';
                    changingWord.style.transform = 'rotateX(0deg)';
                    changingWord.style.opacity = '1';
                    aiPrefix.textContent = '';
                    titleAnimating = false;
                }, 600);
            }
        }
        // Phase 2 (25-32%): Slide "Recommendations" in from right, then slide in "AI-Driven" from left
        else if (cyclePosition >= 0.25 && cyclePosition < 0.32) {
            if (!titleAnimating && changingWord.textContent === 'Ranker') {
                titleAnimating = true;
                // First, slide out "Ranker" to the left
                changingWord.style.transform = 'translateX(-30px)';
                changingWord.style.opacity = '0';
                setTimeout(function() {
                    // Change text and slide in "Recommendations" from the right
                    changingWord.textContent = 'Recommendations';
                    changingWord.style.transform = 'translateX(30px)';
                    setTimeout(function() {
                        changingWord.style.transform = 'translateX(0)';
                        changingWord.style.opacity = '1';
                    }, 50);
                }, 600);
                // Then slide in "AI-Driven" from the left
                setTimeout(function() {
                    aiPrefix.textContent = 'AI-Driven';
                    aiPrefix.style.opacity = '1';
                    aiPrefix.style.transform = 'translateX(0)';
                    titleAnimating = false;
                }, 1800);
            }
        }
        // Phase 3 (32-80%): Hold "AI-Driven Flight Recommendations"
        else if (cyclePosition >= 0.32 && cyclePosition < 0.80) {
            // Just hold
        }
        // Phase 4 (80-85%): Slide out "AI-Driven" and slide "Recommendations" out, then slide "Ranker" back in
        else if (cyclePosition >= 0.80 && cyclePosition < 0.85) {
            if (!titleAnimating && aiPrefix.style.opacity !== '0') {
                titleAnimating = true;
                // Slide out AI-Driven to the left
                aiPrefix.style.opacity = '0';
                aiPrefix.style.transform = 'translateX(-20px)';
                // Slide out "Recommendations" to the right (matching its slide-in direction)
                setTimeout(function() {
                    changingWord.style.transform = 'translateX(30px)';
                    changingWord.style.opacity = '0';
                    setTimeout(function() {
                        // Change text and slide in "Ranker" from the right
                        changingWord.textContent = 'Ranker';
                        changingWord.style.transform = 'translateX(30px)';
                        setTimeout(function() {
                            changingWord.style.transform = 'translateX(0)';
                            changingWord.style.opacity = '1';
                            aiPrefix.textContent = '';
                            titleAnimating = false;  // Now subtitle can animate
                        }, 50);
                    }, 600);
                }, 600);
            }
        }

        lastPhase = currentPhase;
    }

    function animateSubtitleWord(wordId, newText, delay) {
        setTimeout(function() {
            const word = document.getElementById(wordId);
            if (word) {
                word.style.transform = 'rotateX(90deg)';
                word.style.opacity = '0';

                setTimeout(function() {
                    word.textContent = newText;
                    word.style.transform = 'rotateX(0deg)';
                    word.style.opacity = '1';
                }, 600);
            }
        }, delay);
    }

    function animateSubtitle() {
        if (titleAnimating) return; // Wait for title to finish

        const now = Date.now();
        if (now < startTime) return; // Don't animate until after delay

        const cyclePosition = ((now - startTime) % CYCLE_DURATION) / CYCLE_DURATION;

        // Start subtitle animation AFTER title is done (at 40%)
        if (cyclePosition >= 0.40 && cyclePosition < 0.45) {
            if (!subtitleAnimating) {
                subtitleAnimating = true;
                // Animate words one by one, left to right with more time per word
                animateSubtitleWord('word1', 'receive', 0);
                animateSubtitleWord('word2', 'smart', 800);
                animateSubtitleWord('word3', 'fast', 1600);
                animateSubtitleWord('word4', 'flight', 2400);
                animateSubtitleWord('word5', 'results', 3200);
                setTimeout(function() { subtitleAnimating = false; }, 4400);
            }
        }
        // Animate back to original at 88% (after title finishes at ~85%)
        else if (cyclePosition >= 0.88 && cyclePosition < 0.93) {
            if (!subtitleAnimating && !titleAnimating) {  // Wait for title to finish
                subtitleAnimating = true;
                // Revert to original words
                animateSubtitleWord('word1', 'help', 0);
                animateSubtitleWord('word2', 'build', 800);
                animateSubtitleWord('word3', 'better', 1600);
                animateSubtitleWord('word4', 'ranking', 2400);
                animateSubtitleWord('word5', 'systems', 3200);
                setTimeout(function() { subtitleAnimating = false; }, 4400);
            }
        }
    }

    // Run animations
    setInterval(function() {
        animateTitle();
        animateSubtitle();
    }, 100);
})();
</script>

</body>
</html>
""", height=150)

st.info("**Note:** This website is part of a pilot data-collection study. The information collected will be used to improve flight search tools.")

# TOKEN VALIDATION CHECK
# Special handling for DEMO and DATA tokens
if st.session_state.token and st.session_state.token.upper() in ["DEMO", "DATA"]:
    # Bypass validation for special tokens
    st.session_state.token_valid = True
    if st.session_state.token.upper() == "DEMO":
        st.info(f"🎯 **Demo Mode** - Explore the flight search tool freely!")
    else:
        st.info(f"📊 **Data Collection Mode** - Unlimited submissions enabled!")
elif not st.session_state.token:
    st.error("❌ **Access Denied: No Token Provided**")
    st.warning("This study requires a unique access token. Please use the link provided to you by the researchers, or use `?id=DEMO` for demo mode.")
    st.stop()
elif not st.session_state.token_valid:
    if 'already used' in st.session_state.token_message.lower():
        st.error("❌ **Access Not Granted: This token has already been used**")
        st.warning("This access link can only be used once. If you need to participate again, please contact the research team for a new token.")
    else:
        st.error(f"❌ **Access Denied: {st.session_state.token_message}**")
        st.warning("Please check your access link and try again, or contact the researchers if you believe this is an error.")
    st.stop()
else:
    # Show success message for valid token
    st.success(f"✅ Access granted! Token: {st.session_state.token}")

# How to Use section
st.markdown('<div id="how-to-use"></div>', unsafe_allow_html=True)
st.markdown("### 📖 How to Use")

# Container to hide content on survey pages
st.markdown('<div class="hideable-survey-content">', unsafe_allow_html=True)

st.markdown("""
1. **Describe your flight** - Enter your travel details in natural language, as if you are telling a flight itinerary manager how to book your ideal trip. What would you want them to know?
""")

# Tips for writing a good prompt (placed between steps 2 and 3)
with st.expander("💡 Tips for Writing a Good Prompt"):
    st.markdown("""
    💡 **Take some time to write your preferences** — imagine that the results will be reordered based on what you write. The preferences you write will be used in future research to evaluate how well algorithms return flights that align with your preferences, requirements, and persona.

    For example, you may describe your preferences with respect to key metrics:
    - **Price** - Are you budget-conscious or willing to pay more for reduced travel time?
    - **Duration** - Do you prefer the fastest route or are you flexible?
    - **Connections** - Direct flights only, or are layovers acceptable?
    - **Departure/Arrival Times** - Morning person or night owl? Business hours or flexible?
    - **Airlines** - Any preferences or airlines to avoid?

    **Examples of good preference statements:**
    - "Prioritize minimizing flight duration, but I'm flexible on price"
    - "I want the cheapest option, even if it means multiple connections"
    - "Direct flights only, departing in the evening, prefer United or JetBlue"
    - "Balance between price and duration, avoid red-eye flights"
    - "Would pay up to $300 more if it means being there on time" *(expressing trade-offs helps!)*

    The more specific you are about your priorities and trade-offs, the better we can understand your preferences!
    """)

st.markdown("""
2. **Review results** - Browse all available flights using Standard Search or AI Search. After you submit your prompt, use the filter sidebar on the left to narrow down options by price range, number of connections, flight duration, departure/arrival times, airlines, and airports. *Note: AI Search results may be less accurate while we continue to improve the system.*
3. **Select top 5** - Check the boxes next to your 5 favorite flights (for both outbound and return if applicable)
4. **Drag to rank** - Reorder your selections by dragging them in the right panel
5. **Submit** - Click submit to save your rankings (download as CSV optional)

**Note:** If your search includes a return flight, scroll down after the outbound flights to see the return flights section and submit those rankings separately.
""")

# # User Information Section (COMMENTED OUT - Now using token-based auth)
# st.markdown("### 👤 Your Information")
# st.caption("Required for contact")
#
# with st.form(key="user_info_form", clear_on_submit=False):
#     col_name, col_email = st.columns(2)
#
#     with col_name:
#         user_name = st.text_input(
#             "Full Name (First and Last)",
#             value=st.session_state.user_name
#         )
#
#     with col_email:
#         user_email = st.text_input(
#             "Email Address",
#             value=st.session_state.user_email
#         )
#
#     submit_info = st.form_submit_button("Save Information", use_container_width=True)
#
#     if submit_info:
#         st.session_state.user_name = user_name
#         st.session_state.user_email = user_email
#         st.success("✓ Information saved!")

# CSS for animated placeholder overlay
st.markdown("""
<style>
    .stTextArea {
        position: relative !important;
    }
    .stTextArea textarea {
        background-color: transparent !important;
        position: relative !important;
        z-index: 2 !important;
    }
    .anim-placeholder {
        position: absolute !important;
        top: 12px !important;
        left: 12px !important;
        right: 12px !important;
        bottom: 12px !important;
        color: #94a3b8 !important;
        font-family: 'Source Code Pro', monospace !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
        pointer-events: none !important;
        white-space: pre-wrap !important;
        overflow: hidden !important;
        z-index: 1 !important;
    }
    .anim-placeholder.hide {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Add CSS to reduce spacing between headers and inputs
st.markdown("""
<style>
    /* Reduce spacing for prompt section headers */
    div[data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.25rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Always show example prompts header
st.markdown("**Example prompts:**")

# Animated placeholder with carousel controls and toggle button
placeholder_html = r"""
<style>
    body {
        margin: 0;
        padding: 0;
        background: white;
    }
    .carousel-container {
        position: relative;
        border: 1px solid rgb(204, 204, 204);
        border-radius: 0.5rem;
        background: white;
        height: 150px;
    }
    #animBox {
        height: 100%;
        padding: 0.5rem 0.75rem;
        overflow: hidden;
        position: relative;
    }
    #animPlaceholder {
        font-family: 'Source Code Pro', monospace;
        font-size: 14px;
        line-height: 1.6;
        color: rgba(49, 51, 63, 0.4);
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow-wrap: break-word;
        overflow: hidden;
        max-width: 100%;
        transition: opacity 0.5s ease-out;
        pointer-events: none;
    }
    /* Controls are positioned on carousel-container (doesn't scroll) */
    .carousel-nav {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background: rgba(0, 0, 0, 0.15);
        color: rgba(255, 255, 255, 0.4);
        border: none;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        cursor: pointer;
        font-size: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
        z-index: 10;
        pointer-events: auto;
    }
    .carousel-nav:hover {
        background: rgba(0, 0, 0, 0.3);
        color: rgba(255, 255, 255, 0.7);
    }
    .carousel-prev {
        left: 10px;
    }
    .carousel-next {
        right: 10px;
    }
    .carousel-indicator {
        position: absolute;
        bottom: 8px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        gap: 6px;
        pointer-events: auto;
        z-index: 10;
    }
    .indicator-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.15);
        cursor: pointer;
        transition: background 0.2s;
    }
    .indicator-dot.active {
        background: rgba(0, 0, 0, 0.35);
    }
    .indicator-dot:hover {
        background: rgba(0, 0, 0, 0.25);
    }
    .toggle-btn {
        position: absolute;
        bottom: 8px;
        right: 10px;
        background: rgba(0, 0, 0, 0.15);
        border: none;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        color: rgba(255, 255, 255, 0.4);
        transition: all 0.2s;
        z-index: 10;
        pointer-events: auto;
    }
    .toggle-btn:hover {
        background: rgba(0, 0, 0, 0.3);
        color: rgba(255, 255, 255, 0.7);
    }
    .toggle-btn.static-mode {
        background: rgba(0, 0, 0, 0.25);
        color: rgba(255, 255, 255, 0.6);
    }
    #animBox.scrollable {
        overflow-y: auto;
    }
    #animBox.scrollable::-webkit-scrollbar {
        width: 6px;
    }
    #animBox.scrollable::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 3px;
    }
    #animBox.scrollable::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 3px;
    }
</style>
<div class="carousel-container">
    <!-- Controls outside scrollable area (stay fixed when scrolling) -->
    <button class="carousel-nav carousel-prev" onclick="prevPrompt()">‹</button>
    <button class="carousel-nav carousel-next" onclick="nextPrompt()">›</button>
    <div class="carousel-indicator">
        <span class="indicator-dot active" onclick="goToPrompt(0)"></span>
        <span class="indicator-dot" onclick="goToPrompt(1)"></span>
    </div>
    <button class="toggle-btn" id="toggleBtn" onclick="toggleTypewriter()" title="Pause animation">▐▐</button>
    <!-- Scrollable content area -->
    <div id="animBox">
        <div id="animPlaceholder"></div>
    </div>
</div>
<script>
    const prompts = [
        `I would like to take a trip from Chicago to New York City with my brother the weekend of October 11, 2025. Time is of the essence, so I prefer to maximize my time there. I will be leaving from Times Square area, so I can fly from any of the three major airports. I heavily prefer to fly into ORD.
I do not feel the need to strictly minimize cost; however, I would prefer to keep the fare to under 400 dollars. Obviously, if different flights meet my requirements, I prefer the cheaper one. I prefer direct flights.
I would like to maximize my time in NYC on Sunday. It would be ideal to leave on the second-to-last flight leaving from the departure airport to Chicago, in case of delays and cancellations. Worst case, I would like there to be an early Monday morning departure to Chicago from the airport, in case of cancellations.
I have no preference for airline. I would prefer to not leave NYC before 5 PM. I am okay with an early morning departure, as long as I arrive in Chicago by around 9 AM, as I will need to go to work. The earlier the arrival Monday morning, the better.`,
        `On November 3rd I need to fly from where I live, in Ithaca NY, to a conference in Reston VA. The conference starts the next day (November 4th) at 9am. I'd like to sleep well but if my travel plans are disrupted and I arrive late, it's ok. I'll either fly out of Ithaca, Syracuse, Elmira, or Binghamton.
I'll fly to DCA or IAD. For all my flights, I don't like having to get up before 7am to be on time to my flight. I'd like to avoid the amount of time I need to spend driving / taking Ubers / taking transit to airports both at home and at my destination.
I prefer flying out of my local airport in Ithaca rather rather than driving or taking an Uber to a nearby airport in Syracuse, Elmira, or Binghamton.
I want to avoid extra connections because they take more time and increase the chance of missing a connection. I can move pretty quickly through airports so connections longer than 45 min are fine but for connections that are tighter than that I worry about missing my flight if there is a delay. If the connection is earlier in the day and there are lots of other ways to get to my destination in the event of a missed connection, then a 30 min connection is fine.
I prefer to avoid long layovers. Under 60 minutes is fine, under 90 minutes is not a huge deal, and over 90 minutes starts to get annoying.
I feel that flying late in the day or connecting through airports with poor on-time performance like EWR increases my chance of a delay.
I don't like JFK because the food choices are poor (except for Shake Shack). When I fly to Europe from the US, I don't like taking a redeye but I know I'll usually have to take one. When I do take a redeye, I don't like to have a long layover early in the morning — I prefer to just arrive at my destination. If I do have a layover, I prefer to land later in the morning so that I can get some sleep on the plane.
I prefer to fly United because I'm a frequent flyer with them. When I fly for work, my travel is usually reimbursed from federal grants. Because of this, I must comply with the Fly America Act. This requires me to fly on a US carrier unless there are no other options. Even if I'm allowed to reimburse a trip on a non-US carrier, I don't want to because it creates extra paperwork.
For longer trips, I am happy to return to an airport that is different from the one I left from because I probably wouldn't drive my car in any case. When I do this, I'll take an Uber, rent a car, or get a ride. For shorter trips, however, I do prefer to return to the airport I left from so that I can drive to the airport, unless it saves me a lot of trouble.
I am not very price sensitive. It is ok to pay 20% more than the cheapest fare if the itinerary is more convenient. But if the fare is outrageous then that's problematic.
I usually don't check bags except on very long trips.`
    ];

    let idx = 0, charIdx = 0, typing = true, displayText = '';
    const speed = 42, pause = 5000, fadeTime = 500;
    const placeholder = document.getElementById('animPlaceholder');
    const animBox = document.getElementById('animBox');
    const toggleBtn = document.getElementById('toggleBtn');
    let autoPlay = true;
    let typingTimeout = null;
    let staticMode = false;

    // Toggle between animated and static mode
    function toggleTypewriter() {
        staticMode = !staticMode;
        autoPlay = false;
        if (typingTimeout) clearTimeout(typingTimeout);

        if (staticMode) {
            // Show full text with scroll (static mode)
            toggleBtn.classList.add('static-mode');
            animBox.classList.add('scrollable');
            toggleBtn.innerHTML = '▶';
            toggleBtn.title = 'Enable animation';
            displayText = prompts[idx];  // Full text, scrollable
            placeholder.textContent = displayText;
        } else {
            // Resume typing from current position (animated mode)
            toggleBtn.classList.remove('static-mode');
            animBox.classList.remove('scrollable');
            toggleBtn.innerHTML = '▐▐';
            toggleBtn.title = 'Pause animation';
            charIdx = 0;
            displayText = '';
            typing = true;
            type();
        }
    }

    // Carousel navigation functions
    function updateIndicators() {
        const dots = document.querySelectorAll('.indicator-dot');
        dots.forEach((dot, i) => {
            dot.classList.toggle('active', i === idx);
        });
    }

    function prevPrompt() {
        autoPlay = false;
        if (typingTimeout) clearTimeout(typingTimeout);
        placeholder.style.opacity = '0';
        setTimeout(() => {
            idx = (idx - 1 + prompts.length) % prompts.length;
            charIdx = 0;
            displayText = '';
            placeholder.style.opacity = '1';
            updateIndicators();
            animBox.scrollTop = 0;  // Reset scroll position
            if (staticMode) {
                displayText = prompts[idx];
                placeholder.textContent = displayText;
            } else {
                typing = true;
                type();
            }
        }, fadeTime);
    }

    function nextPrompt() {
        autoPlay = false;
        if (typingTimeout) clearTimeout(typingTimeout);
        placeholder.style.opacity = '0';
        setTimeout(() => {
            idx = (idx + 1) % prompts.length;
            charIdx = 0;
            displayText = '';
            placeholder.style.opacity = '1';
            updateIndicators();
            animBox.scrollTop = 0;  // Reset scroll position
            if (staticMode) {
                displayText = prompts[idx];
                placeholder.textContent = displayText;
            } else {
                typing = true;
                type();
            }
        }, fadeTime);
    }

    function goToPrompt(index) {
        if (index === idx) return;
        autoPlay = false;
        if (typingTimeout) clearTimeout(typingTimeout);
        placeholder.style.opacity = '0';
        setTimeout(() => {
            idx = index;
            charIdx = 0;
            displayText = '';
            placeholder.style.opacity = '1';
            updateIndicators();
            animBox.scrollTop = 0;  // Reset scroll position
            if (staticMode) {
                displayText = prompts[idx];
                placeholder.textContent = displayText;
            } else {
                typing = true;
                type();
            }
        }, fadeTime);
    }

    function trimToFit(text) {
        placeholder.textContent = text;

        // If content doesn't overflow, return as-is
        if (placeholder.scrollHeight <= animBox.clientHeight) {
            return text;
        }

        // Find sentence boundaries (period followed by space or newline)
        let trimmed = text;
        const sentencePattern = /\.\s+/g;
        let match;
        let lastSentenceEnd = 0;

        // Keep removing from the start until it fits
        while (placeholder.scrollHeight > animBox.clientHeight && (match = sentencePattern.exec(text))) {
            lastSentenceEnd = match.index + match[0].length;
            trimmed = text.substring(lastSentenceEnd);
            placeholder.textContent = trimmed;

            // Reset regex for next iteration
            if (placeholder.scrollHeight <= animBox.clientHeight) break;
        }

        return trimmed;
    }

    function type() {
        if (typing) {
            if (charIdx < prompts[idx].length) {
                displayText = prompts[idx].substring(0, charIdx + 1);
                displayText = trimToFit(displayText);
                charIdx++;
                typingTimeout = setTimeout(type, speed);
            } else {
                typing = false;
                if (autoPlay) {
                    typingTimeout = setTimeout(() => {
                        placeholder.style.opacity = '0';
                        setTimeout(() => {
                            idx = (idx + 1) % prompts.length;
                            charIdx = 0;
                            displayText = '';
                            typing = true;
                            placeholder.style.opacity = '1';
                            updateIndicators();
                            type();
                        }, fadeTime);
                    }, pause);
                }
            }
        }
    }

    setTimeout(type, 500);
</script>
"""
components.html(placeholder_html, height=178)

# Search mode - AI tab disabled for pilot study (code preserved, re-enable by restoring st.tabs line)
# tab_ai, tab_manual = st.tabs(["Describe Your Flight", "Search by Fields"])
(tab_manual,) = st.tabs(["Search by Fields"])

# Initialize button states (will be set inside tabs)
regular_search = False
manual_search_btn = False
manual_prompt = ""
manual_origins = []
manual_destinations = []
manual_days = []
ai_search = False

if False:  # AI search tab - disabled for pilot study. Re-enable by restoring st.tabs above.
    # with tab_ai:
    # Add negative margin to pull textarea up over the animation
    st.markdown('<div style="margin-top: -178px;">', unsafe_allow_html=True)

    # Add header for real prompt input
    st.markdown("**Your flight prompt:**")

    # Add CSS to hide label and make textarea transparent
    st.markdown("""
    <style>
        /* Hide the label */
        label[data-testid="stWidgetLabel"] {
            display: none !important;
        }
        /* Make textarea background transparent */
        textarea[aria-label="flight prompt input"] {
            background: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Real textarea that user types in
    prompt = st.text_area(
        "flight prompt input",
        height=150,
        placeholder="",
        label_visibility="collapsed",
        key="flight_prompt_input"
    )

    # Voice-to-text microphone button (compact icon style)
    voice_to_text_html = """
    <style>
        .voice-btn-container {
            display: flex;
            justify-content: flex-end;
            margin-top: -10px;
            margin-bottom: 10px;
            align-items: center;
            gap: 10px;
        }
        .voice-btn {
            background: #f0f2f6;
            border: 1px solid #ddd;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }
        .voice-btn:hover {
            background: #e0e2e6;
        }
        .voice-btn.recording {
            background: #ffebee;
            border-color: #ef5350;
        }
        .voice-status {
            font-size: 12px;
            color: #666;
            max-width: 200px;
        }
    </style>
    <div class="voice-btn-container">
        <button class="voice-btn" id="voiceBtn" onclick="toggleVoiceRecording()">🎙️</button>
        <span class="voice-status" id="voiceStatus"></span>
    </div>
    <script>
        let recognition = null;
        let isRecording = false;

        // Helper to properly set textarea value in React/Streamlit
        function setNativeValue(element, value) {
            const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            valueSetter.call(element, value);
            element.dispatchEvent(new Event('input', { bubbles: true }));
        }

        function toggleVoiceRecording() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                document.getElementById('voiceStatus').textContent = 'Speech recognition not supported in this browser';
                return;
            }

            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }

        function startRecording() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            const btn = document.getElementById('voiceBtn');
            const status = document.getElementById('voiceStatus');

            recognition.onstart = function() {
                isRecording = true;
                btn.classList.add('recording');
                btn.textContent = '⏹️';
                status.textContent = 'Listening...';
            };

            recognition.onresult = function(event) {
                // Find the Streamlit textarea in parent document
                const textarea = window.parent.document.querySelector('textarea[aria-label="flight prompt input"]');
                if (textarea) {
                    let finalTranscript = '';
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        if (event.results[i].isFinal) {
                            finalTranscript += event.results[i][0].transcript;
                        }
                    }
                    if (finalTranscript) {
                        // Append to existing text
                        const currentValue = textarea.value;
                        const newValue = currentValue ? currentValue + ' ' + finalTranscript : finalTranscript;
                        // Use native setter to properly update React state
                        setNativeValue(textarea, newValue);
                        status.textContent = 'Added: "' + finalTranscript.substring(0, 30) + (finalTranscript.length > 30 ? '..."' : '"');
                    }
                }
            };

            recognition.onerror = function(event) {
                status.textContent = 'Error: ' + event.error;
                stopRecording();
            };

            recognition.onend = function() {
                if (isRecording) {
                    // Restart if still supposed to be recording
                    recognition.start();
                }
            };

            recognition.start();
        }

        function stopRecording() {
            if (recognition) {
                isRecording = false;
                recognition.stop();
                recognition = null;
            }
            const btn = document.getElementById('voiceBtn');
            btn.classList.remove('recording');
            btn.textContent = '🎙️';
            document.getElementById('voiceStatus').textContent = '';
        }
    </script>
    """
    components.html(voice_to_text_html, height=50)

    # Add ID dynamically to textarea container for tutorial
    st.markdown("""
    <script>
    setTimeout(() => {
        const textarea = document.querySelector('textarea[aria-label="flight prompt input"]');
        if (textarea) {
            let container = textarea.closest('[data-testid="stTextArea"]');
            if (container) container.id = 'demo-prompt';
        }
    }, 200);
    </script>
    """, unsafe_allow_html=True)

    # Close the negative margin div
    st.markdown('</div>', unsafe_allow_html=True)

    # Search button for AI mode
    regular_search = st.button("🔍 Search Flights", type="primary", use_container_width=True, key="ai_search_btn")

with tab_manual:
    col_orig, col_dest = st.columns(2)
    with col_orig:
        manual_origins = st.multiselect(
            "Origin airport(s)",
            options=STATIC_ORIGINS,
            placeholder="Select origin(s)...",
            key="manual_origins_select"
        )
    with col_dest:
        manual_destinations = st.multiselect(
            "Destination airport(s)",
            options=STATIC_DESTINATIONS,
            placeholder="Select destination(s)...",
            key="manual_dests_select"
        )

    manual_days = st.multiselect(
        "Day(s) of week",
        options=DAYS_OF_WEEK,
        placeholder="Select day(s)...",
        key="manual_days_select"
    )

    manual_prompt = st.text_area(
        "Describe any additional preferences (optional)",
        placeholder="e.g. I prefer nonstop flights, cheapest option, morning departures...",
        height=80,
        key="manual_prompt_input"
    )

    manual_search_btn = st.button("🔍 Search Flights", type="primary", use_container_width=True, key="manual_search_btn")

# Close hideable-survey-content container
st.markdown('</div>', unsafe_allow_html=True)

# Check for auto-search request (from "Save & Search Again" button)
auto_search = st.session_state.get('auto_search_requested', False)
if auto_search:
    # Use the saved prompt for auto-search
    prompt = st.session_state.get('auto_search_prompt', prompt)
    # Clear the auto-search flag
    st.session_state.auto_search_requested = False
    st.session_state.auto_search_prompt = None

# Determine search mode
if manual_search_btn:
    st.session_state.search_mode = "manual"
elif regular_search or auto_search:
    st.session_state.search_mode = "ai"

if regular_search or manual_search_btn or auto_search:
    # Reset session state to clear previous results
    st.session_state.all_flights = []
    st.session_state.selected_flights = []
    st.session_state.all_return_flights = []
    st.session_state.selected_return_flights = []
    st.session_state.has_return = False
    st.session_state.outbound_submitted = False
    st.session_state.return_submitted = False
    st.session_state.csv_data_outbound = None
    st.session_state.csv_data_return = None
    st.session_state.review_confirmed = False

    # Validation
    validation_errors = []

    if st.session_state.search_mode == "manual":
        if not manual_origins:
            validation_errors.append("Please select at least one origin airport")
        if not manual_destinations:
            validation_errors.append("Please select at least one destination airport")
        if not manual_days:
            validation_errors.append("Please select at least one day of the week")
    else:
        if not prompt or not prompt.strip():
            validation_errors.append("Please describe your flight needs")

    if validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        # Clear previous selections and submissions when starting new search
        st.session_state.selected_flights = []
        st.session_state.all_flights = []
        st.session_state.selected_return_flights = []
        st.session_state.all_return_flights = []
        st.session_state.has_return = False
        st.session_state.csv_generated = False
        st.session_state.outbound_submitted = False
        st.session_state.return_submitted = False
        st.session_state.csv_data_outbound = None
        st.session_state.csv_data_return = None
        st.session_state.review_confirmed = False

        with st.spinner("✨ Searching flights..."):
            try:
                if st.session_state.search_mode == "manual":
                    # Filter from pre-fetched static flight database (no API call)
                    all_flights = [
                        f for f in STATIC_FLIGHTS
                        if f["origin"] in manual_origins
                        and f["destination"] in manual_destinations
                        and f["day_of_week"] in manual_days
                    ]
                    all_return_flights = []
                    has_return = False
                    st.session_state.has_return = False

                    extra = f" | Preferences: {manual_prompt}" if manual_prompt and manual_prompt.strip() else ""
                    days_str = ", ".join(manual_days)
                    st.session_state.original_prompt = (
                        f"Manual search: {', '.join(manual_origins)} to "
                        f"{', '.join(manual_destinations)} on {days_str}{extra}"
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**From:** {' or '.join(manual_origins)}")
                    with col2:
                        st.info(f"**To:** {' or '.join(manual_destinations)}")
                    with col3:
                        st.info(f"**Days:** {days_str}")

                else:
                    # AI mode: parse with LLM then search Amadeus API
                    st.session_state.original_prompt = prompt

                    st.info("🤖 Parsing your request with Gemini...")
                    parsed = parse_flight_prompt_with_llm(prompt)
                    st.session_state.parsed_params = parsed

                    # Debug: show parsed results
                    with st.expander("🔍 Debug: Parsed Parameters"):
                        st.json(parsed)

                    if not parsed.get('origins') or not parsed.get('destinations'):
                        st.error("Could not extract origin and destination. Please specify airports or cities.")
                        st.stop()

                    st.success("✅ Understood your request!")

                    # Check if return flight is present
                    return_dates = parsed.get('return_dates', [])
                    if not return_dates and parsed.get('return_date'):
                        return_dates = [parsed.get('return_date')]
                    has_return = len(return_dates) > 0
                    st.session_state.has_return = has_return

                    if has_return:
                        col1, col2, col3, col4 = st.columns(4)
                    else:
                        col1, col2, col3 = st.columns(3)

                    with col1:
                        origins_str = " or ".join(parsed['origins'])
                        st.info(f"**From:** {origins_str}")
                    with col2:
                        dests_str = " or ".join(parsed['destinations'])
                        st.info(f"**To:** {dests_str}")
                    with col3:
                        departure_dates = parsed.get('departure_dates', [])
                        dates_str = ", ".join(departure_dates) if departure_dates else 'Not specified'
                        st.info(f"**Depart:** {dates_str}")

                    if has_return:
                        with col4:
                            return_dates_str = ", ".join(return_dates)
                            st.info(f"**Return:** {return_dates_str}")

                    # Search flights from all origin/destination combinations
                    all_flights = []
                    all_return_flights = []

                    provider_name = "Amadeus API" if flight_client.provider == "amadeus" else "SerpAPI (Google Flights)"
                    st.info(f"✈️ Searching outbound flights from {provider_name}...")

                    departure_dates = parsed.get('departure_dates', [])
                    if not departure_dates:
                        st.error("No departure dates found. Please specify when you want to fly.")
                        st.stop()

                    for origin_code in parsed['origins']:
                        for dest_code in parsed['destinations']:
                            origin, origin_warning = get_test_api_fallback(origin_code)
                            dest, dest_warning = get_test_api_fallback(dest_code)

                            if origin_warning:
                                st.warning(origin_warning)
                            if dest_warning:
                                st.warning(dest_warning)

                            for departure_date in departure_dates:
                                st.info(f"Searching: {origin} → {dest} on {departure_date}")

                                results = flight_client.search_flights(
                                    origin=origin,
                                    destination=dest,
                                    departure_date=departure_date,
                                    adults=1,
                                    max_results=250
                                )

                                with st.expander(f"🔍 Debug: {provider_name} Response ({origin}→{dest} on {departure_date})"):
                                    st.write(f"Type: {type(results)}")
                                    if isinstance(results, dict):
                                        st.write(f"Keys: {results.keys()}")
                                        if 'data' in results:
                                            st.write(f"Number of flights: {len(results['data'])}")
                                    elif isinstance(results, list):
                                        st.write(f"Number of flights: {len(results)}")
                                    st.json(results if isinstance(results, (dict, list)) else str(results))

                                if isinstance(results, list):
                                    flight_offers = results
                                elif isinstance(results, dict) and 'data' in results:
                                    flight_offers = results['data']
                                else:
                                    flight_offers = []

                                for offer in flight_offers:
                                    flight_info = flight_client.parse_flight_offer(offer)
                                    if flight_info:
                                        all_flights.append(flight_info)

                            if has_return:
                                return_dates = parsed.get('return_dates', [])
                                if not return_dates and parsed.get('return_date'):
                                    return_dates = [parsed.get('return_date')]

                                for return_date in return_dates:
                                    st.info(f"✈️ Searching return flights: {dest} → {origin} on {return_date}")

                                    return_results = flight_client.search_flights(
                                        origin=dest,
                                        destination=origin,
                                        departure_date=return_date,
                                        adults=1,
                                        max_results=250
                                    )

                                    with st.expander(f"🔍 Debug: Return Flight Response ({dest}→{origin} on {return_date})"):
                                        st.write(f"Type: {type(return_results)}")
                                        if isinstance(return_results, dict):
                                            st.write(f"Keys: {return_results.keys()}")
                                            if 'data' in return_results:
                                                st.write(f"Number of flights: {len(return_results['data'])}")
                                        elif isinstance(return_results, list):
                                            st.write(f"Number of flights: {len(return_results)}")
                                        st.json(return_results if isinstance(return_results, (dict, list)) else str(return_results))

                                    if isinstance(return_results, list):
                                        return_flight_offers = return_results
                                    elif isinstance(return_results, dict) and 'data' in return_results:
                                        return_flight_offers = return_results['data']
                                    else:
                                        return_flight_offers = []

                                    for offer in return_flight_offers:
                                        flight_info = flight_client.parse_flight_offer(offer)
                                        if flight_info:
                                            all_return_flights.append(flight_info)

                if not all_flights:
                    st.error("No outbound flights found. Try different dates or airports.")
                    st.stop()

                # Look up airline names using Amadeus API
                # Use .get() for compatibility with both APIs (airline or carrier_code)
                all_airline_codes = [f.get('airline') or f.get('carrier_code') for f in all_flights if f.get('airline') or f.get('carrier_code')]
                if has_return and all_return_flights:
                    all_airline_codes.extend([f.get('airline') or f.get('carrier_code') for f in all_return_flights if f.get('airline') or f.get('carrier_code')])

                unique_airlines = list(set(all_airline_codes))
                airline_name_map = flight_client.get_airline_names(unique_airlines)
                st.session_state.airline_names = airline_name_map

                # Store all flights
                st.session_state.all_flights = all_flights
                st.session_state.all_return_flights = all_return_flights

                # Session persistence: Save progress after search
                if st.session_state.get('token'):
                    from backend.db import save_session_progress
                    save_session_progress(st.session_state.token, {
                        'session_id': st.session_state.session_id,
                        'current_phase': 'flight_selection',
                        'search_completed': 1,
                        'user_prompt': st.session_state.get('user_prompt', ''),
                        'all_flights_json': all_flights,
                    })
                    print(f"[PILOT] Saved search progress for {st.session_state.token}")

                if has_return:
                    st.success(f"✅ Found {len(all_flights)} outbound flights and {len(all_return_flights)} return flights!")
                else:
                    st.success(f"✅ Found {len(all_flights)} flights!")
                # Don't rerun here - it causes infinite loop

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Display results - NEW SIMPLIFIED VERSION (no algorithm ranking)
if st.session_state.all_flights:
    # Hide content between "How to Use" and "Search Flights" on results page
    st.markdown("""
    <style>
        .hideable-survey-content {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Calculate submission progress
    has_return = st.session_state.has_return and st.session_state.all_return_flights
    num_required = 2 if has_return else 1
    num_completed = (1 if st.session_state.outbound_submitted else 0) + (1 if st.session_state.return_submitted else 0)
    all_submitted = (num_completed == num_required)

    # Check if all submissions are complete
    if all_submitted:
        # COMPLETION SCREEN
        print(f"[DEBUG] Completion screen - search_id: {st.session_state.get('search_id')}")
        print(f"[DEBUG] Completion screen - db_save_error: {st.session_state.get('db_save_error')}")
        print(f"[DEBUG] Completion screen - csv_generated: {st.session_state.get('csv_generated')}")

        # FAILSAFE: If submitted but no search_id and no error, check if we need review or can save
        # Check for either csv_generated (single panel) OR outbound_submitted+return_submitted (dual panel)
        ready_to_save = (st.session_state.get('csv_generated') or
                        (st.session_state.get('outbound_submitted') and st.session_state.get('return_submitted')))

        # REVIEW SECTION: Show before saving to database
        if ready_to_save and not st.session_state.get('search_id') and not st.session_state.get('db_save_error') and not st.session_state.get('review_confirmed'):
            st.markdown("---")
            st.markdown("## 📋 Review Your Results")
            st.markdown("Before finalizing your submission, please review your prompt and ensure it accurately captures your preferences.")

            # Show current prompt
            st.markdown("### Your Current Prompt:")
            st.info(st.session_state.get('original_prompt', ''))

            # Questions for reflection
            with st.expander("❓ Review Questions", expanded=True):
                st.markdown("""
                Please consider the following:
                - Does your prompt accurately describe your persona and preferences?
                - Did you include all the preferences you used when selecting flights?
                - Are your trade-offs and priorities clearly stated?
                - Is there anything you forgot to mention that influenced your rankings?

                **Example:** If you prioritized cheap flights but didn't mention it in your prompt, you should add it!
                """)

            # Option to edit prompt
            st.markdown("### Edit Your Prompt (Optional)")
            edited_prompt = st.text_area(
                "If you'd like to revise your prompt, edit it here:",
                value=st.session_state.get('original_prompt', ''),
                height=150,
                key="edited_prompt"
            )

            # Single button - automatically saves any edits
            if st.button("✅ Confirm & Submit Final Results", type="primary", use_container_width=True):
                # Update prompt if edited
                if edited_prompt != st.session_state.get('original_prompt', ''):
                    st.session_state.original_prompt = edited_prompt
                st.session_state.review_confirmed = True

                # Session persistence: Save progress after flight selection confirmation
                if st.session_state.get('token'):
                    from backend.db import save_session_progress
                    save_session_progress(st.session_state.token, {
                        'session_id': st.session_state.session_id,
                        'current_phase': 'cross_validation',
                        'flight_selection_confirmed': 1,
                        'selected_flights_json': st.session_state.selected_flights,
                    })
                    print(f"[PILOT] Saved flight selection for {st.session_state.token}")

                st.rerun()

            st.stop()  # Stop here until user confirms

        # Now save to database ONLY after review is confirmed
        if ready_to_save and not st.session_state.get('search_id') and not st.session_state.get('db_save_error') and st.session_state.get('review_confirmed'):
            st.info("⚙️ Attempting to save to database...")
            print(f"[DEBUG] FAILSAFE: Attempting database save in completion screen")
            try:
                from backend.db import save_search_and_csv
                import traceback

                csv_data = st.session_state.get('csv_data_outbound')

                # Show what we have
                debug_info = f"""
                csv_data exists: {bool(csv_data)}
                all_flights: {len(st.session_state.all_flights) if st.session_state.all_flights else 0}
                selected_flights: {len(st.session_state.selected_flights) if st.session_state.selected_flights else 0}
                original_prompt: {bool(st.session_state.get('original_prompt'))}
                parsed_params: {bool(st.session_state.parsed_params)}
                """
                print(debug_info)

                if not csv_data:
                    st.session_state.db_save_error = "No CSV data available"
                elif not st.session_state.all_flights:
                    st.session_state.db_save_error = "No flight data available"
                elif not st.session_state.selected_flights:
                    st.session_state.db_save_error = "No selected flights available"
                else:
                    # For DATA token, delete all previous submissions to allow overwriting
                    if st.session_state.token.upper() == "DATA":
                        from backend.db import SessionLocal, Search, FlightCSV, UserRanking, FlightShown, CrossValidation, SurveyResponse
                        db = SessionLocal()
                        try:
                            # Find all previous searches for this token
                            previous_searches = db.query(Search).filter(
                                (Search.session_id == st.session_state.token) | (Search.completion_token == st.session_state.token)
                            ).all()

                            for search in previous_searches:
                                # Delete cross-validation data
                                db.query(CrossValidation).filter_by(reviewer_session_id=search.session_id).delete()
                                db.query(CrossValidation).filter_by(reviewed_session_id=search.session_id).delete()

                                # Delete survey data
                                db.query(SurveyResponse).filter_by(session_id=search.session_id).delete()

                                # Delete rankings and flights
                                rankings = db.query(UserRanking).filter_by(search_id=search.search_id).all()
                                for ranking in rankings:
                                    db.query(FlightShown).filter_by(id=ranking.flight_id).delete()
                                    db.delete(ranking)

                                # Delete CSV records
                                db.query(FlightCSV).filter_by(search_id=search.search_id).delete()

                                # Delete the search itself
                                db.delete(search)

                            db.commit()
                            print(f"[DEBUG] Deleted all previous DATA token submissions")
                        except Exception as e:
                            db.rollback()
                            print(f"[DEBUG] Error deleting previous DATA submissions: {str(e)}")
                        finally:
                            db.close()

                    # Save outbound data
                    search_id = save_search_and_csv(
                        session_id=st.session_state.session_id,
                        user_prompt=st.session_state.get('original_prompt', ''),
                        parsed_params=st.session_state.parsed_params or {},
                        all_flights=st.session_state.all_flights,
                        selected_flights=st.session_state.selected_flights,
                        csv_data=csv_data,
                        token=st.session_state.token
                    )

                    # Note: Token will be marked as used AFTER the 15-second countdown
                    # (see countdown logic below where search_id is displayed)

                    # Save flight results for cross-validation
                    try:
                        from backend.db import update_search_flight_results
                        if st.session_state.all_flights:
                            # Save ranked flights (the order shown to user)
                            update_search_flight_results(
                                session_id=st.session_state.session_id,
                                listen_ranked_flights=st.session_state.all_flights
                            )
                            print(f"[DEBUG] Saved flight results for cross-validation")
                    except ImportError:
                        # Function not available yet (old deployment) - cross-validation will be skipped
                        print(f"[DEBUG] update_search_flight_results not available - skipping")

                    # Also save return flight CSV if it exists
                    if st.session_state.get('csv_data_return'):
                        from backend.db import SessionLocal, FlightCSV
                        db = SessionLocal()
                        try:
                            csv_record = FlightCSV(
                                session_id=st.session_state.session_id,
                                search_id=search_id,
                                csv_data=st.session_state.csv_data_return,
                                num_flights=len(st.session_state.all_return_flights) if st.session_state.all_return_flights else 0,
                                num_selected=len(st.session_state.selected_return_flights) if st.session_state.selected_return_flights else 0
                            )
                            db.add(csv_record)
                            db.commit()
                            print(f"[DEBUG] FAILSAFE: Also saved return flight CSV to search {search_id}")
                        finally:
                            db.close()

                    st.session_state.search_id = search_id
                    st.session_state.csv_generated = True
                    st.session_state.countdown_started = True  # Start countdown phase
                    print(f"[DEBUG] FAILSAFE: Successfully saved! Search ID: {search_id}")
                    st.rerun()  # Rerun to show the success message and start countdown
            except Exception as e:
                print(f"[DEBUG] FAILSAFE: Save failed - {str(e)}")
                import traceback
                print(traceback.format_exc())
                st.session_state.db_save_error = str(e)

        if st.session_state.get('search_id'):
            st.success("✅ All rankings submitted successfully!")

            # Download Rankings CSV
            token = st.session_state.get('token')
            if token:
                try:
                    from export_session_data import export_simple_rankings_csv
                    simple_csv_file = export_simple_rankings_csv(token, output_file=None)
                    if simple_csv_file:
                        with open(simple_csv_file, 'r', encoding='utf-8') as f:
                            simple_csv_data = f.read()
                        st.download_button(
                            label="📊 Download Your Rankings (CSV)",
                            data=simple_csv_data,
                            file_name=f"rankings_{token}.csv",
                            mime="text/csv",
                            help="Contains: prompt, flights, and your rankings"
                        )
                        import os
                        if os.path.exists(simple_csv_file):
                            os.remove(simple_csv_file)
                except Exception as e:
                    pass  # Silently fail - not critical

            st.markdown("---")

            # Cross-validation section (before survey)
            # PILOT STUDY: Group A tokens skip cross-validation entirely
            # Groups B/C do 4 sequential re-rankings of assigned prompts
            token_group = st.session_state.get('token_group')
            rerank_targets = st.session_state.get('rerank_targets', [])

            # Skip cross-validation for Group A (first group has nothing to re-rank)
            if token_group == 'A' and not st.session_state.get('cross_validation_completed'):
                st.session_state.cross_validation_completed = True
                st.session_state.all_reranks_completed = True
                print(f"[PILOT] Group A token - skipping cross-validation")
                # Mark token as used immediately for Group A (they complete here)
                if not st.session_state.get('token_marked_used'):
                    from backend.db import mark_token_used
                    token = st.session_state.get('token')
                    if token and token.upper() not in ["DEMO", "DATA"]:
                        mark_token_used(token)
                        st.session_state.token_marked_used = True
                        print(f"[PILOT] Token {token} marked as used (Group A complete)")
                # Trigger backup on session completion
                if not st.session_state.get('backup_triggered'):
                    st.session_state.backup_triggered = True
                    trigger_backup_on_completion(st.session_state.get('token', 'unknown'))

            if not st.session_state.get('cross_validation_completed'):
                try:
                    from backend.db import get_previous_search_for_validation, get_assigned_search_for_validation

                    # Hide all content above cross-validation section
                    st.markdown("""
                    <style>
                        /* Hide all content before cross-validation */
                        .main > div:has(+ div .cross-validation-section) {
                            display: none !important;
                        }
                        /* Dim header and main content but NOT sidebar */
                        .stApp > header,
                        .main > div:not(:has(.cross-validation-section)) {
                            opacity: 0.1;
                            pointer-events: none;
                        }
                        /* Keep sidebar fully functional */
                        [data-testid="stSidebar"] {
                            opacity: 1 !important;
                            pointer-events: auto !important;
                        }
                        .cross-validation-section {
                            background: white;
                            position: relative;
                            z-index: 1000;
                        }
                    </style>
                    """, unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown('<div class="cross-validation-section">', unsafe_allow_html=True)

                    # PILOT STUDY: Sequential re-rankings for Groups B/C
                    if rerank_targets:
                        # Get current progress
                        completed_reranks = st.session_state.get('completed_reranks', [])
                        remaining_targets = [t for t in rerank_targets if t not in completed_reranks]
                        current_rerank_num = len(completed_reranks) + 1
                        total_reranks = len(rerank_targets)

                        if not remaining_targets:
                            # All re-rankings complete
                            st.session_state.all_reranks_completed = True
                            st.session_state.cross_validation_completed = True
                            # Trigger backup on session completion
                            if not st.session_state.get('backup_triggered'):
                                st.session_state.backup_triggered = True
                                trigger_backup_on_completion(st.session_state.get('token', 'unknown'))
                            st.rerun()

                        # Progress indicator
                        st.progress(len(completed_reranks) / total_reranks)
                        st.markdown(f"### 🤝 Re-ranking {current_rerank_num} of {total_reranks}")
                        st.markdown("*Help us by ranking flights for another user's search*")

                        # Get the current target's search data
                        current_target = remaining_targets[0]
                        if 'cross_val_data' not in st.session_state or st.session_state.get('current_cv_target') != current_target:
                            st.session_state.cross_val_data = get_assigned_search_for_validation(
                                st.session_state.get('token'),
                                current_target
                            )
                            st.session_state.current_cv_target = current_target
                            # Reset selection for new target
                            st.session_state.cross_val_selected_flights = []
                            st.session_state.cv_checkbox_version = st.session_state.get('cv_checkbox_version', 0) + 1
                    else:
                        # Legacy behavior: Non-pilot tokens use regular cross-validation
                        st.markdown("### 🤝 Help Validate Another Search")
                        st.markdown("*Before completing your session, please help us by ranking flights for another user's search*")

                        # Fetch previous search for cross-validation (separate queues by token type)
                        if 'cross_val_data' not in st.session_state:
                            st.session_state.cross_val_data = get_previous_search_for_validation(
                                st.session_state.session_id,
                                st.session_state.get('token')
                            )
                except ImportError:
                    # Function not available yet (old deployment) - skip cross-validation
                    st.session_state.cross_validation_completed = True
                    st.session_state.cross_val_data = None
                    if not st.session_state.get('backup_triggered'):
                        st.session_state.backup_triggered = True
                        trigger_backup_on_completion(st.session_state.get('token', 'unknown'))

                if st.session_state.get('cross_val_data') is not None:
                    if not st.session_state.cross_val_data:
                        # No previous search available - skip cross-validation
                        st.info("ℹ️ No previous searches available for validation. You're one of the first users!")
                        st.session_state.cross_validation_completed = True
                        if not st.session_state.get('backup_triggered'):
                            st.session_state.backup_triggered = True
                            trigger_backup_on_completion(st.session_state.get('token', 'unknown'))
                        st.rerun()
                    else:
                        cross_val = st.session_state.cross_val_data

                        # Display the user's original prompt
                        st.info(f"**Another user's search prompt:**\n\n> {cross_val['prompt']}")
                        st.markdown("**Task:** Review their flights and select your top 5 that best match their needs, then drag to rank them.")
                        st.markdown("---")

                        # Set up cross-validation session state (mirror main interface)
                        cv_flights = cross_val['flights']

                        if 'cross_val_selected_flights' not in st.session_state:
                            st.session_state.cross_val_selected_flights = []
                        if 'cv_filter_reset_counter' not in st.session_state:
                            st.session_state.cv_filter_reset_counter = 0
                        if 'cv_sort_version' not in st.session_state:
                            st.session_state.cv_sort_version = 0
                        if 'cv_checkbox_version' not in st.session_state:
                            st.session_state.cv_checkbox_version = 0
                        if 'cv_sort_price_dir' not in st.session_state:
                            st.session_state.cv_sort_price_dir = 'asc'
                        if 'cv_sort_duration_dir' not in st.session_state:
                            st.session_state.cv_sort_duration_dir = 'asc'

                        # Display flight count
                        st.markdown(f"### ✈️ Found {len(cv_flights)} Flights")
                        st.markdown("**Select your top 5 flights and drag to rank them →**")

                        # SIDEBAR FILTERS (EXACT copy from main interface)
                        with st.sidebar:
                            st.markdown("---")
                            st.markdown('<h2><span class="filter-heading-neon">🔍 Filters</span></h2>', unsafe_allow_html=True)

                            # Get unique values
                            unique_airlines_cv = sorted(set([f['airline'] for f in cv_flights]))
                            airline_names_map_cv = {code: get_airline_name(code) for code in unique_airlines_cv}
                            unique_connections_cv = sorted(set([f['stops'] for f in cv_flights]))
                            prices_cv = [f['price'] for f in cv_flights]
                            min_price_cv, max_price_cv = (min(prices_cv), max(prices_cv)) if prices_cv else (0, 1000)
                            durations_cv = [f['duration_min'] for f in cv_flights]
                            min_duration_cv, max_duration_cv = (min(durations_cv), max(durations_cv)) if durations_cv else (0, 1440)

                            # Filters (exact copy)
                            with st.expander("✈️ Airlines", expanded=False):
                                selected_airlines_cv = []
                                for airline_code in unique_airlines_cv:
                                    if st.checkbox(airline_names_map_cv[airline_code], key=f"cv_airline_{airline_code}_{st.session_state.cv_filter_reset_counter}"):
                                        selected_airlines_cv.append(airline_code)

                            with st.expander("🔄 Connections", expanded=False):
                                selected_connections_cv = []
                                for conn_count in unique_connections_cv:
                                    conn_label = "Direct" if conn_count == 0 else f"{conn_count} stop{'s' if conn_count > 1 else ''}"
                                    if st.checkbox(conn_label, key=f"cv_conn_{conn_count}_{st.session_state.cv_filter_reset_counter}"):
                                        selected_connections_cv.append(conn_count)

                            with st.expander("💰 Price Range", expanded=False):
                                price_range_cv = st.slider(
                                    "Select price range",
                                    min_value=float(min_price_cv),
                                    max_value=float(max_price_cv),
                                    value=(float(min_price_cv), float(max_price_cv)),
                                    step=10.0,
                                    format="$%.0f",
                                    key=f"cv_price_{st.session_state.cv_filter_reset_counter}"
                                )

                            with st.expander("⏱️ Flight Duration", expanded=False):
                                duration_range_cv = st.slider(
                                    "Select duration range",
                                    min_value=int(min_duration_cv),
                                    max_value=int(max_duration_cv),
                                    value=(int(min_duration_cv), int(max_duration_cv)),
                                    step=30,
                                    format="%d min",
                                    key=f"cv_duration_{st.session_state.cv_filter_reset_counter}"
                                )
                                min_h, min_m = divmod(duration_range_cv[0], 60)
                                max_h, max_m = divmod(duration_range_cv[1], 60)
                                st.caption(f"{min_h}h {min_m}m - {max_h}h {max_m}m")

                            with st.expander("🛫 Departure Time", expanded=False):
                                dept_range_cv = st.slider(
                                    "Select departure time range",
                                    min_value=0.0,
                                    max_value=24.0,
                                    value=(0.0, 24.0),
                                    step=0.5,
                                    format="%.1f",
                                    key=f"cv_dept_{st.session_state.cv_filter_reset_counter}"
                                )
                                def hours_to_time(h):
                                    hours = int(h)
                                    mins = int((h - hours) * 60)
                                    return f"{hours:02d}:{mins:02d}"
                                st.caption(f"{hours_to_time(dept_range_cv[0])} - {hours_to_time(dept_range_cv[1])}")

                            with st.expander("🛬 Arrival Time", expanded=False):
                                arr_range_cv = st.slider(
                                    "Select arrival time range",
                                    min_value=0.0,
                                    max_value=24.0,
                                    value=(0.0, 24.0),
                                    step=0.5,
                                    format="%.1f",
                                    key=f"cv_arr_{st.session_state.cv_filter_reset_counter}"
                                )
                                st.caption(f"{hours_to_time(arr_range_cv[0])} - {hours_to_time(arr_range_cv[1])}")

                            if st.button("Clear All Filters", use_container_width=True, key="clear_cv"):
                                st.session_state.cv_filter_reset_counter += 1
                                st.rerun()

                        # Apply filters (exact copy from apply_filters)
                        filtered_cv = apply_filters(
                            cv_flights,
                            airlines=selected_airlines_cv if selected_airlines_cv else None,
                            connections=selected_connections_cv if selected_connections_cv else None,
                            price_range=price_range_cv if price_range_cv != (float(min_price_cv), float(max_price_cv)) else None,
                            duration_range=duration_range_cv if duration_range_cv != (int(min_duration_cv), int(max_duration_cv)) else None,
                            departure_range=dept_range_cv if dept_range_cv != (0.0, 24.0) else None,
                            arrival_range=arr_range_cv if arr_range_cv != (0.0, 24.0) else None,
                            origins=None,
                            destinations=None
                        )

                        # Two-column layout (EXACT copy)
                        col_flights_cv, col_ranking_cv = st.columns([2, 1])

                        with col_flights_cv:
                            st.markdown("#### All Flights")

                            if len(filtered_cv) < len(cv_flights):
                                st.info(f"🔍 Filters applied: Showing {len(filtered_cv)} of {len(cv_flights)} flights")

                            # Sort buttons - fixed to sort filtered results
                            col_sort1, col_sort2 = st.columns(2)
                            with col_sort1:
                                arrow = "↑" if st.session_state.cv_sort_price_dir == 'asc' else "↓"
                                if st.button(f"💰 Sort by Price {arrow}", key="cv_sort_price", use_container_width=True):
                                    reverse = st.session_state.cv_sort_price_dir == 'desc'
                                    filtered_cv[:] = sorted(filtered_cv, key=lambda x: x['price'], reverse=reverse)
                                    st.session_state.cv_sort_price_dir = 'desc' if st.session_state.cv_sort_price_dir == 'asc' else 'asc'
                                    st.rerun()
                            with col_sort2:
                                arrow = "↑" if st.session_state.cv_sort_duration_dir == 'asc' else "↓"
                                if st.button(f"⏱️ Sort by Duration {arrow}", key="cv_sort_dur", use_container_width=True):
                                    reverse = st.session_state.cv_sort_duration_dir == 'desc'
                                    filtered_cv[:] = sorted(filtered_cv, key=lambda x: x['duration_min'], reverse=reverse)
                                    st.session_state.cv_sort_duration_dir = 'desc' if st.session_state.cv_sort_duration_dir == 'asc' else 'asc'
                                    st.rerun()

                            # Display flights (EXACT HTML copy)
                            for idx, flight in enumerate(filtered_cv):
                                flight_unique_key = f"{flight['id']}_{flight['departure_time']}"
                                is_selected = any(f"{f['id']}_{f['departure_time']}" == flight_unique_key for f in st.session_state.cross_val_selected_flights)

                                col1, col2 = st.columns([1, 5])

                                with col1:
                                    # Use checkbox to match original flight selection interface
                                    checkbox_key = f"cv_chk_{idx}_v{st.session_state.cv_checkbox_version}"
                                    selected = st.checkbox(
                                        "Select flight",
                                        value=is_selected,
                                        key=checkbox_key,
                                        label_visibility="collapsed",
                                        disabled=(not is_selected and len(st.session_state.cross_val_selected_flights) >= 5)
                                    )

                                    if selected and not is_selected:
                                        # Add to selected flights
                                        if len(st.session_state.cross_val_selected_flights) < 5:
                                            st.session_state.cross_val_selected_flights.append(flight)
                                    elif not selected and is_selected:
                                        # Deselect - remove from selected flights
                                        st.session_state.cross_val_selected_flights = [
                                            f for f in st.session_state.cross_val_selected_flights
                                            if f"{f['id']}_{f['departure_time']}" != flight_unique_key
                                        ]

                                with col2:
                                    dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                                    arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                                    dept_time_display = dept_dt.strftime("%I:%M %p")
                                    arr_time_display = arr_dt.strftime("%I:%M %p")
                                    dept_date_display = dept_dt.strftime("%a, %b %d")

                                    duration_hours = flight['duration_min'] // 60
                                    duration_mins = flight['duration_min'] % 60
                                    duration_display = f"{duration_hours} hr {duration_mins} min" if duration_hours > 0 else f"{duration_mins} min"

                                    airline_name = get_airline_name(flight['airline'])
                                    stops_text = "Direct" if flight['stops'] == 0 else f"{flight['stops']} stop{'s' if flight['stops'] > 1 else ''}"

                                    st.markdown(f"""
                                    <div style="line-height: 1.4; margin: 0; padding: 0.4rem 0; border-bottom: 1px solid #eee;">
                                    <div style="font-size: 1.1em; margin-bottom: 0.2rem;">
                                        <span style="font-weight: 700;">{format_price(flight['price'])}</span> •
                                        <span style="font-weight: 600;">{duration_display}</span> •
                                        <span style="font-weight: 500;">{stops_text}</span> •
                                        <span style="font-weight: 500;">{dept_time_display} - {arr_time_display}</span>
                                    </div>
                                    <div style="font-size: 0.9em; color: #666;">
                                        <span>{airline_name} {flight['flight_number']}</span> |
                                        <span>{flight['origin']} → {flight['destination']}</span> |
                                        <span>{dept_date_display}</span>
                                    </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                        with col_ranking_cv:
                            st.markdown("#### 📋 Top 5 (Drag to Rank)")
                            st.markdown(f"**{len(st.session_state.cross_val_selected_flights)}/5 selected**")

                            if st.session_state.cross_val_selected_flights:
                                # Enforce limit
                                if len(st.session_state.cross_val_selected_flights) > 5:
                                    st.session_state.cross_val_selected_flights = st.session_state.cross_val_selected_flights[:5]
                                    st.rerun()

                                # Create draggable items (exact copy)
                                draggable_items = []
                                for i, flight in enumerate(st.session_state.cross_val_selected_flights):
                                    airline_name = get_airline_name(flight['airline'])
                                    dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                                    arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                                    dept_time_display = dept_dt.strftime("%I:%M %p")
                                    arr_time_display = arr_dt.strftime("%I:%M %p")
                                    dept_date_display = dept_dt.strftime("%a, %b %d")

                                    duration_hours = flight['duration_min'] // 60
                                    duration_mins = flight['duration_min'] % 60
                                    duration_display = f"{duration_hours}h {duration_mins}m" if duration_hours > 0 else f"{duration_mins}m"

                                    stops = int(flight.get('stops', 0))
                                    stops_text = "Nonstop" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"

                                    item = f"""#{i+1}: {format_price(flight['price'])} • {duration_display} • {stops_text}
{dept_time_display} - {arr_time_display}
{airline_name} {flight['flight_number']}
{flight['origin']} → {flight['destination']} | {dept_date_display}"""
                                    draggable_items.append(item)

                                # Sortable list
                                sorted_items = sort_items(
                                    draggable_items,
                                    multi_containers=False,
                                    direction='vertical',
                                    key=f'cv_sort_v{st.session_state.cv_sort_version}_n{len(st.session_state.cross_val_selected_flights)}'
                                )

                                # Update order if dragged
                                if sorted_items != draggable_items and len(sorted_items) == len(draggable_items):
                                    new_order = []
                                    for sorted_item in sorted_items:
                                        rank = int(sorted_item.split(':')[0].replace('#', '')) - 1
                                        if rank < len(st.session_state.cross_val_selected_flights):
                                            new_order.append(st.session_state.cross_val_selected_flights[rank])
                                    if len(new_order) == len(st.session_state.cross_val_selected_flights):
                                        st.session_state.cross_val_selected_flights = new_order
                                        st.session_state.cv_sort_version += 1
                                        st.rerun()

                                # X buttons
                                st.markdown("---")
                                cols = st.columns(5)
                                for i, flight in enumerate(st.session_state.cross_val_selected_flights):
                                    with cols[i]:
                                        if st.button("✖", key=f"cv_remove_{i}_{flight['id']}", help=f"Remove #{i+1}"):
                                            flight_unique_key = f"{flight['id']}_{flight['departure_time']}"
                                            st.session_state.cross_val_selected_flights = [
                                                f for f in st.session_state.cross_val_selected_flights
                                                if f"{f['id']}_{f['departure_time']}" != flight_unique_key
                                            ]
                                            st.session_state.cv_checkbox_version += 1
                                            st.rerun()

                                # Submit button
                                if len(st.session_state.cross_val_selected_flights) == 5:
                                    # PILOT STUDY: Different button label for sequential re-rankings
                                    remaining = len([t for t in st.session_state.get('rerank_targets', [])
                                                    if t not in st.session_state.get('completed_reranks', [])]) - 1
                                    button_label = f"✅ Submit Rankings ({remaining} more to go)" if remaining > 0 else "✅ Submit Final Rankings"

                                    if st.button(button_label, key="cv_submit", type="primary", use_container_width=True):
                                        from backend.db import save_cross_validation, save_session_progress

                                        # Determine rerank_sequence for pilot study
                                        completed_reranks = st.session_state.get('completed_reranks', [])
                                        rerank_sequence = len(completed_reranks) + 1 if st.session_state.get('rerank_targets') else None
                                        source_token = cross_val.get('source_token')  # New field from get_assigned_search_for_validation

                                        success = save_cross_validation(
                                            reviewer_session_id=st.session_state.session_id,
                                            reviewed_session_id=cross_val['session_id'],
                                            reviewed_search_id=cross_val['search_id'],
                                            reviewed_prompt=cross_val['prompt'],
                                            reviewed_flights=cross_val['flights'],
                                            selected_flights=st.session_state.cross_val_selected_flights,
                                            reviewer_token=st.session_state.get('token'),
                                            rerank_sequence=rerank_sequence,
                                            source_token=source_token
                                        )
                                        if success:
                                            # PILOT STUDY: Handle sequential re-rankings
                                            if st.session_state.get('rerank_targets'):
                                                current_target = st.session_state.get('current_cv_target')
                                                if current_target:
                                                    # Add to completed list
                                                    if 'completed_reranks' not in st.session_state:
                                                        st.session_state.completed_reranks = []
                                                    st.session_state.completed_reranks.append(current_target)
                                                    st.session_state.current_rerank_index = len(st.session_state.completed_reranks)

                                                    # Save progress for session persistence
                                                    save_session_progress(st.session_state.get('token'), {
                                                        'current_rerank_index': st.session_state.current_rerank_index,
                                                        'completed_reranks_json': st.session_state.completed_reranks,
                                                    })
                                                    print(f"[PILOT] Completed re-ranking for {current_target} ({st.session_state.current_rerank_index}/{len(st.session_state.rerank_targets)})")

                                                    # Clear for next target
                                                    del st.session_state.cross_val_data
                                                    del st.session_state.current_cv_target
                                                    st.session_state.cross_val_selected_flights = []

                                                    # Check if all done
                                                    if len(st.session_state.completed_reranks) >= len(st.session_state.rerank_targets):
                                                        st.session_state.all_reranks_completed = True
                                                        st.session_state.cross_validation_completed = True
                                                        save_session_progress(st.session_state.get('token'), {
                                                            'all_reranks_completed': 1,
                                                            'current_phase': 'complete',
                                                        })
                                                        # Trigger backup on session completion
                                                        if not st.session_state.get('backup_triggered'):
                                                            st.session_state.backup_triggered = True
                                                            trigger_backup_on_completion(st.session_state.get('token', 'unknown'))
                                                        # Mark token as used (session complete)
                                                        if not st.session_state.get('token_marked_used'):
                                                            from backend.db import mark_token_used
                                                            token = st.session_state.get('token')
                                                            if token and token.upper() not in ["DEMO", "DATA"]:
                                                                mark_token_used(token)
                                                                st.session_state.token_marked_used = True
                                                                print(f"[PILOT] Token {token} marked as used (all re-rankings complete)")
                                                        st.success("✅ All re-rankings complete! Thank you!")
                                                    else:
                                                        st.success(f"✅ Submitted! Moving to next re-ranking...")
                                            else:
                                                # Legacy single re-ranking
                                                st.session_state.cross_validation_completed = True
                                                # Trigger backup on session completion
                                                if not st.session_state.get('backup_triggered'):
                                                    st.session_state.backup_triggered = True
                                                    trigger_backup_on_completion(st.session_state.get('token', 'unknown'))
                                                st.success("✅ Thank you for helping validate!")

                                            st.rerun()
                                        else:
                                            st.error("⚠️ Failed to save. Please try again.")
                            else:
                                st.caption("No flights selected yet")

                    # Close cross-validation div
                    st.markdown('</div>', unsafe_allow_html=True)

            # Show completion page with CSV download after cross-validation
            if (st.session_state.get('cross_validation_completed') and
                not st.session_state.get('completion_page_dismissed')):

                # Hide all previous content
                st.markdown("""
                <style>
                    /* Hide all content above completion page */
                    .main > div:not(:last-child) {
                        display: none !important;
                    }
                </style>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("# 🎉 Thank You for Participating!")
                st.markdown("---")

                st.markdown("""
                Thank you for completing our flight search study! Your feedback is invaluable
                to our research on AI-powered preference learning systems.

                **Your session has been saved successfully.**
                """)

                st.markdown("### 📥 Download Your Session Data")
                st.markdown("You can download your session data in two formats:")

                # Generate CSVs
                token = st.session_state.get('token')
                if token:
                    try:
                        from export_session_data import export_session_to_csv, export_simple_rankings_csv
                        import io

                        # Generate simplified rankings CSV
                        simple_csv_file = export_simple_rankings_csv(token, output_file=None)

                        if simple_csv_file:
                            with open(simple_csv_file, 'r', encoding='utf-8') as f:
                                simple_csv_data = f.read()

                            # Provide simplified download button
                            st.download_button(
                                label="📊 Download Rankings Only (Simple CSV)",
                                data=simple_csv_data,
                                file_name=f"rankings_{token}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                help="Contains: prompt, flights, and your rankings"
                            )

                            # Clean up temp file
                            import os
                            if os.path.exists(simple_csv_file):
                                os.remove(simple_csv_file)

                        # Generate comprehensive CSV
                        csv_file = export_session_to_csv(token, output_file=None)

                        # Read the CSV file
                        if csv_file:
                            with open(csv_file, 'r', encoding='utf-8') as f:
                                csv_data = f.read()

                            # Provide download button
                            st.download_button(
                                label="📄 Download Complete Session Data (Full CSV)",
                                data=csv_data,
                                file_name=f"session_data_{token}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                help="Contains: rankings, cross-validation, and survey"
                            )

                            st.success("✅ Your session data is ready for download!")

                            # Clean up temp file
                            import os
                            if os.path.exists(csv_file):
                                os.remove(csv_file)
                        else:
                            st.warning("⚠️ Could not generate CSV. Session may be incomplete.")

                    except Exception as e:
                        st.error(f"⚠️ Error generating CSV: {str(e)}")
                        import traceback
                        print(f"[CSV ERROR] {traceback.format_exc()}")

                st.markdown("---")

                # Button to proceed to end session
                if st.button("Continue →", type="primary", use_container_width=True):
                    st.session_state.completion_page_dismissed = True
                    st.rerun()

            # Show countdown if both cross-validation and survey completed
            if (st.session_state.get('cross_validation_completed') and
                st.session_state.get('survey_completed') and
                st.session_state.get('completion_page_dismissed') and
                st.session_state.get('countdown_started') and
                not st.session_state.get('countdown_completed')):
                import time
                st.info("📋 Please note your Search ID above. This session will end in:")

                # Create countdown display
                countdown_placeholder = st.empty()
                progress_placeholder = st.empty()

                for remaining in range(15, 0, -1):
                    countdown_placeholder.markdown(f"### ⏱️ {remaining} seconds")
                    progress_placeholder.progress((15 - remaining) / 15)
                    time.sleep(1)

                # Countdown complete
                countdown_placeholder.markdown("### ⏱️ 0 seconds")
                progress_placeholder.progress(1.0)

                # Mark token as used NOW (unless it's a special token like DEMO or DATA)
                if st.session_state.token.upper() not in ["DEMO", "DATA"]:
                    from backend.db import mark_token_used
                    mark_token_used(st.session_state.token)

                    # Mark countdown as completed
                    st.session_state.countdown_completed = True

                    # Show session ended message
                    st.warning("🔒 Session ended. Thank you for your participation!")
                    st.info("This link can no longer be used. To participate again, please request a new token from the research team.")
                    st.stop()
                else:
                    # Special tokens (DEMO/DATA) - allow continued use
                    st.session_state.countdown_completed = True
                    st.success("✅ Thank you! You can continue using this link to submit more rankings.")
                    if st.session_state.token.upper() == "DEMO":
                        st.info("💡 This is a DEMO token - you can use it as many times as you want!")
                    else:
                        st.info("📊 This is a DATA collection token - you can use it for unlimited submissions!")

                    # Clear the submission states to allow new submission
                    st.session_state.outbound_submitted = False
                    st.session_state.return_submitted = False
                    st.session_state.csv_generated = False
                    st.session_state.countdown_started = False
                    st.session_state.countdown_completed = False
                    st.session_state.selected_flights = []
                    st.session_state.selected_return_flights = []
                    st.session_state.all_flights = []
                    st.session_state.all_return_flights = []
                    st.session_state.review_confirmed = False
                    st.session_state.cross_validation_completed = False
                    st.session_state.cross_val_data = None
                    st.session_state.cross_val_selected = []
                    st.session_state.survey_completed = False
                    st.session_state.completion_page_dismissed = False

                    st.info("🔄 Refresh the page to start a new search!")
                    st.stop()
        elif st.session_state.get('db_save_error'):
            st.warning("⚠️ Rankings submitted but database save failed")
            st.error(f"Database error: {st.session_state.db_save_error}")
        else:
            # Debug: Show what we have in session state
            st.error(f"""
            ⚠️ Rankings submitted but no search ID!

            Debug info:
            - csv_generated: {st.session_state.get('csv_generated')}
            - search_id: {st.session_state.get('search_id')}
            - db_save_error: {st.session_state.get('db_save_error')}
            - csv_data_outbound exists: {bool(st.session_state.get('csv_data_outbound'))}
            - all_flights count: {len(st.session_state.all_flights) if st.session_state.all_flights else 0}
            - selected_flights count: {len(st.session_state.selected_flights) if st.session_state.selected_flights else 0}

            Failsafe should have triggered! Please screenshot this and report the bug.
            """)

        # Show options after results are submitted
        if st.session_state.get('search_id'):
            st.markdown("### What would you like to do next?")

            # New Search button takes full width
            if st.button("🔍 New Search", use_container_width=True, type="primary"):
                # Reset session state for new search
                st.session_state.all_flights = []
                st.session_state.selected_flights = []
                st.session_state.all_return_flights = []
                st.session_state.selected_return_flights = []
                st.session_state.csv_generated = False
                st.session_state.outbound_submitted = False
                st.session_state.return_submitted = False
                st.session_state.csv_data_outbound = None
                st.session_state.csv_data_return = None
                st.session_state.has_return = False
                st.session_state.parsed_params = None
                st.session_state.review_confirmed = False
                st.session_state.search_id = None
                st.session_state.db_save_error = None
                st.session_state.survey_completed = False
                st.session_state.completion_page_dismissed = False
                st.session_state.cross_validation_completed = False
                st.session_state.cross_val_data = None
                st.session_state.cross_val_selected_flights = []
                st.session_state.cv_filter_reset_counter = 0
                st.session_state.cv_sort_version = 0
                st.session_state.cv_checkbox_version = 0
                st.session_state.cv_sort_price_dir = 'asc'
                st.session_state.cv_sort_duration_dir = 'asc'
                # Delete the prompt key to reset it (can't set widget values directly)
                if 'flight_prompt_input' in st.session_state:
                    del st.session_state.flight_prompt_input
                st.rerun()

    else:
        # FLIGHT SELECTION INTERFACE
        # Persistent progress bar with pulsing animation
        progress_percent = (num_completed / num_required) if num_required > 0 else 0.0
        # Clamp to valid range [0.0, 1.0]
        progress_percent = max(0.0, min(1.0, progress_percent))

        # Custom persistent progress bar with pulsing animation (60s cycle: transparent, then gradual 5s to opaque, then gradual 5s back)
        st.markdown(f"""
            <style>
                @keyframes progressPulse {{
                    0%, 83.33% {{
                        opacity: 0.4;
                    }}
                    83.34% {{
                        opacity: 0.4;
                    }}
                    87.5% {{
                        opacity: 1;
                    }}
                    91.67% {{
                        opacity: 1;
                    }}
                    95.84% {{
                        opacity: 0.4;
                    }}
                    100% {{
                        opacity: 0.4;
                    }}
                }}
                .persistent-progress-container {{
                    position: fixed;
                    top: 60px;
                    left: 280px;
                    right: 20px;
                    z-index: 999;
                    background-color: rgba(255, 255, 255, 1);
                    padding: 8px 16px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    animation: progressPulse 60s ease-in-out infinite;
                }}
                .persistent-progress-bar {{
                    width: 100%;
                    height: 8px;
                    background-color: #e0e0e0;
                    border-radius: 4px;
                    overflow: hidden;
                }}
                .persistent-progress-fill {{
                    height: 100%;
                    background-color: #4CAF50;
                    width: {progress_percent * 100}%;
                    transition: width 0.3s ease;
                }}
                .persistent-progress-text {{
                    margin-top: 4px;
                    font-size: 12px;
                    color: #666;
                    text-align: center;
                    font-weight: 500;
                }}
            </style>
            <div class="persistent-progress-container">
                <div class="persistent-progress-bar">
                    <div class="persistent-progress-fill"></div>
                </div>
                <div class="persistent-progress-text">{num_completed} / {num_required} sections submitted</div>
            </div>
        """, unsafe_allow_html=True)

        # Check if we have return flights
        has_return = st.session_state.has_return and st.session_state.all_return_flights

        if has_return:
            st.markdown(f"### ✈️ Found {len(st.session_state.all_flights)} Outbound Flights and {len(st.session_state.all_return_flights)} Return Flights")
        else:
            st.markdown(f"### ✈️ Found {len(st.session_state.all_flights)} Flights")

        # Show current prompt with edit option (expanded by default so users see it)
        with st.expander("📝 Your Search Prompt (click to edit)", expanded=True):
            st.markdown("**Your prompt:**")
            st.info(st.session_state.get('original_prompt', ''))

            # Edit prompt toggle
            if 'editing_prompt_main' not in st.session_state:
                st.session_state.editing_prompt_main = False

            if not st.session_state.editing_prompt_main:
                if st.button("✏️ Make Edits", key="edit_prompt_main_btn"):
                    st.session_state.editing_prompt_main = True
                    st.rerun()
            else:
                edited_prompt_main = st.text_area(
                    "Edit your prompt:",
                    value=st.session_state.get('original_prompt', ''),
                    height=150,
                    key="edited_prompt_main"
                )

                col_save1, col_save2, col_cancel = st.columns([1, 1, 1])
                with col_save1:
                    if st.button("💾 Save", key="save_prompt_main"):
                        st.session_state.original_prompt = edited_prompt_main
                        st.session_state.editing_prompt_main = False
                        st.success("Prompt saved!")
                        st.rerun()
                with col_save2:
                    if st.button("🔄 Save & Search Again", key="save_search_prompt_main"):
                        st.session_state.original_prompt = edited_prompt_main
                        st.session_state.editing_prompt_main = False
                        # Set flag to auto-search with the new prompt
                        st.session_state.auto_search_requested = True
                        st.session_state.auto_search_prompt = edited_prompt_main
                        # Clear flight data
                        st.session_state.all_flights = []
                        st.session_state.all_return_flights = []
                        st.session_state.selected_flights = []
                        st.session_state.selected_return_flights = []
                        st.session_state.search_complete = False
                        st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel", key="cancel_prompt_main"):
                        st.session_state.editing_prompt_main = False
                        st.rerun()

        # Flight selection instructions
        if has_return:
            st.markdown("**Select your top 5 flights for EACH direction and drag to rank them →**")
        else:
            st.markdown("**Select your top 5 flights and drag to rank them →**")

        # SIDEBAR FILTERS
        with st.sidebar:
            # Add neon trace effect for Filters heading
            st.markdown("""
                <style>
                    /* Neon glow animation - runs for 10 seconds then fades out */
                    @keyframes neonGlow10s {
                        0% {
                            box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444;
                            border-color: #ff4444;
                        }
                        15% {
                            box-shadow: 0 0 8px #ff4444, 0 0 16px #ff4444, 0 0 24px #ff4444;
                            border-color: #ff6666;
                        }
                        30% {
                            box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444;
                            border-color: #ff4444;
                        }
                        45% {
                            box-shadow: 0 0 8px #ff4444, 0 0 16px #ff4444, 0 0 24px #ff4444;
                            border-color: #ff6666;
                        }
                        60% {
                            box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444;
                            border-color: #ff4444;
                        }
                        75% {
                            box-shadow: 0 0 8px #ff4444, 0 0 16px #ff4444, 0 0 24px #ff4444;
                            border-color: #ff6666;
                        }
                        90% {
                            box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444;
                            border-color: #ff4444;
                        }
                        100% {
                            box-shadow: none;
                            border-color: transparent;
                        }
                    }

                    /* Shrink padding after glow ends */
                    @keyframes shrinkPadding {
                        0% {
                            padding: 2px 6px;
                            margin: 0 2px;
                        }
                        100% {
                            padding: 0;
                            margin: 0;
                        }
                    }

                    .filter-heading-neon {
                        display: inline-block;
                        animation: neonGlow10s 10s ease-in-out forwards;
                        padding: 4px 12px;
                        border-radius: 6px;
                        border: 1.5px solid #ff4444;
                    }

                    .metric-neon {
                        display: inline-block;
                        animation:
                            neonGlow10s 10s ease-in-out forwards,
                            shrinkPadding 2s ease-in-out 10s forwards;
                        padding: 2px 6px;
                        border-radius: 4px;
                        border: 1px solid #ff4444;
                        margin: 0 2px;
                    }
                </style>
                <h2><span class="filter-heading-neon" key="filter-heading-{id(st.session_state)}">🔍 Filters</span></h2>
                <script>
                    // Force animation to replay on every Streamlit rerun
                    setTimeout(() => {{
                        const elem = document.querySelector('.filter-heading-neon');
                        if (elem) {{
                            elem.style.animation = 'none';
                            setTimeout(() => {{
                                elem.style.animation = 'neonGlow10s 10s ease-in-out forwards';
                            }}, 10);
                        }}
                    }}, 100);
                </script>
            """, unsafe_allow_html=True)

            # Get all flights for filter options (combine outbound and return if applicable)
            all_combined_flights = st.session_state.all_flights[:]
            if has_return:
                all_combined_flights.extend(st.session_state.all_return_flights)

            # Get unique airlines
            unique_airlines = sorted(set([f['airline'] for f in all_combined_flights]))
            airline_names_map = {code: get_airline_name(code) for code in unique_airlines}

            # Get unique connection counts
            unique_connections = sorted(set([f['stops'] for f in all_combined_flights]))

            # Get price range
            prices = [f['price'] for f in all_combined_flights]
            min_price, max_price = (min(prices), max(prices)) if prices else (0, 1000)

            # Get duration range
            durations = [f['duration_min'] for f in all_combined_flights]
            min_duration, max_duration = (min(durations), max(durations)) if durations else (0, 1440)

            # Get unique origins and destinations
            unique_origins = sorted(set([f['origin'] for f in all_combined_flights]))
            unique_destinations = sorted(set([f['destination'] for f in all_combined_flights]))

            # Check if we should show origin/destination filters
            # Only show if there are multiple origins or destinations WITHIN A LEG
            show_origin_filter = False
            show_dest_filter = False

            if has_return:
                # Check outbound leg
                outbound_origins = set([f['origin'] for f in st.session_state.all_flights])
                outbound_dests = set([f['destination'] for f in st.session_state.all_flights])
                # Check return leg
                return_origins = set([f['origin'] for f in st.session_state.all_return_flights])
                return_dests = set([f['destination'] for f in st.session_state.all_return_flights])

                # Show filter if ANY leg has multiple origins/destinations
                show_origin_filter = len(outbound_origins) > 1 or len(return_origins) > 1
                show_dest_filter = len(outbound_dests) > 1 or len(return_dests) > 1
            else:
                # Single leg - check if multiple origins/destinations
                show_origin_filter = len(unique_origins) > 1
                show_dest_filter = len(unique_destinations) > 1

            # Airline filter (expandable)
            with st.expander("✈️ Airlines", expanded=False):
                selected_airlines = []
                for airline_code in unique_airlines:
                    if st.checkbox(airline_names_map[airline_code], key=f"airline_{airline_code}_{st.session_state.filter_reset_counter}"):
                        selected_airlines.append(airline_code)
                st.session_state.filter_airlines = selected_airlines if selected_airlines else None

            # Origin filter (only show if multiple origins in a leg)
            if show_origin_filter:
                with st.expander("🛫 Origin Airport", expanded=False):
                    selected_origins = []
                    for origin_code in unique_origins:
                        if st.checkbox(origin_code, key=f"origin_{origin_code}_{st.session_state.filter_reset_counter}"):
                            selected_origins.append(origin_code)
                    st.session_state.filter_origins = selected_origins if selected_origins else None

            # Destination filter (only show if multiple destinations in a leg)
            if show_dest_filter:
                with st.expander("🛬 Destination Airport", expanded=False):
                    selected_destinations = []
                    for dest_code in unique_destinations:
                        if st.checkbox(dest_code, key=f"dest_{dest_code}_{st.session_state.filter_reset_counter}"):
                            selected_destinations.append(dest_code)
                    st.session_state.filter_destinations = selected_destinations if selected_destinations else None

            # Connections filter (expandable)
            with st.expander("🔄 Connections", expanded=False):
                selected_connections = []
                for conn_count in unique_connections:
                    conn_label = "Direct" if conn_count == 0 else f"{conn_count} stop{'s' if conn_count > 1 else ''}"
                    if st.checkbox(conn_label, key=f"conn_{conn_count}_{st.session_state.filter_reset_counter}"):
                        selected_connections.append(conn_count)
                st.session_state.filter_connections = selected_connections if selected_connections else None

            # Price filter (slider)
            with st.expander("💰 Price Range", expanded=False):
                price_range = st.slider(
                    "Select price range",
                    min_value=float(min_price),
                    max_value=float(max_price),
                    value=(float(min_price), float(max_price)),
                    step=10.0,
                    format="$%.0f",
                    key=f"filter_price_slider_{st.session_state.filter_reset_counter}"
                )
                # Only set if user changed from default
                if price_range != (float(min_price), float(max_price)):
                    st.session_state.filter_price_range = price_range
                else:
                    st.session_state.filter_price_range = None

            # Duration filter (slider)
            with st.expander("⏱️ Flight Duration", expanded=False):
                duration_range = st.slider(
                    "Select duration range",
                    min_value=int(min_duration),
                    max_value=int(max_duration),
                    value=(int(min_duration), int(max_duration)),
                    step=30,
                    format="%d min",
                    key=f"filter_duration_slider_{st.session_state.filter_reset_counter}"
                )
                # Convert to hours and minutes for display
                min_h, min_m = divmod(duration_range[0], 60)
                max_h, max_m = divmod(duration_range[1], 60)
                st.caption(f"{min_h}h {min_m}m - {max_h}h {max_m}m")

                # Only set if user changed from default
                if duration_range != (int(min_duration), int(max_duration)):
                    st.session_state.filter_duration_range = duration_range
                else:
                    st.session_state.filter_duration_range = None

            # Departure time filter (slider)
            with st.expander("🛫 Departure Time", expanded=False):
                dept_range = st.slider(
                    "Select departure time range",
                    min_value=0.0,
                    max_value=24.0,
                    value=(0.0, 24.0),
                    step=0.5,
                    format="%.1f",
                    key=f"filter_departure_slider_{st.session_state.filter_reset_counter}"
                )
                # Convert to readable time
                def hours_to_time(h):
                    hours = int(h)
                    mins = int((h - hours) * 60)
                    return f"{hours:02d}:{mins:02d}"

                st.caption(f"{hours_to_time(dept_range[0])} - {hours_to_time(dept_range[1])}")

                # Only set if user changed from default
                if dept_range != (0.0, 24.0):
                    st.session_state.filter_departure_time_range = dept_range
                else:
                    st.session_state.filter_departure_time_range = None

            # Arrival time filter (slider)
            with st.expander("🛬 Arrival Time", expanded=False):
                arr_range = st.slider(
                    "Select arrival time range",
                    min_value=0.0,
                    max_value=24.0,
                    value=(0.0, 24.0),
                    step=0.5,
                    format="%.1f",
                    key=f"filter_arrival_slider_{st.session_state.filter_reset_counter}"
                )
                st.caption(f"{hours_to_time(arr_range[0])} - {hours_to_time(arr_range[1])}")

                # Only set if user changed from default
                if arr_range != (0.0, 24.0):
                    st.session_state.filter_arrival_time_range = arr_range
                else:
                    st.session_state.filter_arrival_time_range = None

            # Clear all filters button
            if st.button("Clear All Filters", use_container_width=True):
                # Clear filter session state variables
                st.session_state.filter_airlines = None
                st.session_state.filter_connections = None
                st.session_state.filter_price_range = None
                st.session_state.filter_duration_range = None
                st.session_state.filter_departure_time_range = None
                st.session_state.filter_arrival_time_range = None
                st.session_state.filter_origins = None
                st.session_state.filter_destinations = None

                # Increment reset counter to force all widgets to recreate with default values
                st.session_state.filter_reset_counter += 1

                st.rerun()

        # Apply filters to flight lists
        filtered_outbound = apply_filters(
            st.session_state.all_flights,
            airlines=st.session_state.filter_airlines,
            connections=st.session_state.filter_connections,
            price_range=st.session_state.filter_price_range,
            duration_range=st.session_state.filter_duration_range,
            departure_range=st.session_state.filter_departure_time_range,
            arrival_range=st.session_state.filter_arrival_time_range,
            origins=st.session_state.filter_origins,
            destinations=st.session_state.filter_destinations
        )

        filtered_return = []
        if has_return:
            filtered_return = apply_filters(
                st.session_state.all_return_flights,
                airlines=st.session_state.filter_airlines,
                connections=st.session_state.filter_connections,
                price_range=st.session_state.filter_price_range,
                duration_range=st.session_state.filter_duration_range,
                departure_range=st.session_state.filter_departure_time_range,
                arrival_range=st.session_state.filter_arrival_time_range,
                origins=st.session_state.filter_origins,
                destinations=st.session_state.filter_destinations
            )

        # Check if there are any codeshares to determine if we should show the info banner
        outbound_codeshares = detect_codeshares(st.session_state.all_flights)
        return_codeshares = detect_codeshares(st.session_state.all_return_flights) if has_return else {}
        has_codeshares = any(outbound_codeshares.values()) or any(return_codeshares.values())

        # Info banner about codeshares (only show if codeshares exist)
        if has_codeshares:
            with st.container():
                st.info(
                    "ℹ️ **About Your Results:** Codeshares can show the same flight under different "
                    "airlines at the same or different prices—we label these so you know it's one aircraft."
                )

                with st.expander("📖 Learn More: Why Do I See Multiple Entries for the Same Flight?"):
                    st.markdown("""
                    ### FAQ: Codeshare Flights Explained

                    **Q: Why do some flights appear multiple times in search results?**

                    Many airlines operate "codeshare" flights. This means one airline operates the plane,
                    but multiple partner airlines sell seats on that same physical flight under different
                    flight numbers.

                    **Example:**
                    - United flight UA123
                    - Lufthansa flight LH9001

                    Both may refer to the same aircraft, same departure and arrival times, same route,
                    and often the same price.

                    ---

                    **Q: Are these duplicate flights?**

                    Not technically. Each codeshare entry is a different booking option, even though
                    they correspond to the same aircraft.
                    """)

        # CONDITIONAL UI: Show single or dual panels based on has_return
        if has_return:
            # DUAL PANEL LAYOUT: Outbound + Return
            st.markdown('<div id="top-of-page"></div>', unsafe_allow_html=True)
            st.markdown('<div id="outbound-flights"></div>', unsafe_allow_html=True)
            st.markdown("## 🛫 Outbound Flights")

            col_flights_out, col_ranking_out = st.columns([2, 1])

            with col_flights_out:
                st.markdown("#### All Outbound Flights")

                # Show filter status if filters are active
                if len(filtered_outbound) < len(st.session_state.all_flights):
                    st.info(f"🔍 Filters applied: Showing {len(filtered_outbound)} of {len(st.session_state.all_flights)} outbound flights")

                # Sort buttons with direction arrows
                col_sort1, col_sort2 = st.columns(2)
                with col_sort1:
                    arrow = "↑" if st.session_state.sort_price_dir == 'asc' else "↓"
                    if st.button(f"💰 Sort by Price {arrow}", key="sort_price_out", use_container_width=True):
                        # Toggle direction
                        reverse = st.session_state.sort_price_dir == 'desc'
                        st.session_state.all_flights = sorted(st.session_state.all_flights, key=lambda x: x['price'], reverse=reverse)
                        st.session_state.sort_price_dir = 'desc' if st.session_state.sort_price_dir == 'asc' else 'asc'
                        st.rerun()
                with col_sort2:
                    arrow = "↑" if st.session_state.sort_duration_dir == 'asc' else "↓"
                    if st.button(f"⏱️ Sort by Duration {arrow}", key="sort_duration_out", use_container_width=True):
                        # Toggle direction
                        reverse = st.session_state.sort_duration_dir == 'desc'
                        st.session_state.all_flights = sorted(st.session_state.all_flights, key=lambda x: x['duration_min'], reverse=reverse)
                        st.session_state.sort_duration_dir = 'desc' if st.session_state.sort_duration_dir == 'asc' else 'asc'
                        st.rerun()

                # Detect codeshares in outbound flights (on filtered list)
                outbound_codeshare_map = detect_codeshares(filtered_outbound)

                # Add neon trace effect CSS (only for first flight in first 10 seconds)
                st.markdown("""
                    <style>
                        @keyframes redNeonPulse {
                            0% {
                                box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444, 0 0 9px #ff4444;
                                border-color: #ff4444;
                                transform: scale(1);
                            }
                            50% {
                                box-shadow: 0 0 6px #ff4444, 0 0 12px #ff4444, 0 0 18px #ff4444;
                                border-color: #ff6666;
                                transform: scale(1.05);
                            }
                            100% {
                                box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444, 0 0 9px #ff4444;
                                border-color: #ff4444;
                                transform: scale(1);
                            }
                        }
                        .neon-metric-box {
                            display: inline-block;
                            animation: redNeonPulse 2s ease-in-out infinite;
                            padding: 3px 8px;
                            border-radius: 5px;
                            margin: 0 3px;
                            border: 2px solid #ff4444;
                            box-shadow: 0 0 3px #ff4444, 0 0 6px #ff4444, 0 0 9px #ff4444;
                            transition: all 0.3s ease;
                        }
                    </style>
                """, unsafe_allow_html=True)

                # Display all outbound flights with checkboxes
                for idx, flight in enumerate(filtered_outbound):
                    unique_id = f"{flight['origin']}_{flight['destination']}{idx + 1}"

                    # Add unique index to flight for tracking
                    flight['_ui_index'] = idx

                    # Check if selected using _ui_index
                    is_selected = any(f.get('_ui_index') == idx for f in st.session_state.selected_flights)

                    col1, col2 = st.columns([1, 5])

                    with col1:
                        # Use flight index for checkbox key
                        checkbox_key = f"chk_out_{idx}_v{st.session_state.checkbox_version}"
                        selected = st.checkbox(
                            "Select flight",
                            value=is_selected,
                            key=checkbox_key,
                            label_visibility="collapsed",
                            disabled=(not is_selected and len(st.session_state.selected_flights) >= 5)
                        )

                        if selected and not is_selected:
                            if len(st.session_state.selected_flights) < 5:
                                st.session_state.selected_flights.append(flight)
                        elif not selected and is_selected:
                            st.session_state.selected_flights = [
                                f for f in st.session_state.selected_flights
                                if f.get('_ui_index') != idx
                            ]

                    with col2:
                        dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                        arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                        dept_time_display = dept_dt.strftime("%I:%M %p")
                        arr_time_display = arr_dt.strftime("%I:%M %p")
                        dept_date_display = dept_dt.strftime("%a, %b %d")  # e.g., "Fri, Jan 5"

                        duration_hours = flight['duration_min'] // 60
                        duration_mins = flight['duration_min'] % 60
                        duration_display = f"{duration_hours} hr {duration_mins} min" if duration_hours > 0 else f"{duration_mins} min"

                        airline_name = get_airline_name(flight['airline'])

                        # Check if this flight is a codeshare
                        codeshare_label = ""
                        if outbound_codeshare_map.get(idx, False):
                            codeshare_label = '<span style="font-size: 0.75em; color: #666; font-style: italic;"> (Codeshare)</span>'

                        # Format stops for display
                        stops_text = "Direct" if flight['stops'] == 0 else f"{flight['stops']} stop{'s' if flight['stops'] > 1 else ''}"

                        # Apply neon boxes only to first flight (7 small boxes for each metric)
                        neon_class = 'class="neon-metric-box"' if idx == 0 else ''

                        st.markdown(f"""
                        <div style="line-height: 1.4; margin: 0; padding: 0.4rem 0; border-bottom: 1px solid #eee;">
                        <div style="font-size: 1.1em; margin-bottom: 0.2rem;">
                            <span {neon_class} style="font-weight: 700;">{format_price(flight['price'])}</span> •
                            <span {neon_class} style="font-weight: 600;">{duration_display}</span> •
                            <span {neon_class} style="font-weight: 500;">{stops_text}</span> •
                            <span {neon_class} style="font-weight: 500;">{dept_time_display} - {arr_time_display}</span>
                        </div>
                        <div style="font-size: 0.9em; color: #666;">
                            <span {neon_class}>{airline_name} {flight['flight_number']}{codeshare_label}</span> |
                            <span {neon_class}>{flight['origin']} → {flight['destination']}</span> |
                            <span {neon_class}>{dept_date_display}</span>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)

            with col_ranking_out:
                st.markdown("#### 📋 Top 5 Outbound (Drag to Rank)")
                st.markdown(f"**{len(st.session_state.selected_flights)}/5 selected**")

                if st.session_state.selected_flights:
                    # Enforce limit
                    if len(st.session_state.selected_flights) > 5:
                        st.session_state.selected_flights = st.session_state.selected_flights[:5]
                        st.rerun()

                    # Create draggable items with X buttons inline
                    draggable_items = []
                    for i, flight in enumerate(st.session_state.selected_flights):
                        airline_name = get_airline_name(flight['airline'])

                        # Format flight data same way as main list
                        from datetime import datetime
                        dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                        arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                        dept_time_display = dept_dt.strftime("%I:%M %p")
                        arr_time_display = arr_dt.strftime("%I:%M %p")
                        dept_date_display = dept_dt.strftime("%a, %b %d")

                        duration_hours = flight['duration_min'] // 60
                        duration_mins = flight['duration_min'] % 60
                        duration_display = f"{duration_hours}h {duration_mins}m" if duration_hours > 0 else f"{duration_mins}m"

                        stops = int(flight.get('stops', 0))
                        stops_text = "Nonstop" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"

                        # Create draggable item string
                        item = f"""#{i+1}: {format_price(flight['price'])} • {duration_display} • {stops_text}
{dept_time_display} - {arr_time_display}
{airline_name} {flight['flight_number']}
{flight['origin']} → {flight['destination']} | {dept_date_display}"""
                        draggable_items.append(item)

                    # Sortable list
                    sorted_items = sort_items(
                        draggable_items,
                        multi_containers=False,
                        direction='vertical',
                        key=f'outbound_sort_v{st.session_state.outbound_sort_version}_n{len(st.session_state.selected_flights)}'
                    )

                    # Update order if dragged
                    if sorted_items != draggable_items and len(sorted_items) == len(draggable_items):
                        new_order = []
                        for sorted_item in sorted_items:
                            rank = int(sorted_item.split(':')[0].replace('#', '')) - 1
                            if rank < len(st.session_state.selected_flights):
                                new_order.append(st.session_state.selected_flights[rank])
                        if len(new_order) == len(st.session_state.selected_flights):
                            st.session_state.selected_flights = new_order
                            st.session_state.outbound_sort_version += 1
                            st.rerun()

                    # X buttons below
                    st.markdown("---")
                    cols = st.columns(5)
                    for i, flight in enumerate(st.session_state.selected_flights):
                        with cols[i]:
                            if st.button("✖", key=f"remove_outbound_{i}_{flight['id']}", help=f"Remove #{i+1}"):
                                flight_unique_key = f"{flight['id']}_{flight['departure_time']}"

                                # Remove the flight from selected list
                                st.session_state.selected_flights = [
                                    f for f in st.session_state.selected_flights
                                    if f"{f['id']}_{f['departure_time']}" != flight_unique_key
                                ]

                                # Increment version to force checkbox recreation
                                st.session_state.checkbox_version += 1

                                # Clear the checkbox widget state
                                checkbox_key = f"chk_out_{flight_unique_key}".replace(':', '').replace('-', '').replace('+', '')
                                if checkbox_key in st.session_state:
                                    del st.session_state[checkbox_key]

                                st.rerun()

                    # Submit button for outbound
                    if len(st.session_state.selected_flights) == 5 and not st.session_state.outbound_submitted:
                        if st.button("✅ Submit Outbound Rankings", key="submit_outbound", type="primary", use_container_width=True):
                            # Generate CSV
                            csv_data = generate_flight_csv(
                                st.session_state.all_flights,
                                st.session_state.selected_flights,
                                k=5
                            )
                            st.session_state.csv_data_outbound = csv_data
                            st.session_state.outbound_submitted = True
                            st.success("✅ Outbound rankings submitted!")
                            st.rerun()
                    elif st.session_state.outbound_submitted:
                        st.success("✅ Outbound rankings submitted")
                else:
                    st.info("Select 5 outbound flights")

            # Subway-line navigation on the left side
#             nav_items = ['How to Use', 'Outbound', 'Return']
#             nav_ids = ['how-to-use', 'outbound-flights', 'return-flights']
# 
#             st.markdown(f"""
#                 <style>
#                     .subway-nav {{
#                         position: fixed;
#                         left: 30px;
#                         top: 50%;
#                         transform: translateY(-50%);
#                         z-index: 1000;
#                         padding: 0;
#                         transition: left 0.3s ease;
#                     }}
#                     /* Adjust position when sidebar is open */
#                     [data-testid="stSidebar"]:not([aria-hidden="true"]) ~ div .subway-nav {{
#                         left: 340px;  /* 280px sidebar + 30px margin + 30px spacing = 340px */
#                     }}
#                     .subway-nav ul {{
#                         list-style: none;
#                         padding: 0;
#                         margin: 0;
#                         position: relative;
#                     }}
#                     /* Vertical line connecting stations */
#                     .subway-nav ul::before {{
#                         content: '';
#                         position: absolute;
#                         left: 12px;
#                         top: 20px;
#                         bottom: 20px;
#                         width: 2px;
#                         background-color: rgba(150, 150, 150, 0.3);
#                         z-index: 0;
#                     }}
#                     .subway-nav li {{
#                         position: relative;
#                         margin: 40px 0;
#                     }}
#                     .subway-nav li:first-child {{
#                         margin-top: 0;
#                     }}
#                     .subway-nav li:last-child {{
#                         margin-bottom: 0;
#                     }}
#                     /* Station circles */
#                     .subway-nav a {{
#                         display: flex;
#                         align-items: center;
#                         text-decoration: none;
#                         position: relative;
#                         z-index: 1;
#                     }}
#                     .subway-nav a .station-circle {{
#                         width: 20px;
#                         height: 20px;
#                         border-radius: 50%;
#                         background-color: rgba(255, 255, 255, 0.7);
#                         border: 4px solid #FF6B35;
#                         position: relative;
#                         transition: all 0.3s ease;
#                         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
#                     }}
#                     .subway-nav a:hover .station-circle {{
#                         transform: scale(1.3);
#                         border-width: 5px;
#                         box-shadow: 0 3px 8px rgba(255, 107, 53, 0.4);
#                     }}
#                     .subway-nav a.active .station-circle {{
#                         background-color: #FF6B35;
#                         border-color: #E55A2B;
#                         box-shadow: 0 0 12px rgba(255, 107, 53, 0.6);
#                     }}
#                     /* Station labels */
#                     .subway-nav a .station-label {{
#                         position: absolute;
#                         left: 35px;
#                         white-space: nowrap;
#                         background-color: rgba(0, 0, 0, 0.85);
#                         color: white;
#                         padding: 6px 12px;
#                         border-radius: 6px;
#                         font-size: 13px;
#                         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
#                         font-weight: 500;
#                         opacity: 0;
#                         pointer-events: none;
#                         transition: opacity 0.2s ease;
#                         box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
#                     }}
#                     .subway-nav a:hover .station-label {{
#                         opacity: 1;
#                     }}
#                     .subway-nav a.active .station-label {{
#                         opacity: 1;
#                         background-color: rgba(255, 107, 53, 0.95);
#                     }}
#                 </style>
#                 <div class="subway-nav">
#                     <ul>
#                         {''.join(f'<li><a href="#{nav_ids[i]}"><div class="station-circle"></div><div class="station-label">{nav_items[i]}</div></a></li>' for i in range(len(nav_items)))}
#                     </ul>
#                 </div>
#                 <script>
#                     // Update active state based on scroll position
#                     function updateSubwayNav() {{
#                         const sections = {nav_ids};
#                         const navLinks = document.querySelectorAll('.subway-nav a');
# 
#                         let currentSection = '';
#                         sections.forEach((sectionId, index) => {{
#                             const section = document.getElementById(sectionId);
#                             if (section) {{
#                                 const rect = section.getBoundingClientRect();
#                                 if (rect.top <= window.innerHeight / 3) {{
#                                     currentSection = sectionId;
#                                 }}
#                             }}
#                         }});
# 
#                         navLinks.forEach(link => {{
#                             link.classList.remove('active');
#                             if (link.getAttribute('href') === '#' + currentSection) {{
#                                 link.classList.add('active');
#                             }}
#                         }});
#                     }}
# 
#                     window.addEventListener('scroll', updateSubwayNav);
#                     setTimeout(updateSubwayNav, 200);
#                 </script>
#             """, unsafe_allow_html=True)

            # RETURN FLIGHTS SECTION
            st.markdown('<div id="return-flights"></div>', unsafe_allow_html=True)
            st.markdown("## 🛬 Return Flights")

            col_flights_ret, col_ranking_ret = st.columns([2, 1])

            with col_flights_ret:
                st.markdown("#### All Return Flights")

                # Show filter status if filters are active
                if len(filtered_return) < len(st.session_state.all_return_flights):
                    st.info(f"🔍 Filters applied: Showing {len(filtered_return)} of {len(st.session_state.all_return_flights)} return flights")

                # Sort buttons with direction arrows
                col_sort1, col_sort2 = st.columns(2)
                with col_sort1:
                    arrow = "↑" if st.session_state.sort_price_dir_ret == 'asc' else "↓"
                    if st.button(f"💰 Sort by Price {arrow}", key="sort_price_ret", use_container_width=True):
                        # Toggle direction
                        reverse = st.session_state.sort_price_dir_ret == 'desc'
                        st.session_state.all_return_flights = sorted(st.session_state.all_return_flights, key=lambda x: x['price'], reverse=reverse)
                        st.session_state.sort_price_dir_ret = 'desc' if st.session_state.sort_price_dir_ret == 'asc' else 'asc'
                        st.rerun()
                with col_sort2:
                    arrow = "↑" if st.session_state.sort_duration_dir_ret == 'asc' else "↓"
                    if st.button(f"⏱️ Sort by Duration {arrow}", key="sort_duration_ret", use_container_width=True):
                        # Toggle direction
                        reverse = st.session_state.sort_duration_dir_ret == 'desc'
                        st.session_state.all_return_flights = sorted(st.session_state.all_return_flights, key=lambda x: x['duration_min'], reverse=reverse)
                        st.session_state.sort_duration_dir_ret = 'desc' if st.session_state.sort_duration_dir_ret == 'asc' else 'asc'
                        st.rerun()

                # Detect codeshares in return flights (on filtered list)
                return_codeshare_map = detect_codeshares(filtered_return)

                # Display all return flights with checkboxes
                for idx, flight in enumerate(filtered_return):
                    unique_id = f"{flight['origin']}_{flight['destination']}{idx + 1}"

                    # Track flight by index for reliable selection
                    flight['_ui_index'] = idx
                    is_selected = any(f.get('_ui_index') == idx for f in st.session_state.selected_return_flights)

                    col1, col2 = st.columns([1, 5])

                    with col1:
                        # Use simple index-based key
                        checkbox_key = f"chk_ret_{idx}_v{st.session_state.checkbox_version}"
                        selected = st.checkbox(
                            "Select flight",
                            value=is_selected,
                            key=checkbox_key,
                            label_visibility="collapsed",
                            disabled=(not is_selected and len(st.session_state.selected_return_flights) >= 5)
                        )

                        if selected and not is_selected:
                            if len(st.session_state.selected_return_flights) < 5:
                                st.session_state.selected_return_flights.append(flight)
                        elif not selected and is_selected:
                            st.session_state.selected_return_flights = [
                                f for f in st.session_state.selected_return_flights
                                if f.get('_ui_index') != idx
                            ]

                    with col2:
                        dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                        arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                        dept_time_display = dept_dt.strftime("%I:%M %p")
                        arr_time_display = arr_dt.strftime("%I:%M %p")
                        dept_date_display = dept_dt.strftime("%a, %b %d")  # e.g., "Fri, Jan 5"

                        duration_hours = flight['duration_min'] // 60
                        duration_mins = flight['duration_min'] % 60
                        duration_display = f"{duration_hours} hr {duration_mins} min" if duration_hours > 0 else f"{duration_mins} min"

                        airline_name = get_airline_name(flight['airline'])

                        # Check if this flight is a codeshare
                        codeshare_label = ""
                        if return_codeshare_map.get(idx, False):
                            codeshare_label = '<span style="font-size: 0.75em; color: #666; font-style: italic;"> (Codeshare)</span>'

                        # Format stops for display
                        stops_text = "Direct" if flight['stops'] == 0 else f"{flight['stops']} stop{'s' if flight['stops'] > 1 else ''}"

                        # Add neon glow to first flight only
                        neon_class = "metric-neon" if idx == 0 else ""

                        st.markdown(f"""
                        <div style="line-height: 1.4; margin: 0; padding: 0.4rem 0; border-bottom: 1px solid #eee;">
                        <div style="font-size: 1.1em; margin-bottom: 0.2rem;">
                            <span class="{neon_class}" style="font-weight: 700;">{format_price(flight['price'])}</span> •
                            <span class="{neon_class}" style="font-weight: 600;">{duration_display}</span> •
                            <span class="{neon_class}" style="font-weight: 500;">{stops_text}</span> •
                            <span class="{neon_class}" style="font-weight: 500;">{dept_time_display} - {arr_time_display}</span>
                        </div>
                        <div style="font-size: 0.9em; color: #666;">
                            <span class="{neon_class}">{airline_name} {flight['flight_number']}{codeshare_label}</span> | <span class="{neon_class}">{flight['origin']} → {flight['destination']}</span> | <span class="{neon_class}">{dept_date_display}</span>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)

            with col_ranking_ret:
                st.markdown("#### 📋 Top 5 Return (Drag to Rank)")
                st.markdown(f"**{len(st.session_state.selected_return_flights)}/5 selected**")

                if st.session_state.selected_return_flights:
                    # Enforce limit
                    if len(st.session_state.selected_return_flights) > 5:
                        st.session_state.selected_return_flights = st.session_state.selected_return_flights[:5]
                        st.rerun()

                    # Create draggable items with X buttons inline
                    draggable_items = []
                    for i, flight in enumerate(st.session_state.selected_return_flights):
                        airline_name = get_airline_name(flight['airline'])

                        # Format flight data same way as main list
                        from datetime import datetime
                        dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                        arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                        dept_time_display = dept_dt.strftime("%I:%M %p")
                        arr_time_display = arr_dt.strftime("%I:%M %p")
                        dept_date_display = dept_dt.strftime("%a, %b %d")

                        duration_hours = flight['duration_min'] // 60
                        duration_mins = flight['duration_min'] % 60
                        duration_display = f"{duration_hours}h {duration_mins}m" if duration_hours > 0 else f"{duration_mins}m"

                        stops = int(flight.get('stops', 0))
                        stops_text = "Nonstop" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"

                        # Create draggable item string
                        item = f"""#{i+1}: {format_price(flight['price'])} • {duration_display} • {stops_text}
{dept_time_display} - {arr_time_display}
{airline_name} {flight['flight_number']}
{flight['origin']} → {flight['destination']} | {dept_date_display}"""
                        draggable_items.append(item)

                    # Sortable list
                    sorted_items = sort_items(
                        draggable_items,
                        multi_containers=False,
                        direction='vertical',
                        key=f'return_sort_v{st.session_state.return_sort_version}_n{len(st.session_state.selected_return_flights)}'
                    )

                    # Update order if dragged
                    if sorted_items != draggable_items and len(sorted_items) == len(draggable_items):
                        new_order = []
                        for sorted_item in sorted_items:
                            rank = int(sorted_item.split(':')[0].replace('#', '')) - 1
                            if rank < len(st.session_state.selected_return_flights):
                                new_order.append(st.session_state.selected_return_flights[rank])
                        if len(new_order) == len(st.session_state.selected_return_flights):
                            st.session_state.selected_return_flights = new_order
                            st.session_state.return_sort_version += 1
                            st.rerun()

                    # X buttons below
                    st.markdown("---")
                    cols = st.columns(5)
                    for i, flight in enumerate(st.session_state.selected_return_flights):
                        with cols[i]:
                            if st.button("✖", key=f"remove_return_{i}_{flight['id']}", help=f"Remove #{i+1}"):
                                flight_unique_key = f"{flight['id']}_{flight['departure_time']}"

                                # Remove the flight from selected list
                                st.session_state.selected_return_flights = [
                                    f for f in st.session_state.selected_return_flights
                                    if f"{f['id']}_{f['departure_time']}" != flight_unique_key
                                ]

                                # Increment version to force checkbox recreation
                                st.session_state.checkbox_version += 1

                                # Clear the checkbox widget state
                                checkbox_key = f"chk_ret_{flight_unique_key}".replace(':', '').replace('-', '').replace('+', '')
                                if checkbox_key in st.session_state:
                                    del st.session_state[checkbox_key]

                                st.rerun()

                    # Submit button for return
                    if len(st.session_state.selected_return_flights) == 5 and not st.session_state.return_submitted:
                        if st.button("✅ Submit Return Rankings", key="submit_return", type="primary", use_container_width=True):
                            # Generate CSV
                            csv_data = generate_flight_csv(
                                st.session_state.all_return_flights,
                                st.session_state.selected_return_flights,
                                k=5
                            )
                            st.session_state.csv_data_return = csv_data
                            st.session_state.return_submitted = True
                            st.success("✅ Return rankings submitted!")
                            st.rerun()
                    elif st.session_state.return_submitted:
                        st.success("✅ Return rankings submitted")
                else:
                    st.info("Select 5 return flights")

            # Save to database when both are submitted
            if st.session_state.outbound_submitted and st.session_state.return_submitted and not st.session_state.csv_generated:
                try:
                    from backend.db import save_search_and_csv, SessionLocal, FlightCSV
                    import traceback

                    # For DATA token, delete all previous submissions to allow overwriting
                    if st.session_state.token.upper() == "DATA":
                        from backend.db import Search, UserRanking, FlightShown, CrossValidation, SurveyResponse
                        db = SessionLocal()
                        try:
                            # Find all previous searches for this token
                            previous_searches = db.query(Search).filter(
                                (Search.session_id == st.session_state.token) | (Search.completion_token == st.session_state.token)
                            ).all()

                            for search in previous_searches:
                                # Delete cross-validation data
                                db.query(CrossValidation).filter_by(reviewer_session_id=search.session_id).delete()
                                db.query(CrossValidation).filter_by(reviewed_session_id=search.session_id).delete()

                                # Delete survey data
                                db.query(SurveyResponse).filter_by(session_id=search.session_id).delete()

                                # Delete rankings and flights
                                rankings = db.query(UserRanking).filter_by(search_id=search.search_id).all()
                                for ranking in rankings:
                                    db.query(FlightShown).filter_by(id=ranking.flight_id).delete()
                                    db.delete(ranking)

                                # Delete CSV records
                                db.query(FlightCSV).filter_by(search_id=search.search_id).delete()

                                # Delete the search itself
                                db.delete(search)

                            db.commit()
                            print(f"[DEBUG] Deleted all previous DATA token submissions (dual panel)")
                        except Exception as e:
                            db.rollback()
                            print(f"[DEBUG] Error deleting previous DATA submissions (dual panel): {str(e)}")
                        finally:
                            db.close()

                    # Save outbound as primary
                    search_id = save_search_and_csv(
                        session_id=st.session_state.session_id,
                        user_prompt=st.session_state.get('original_prompt', ''),
                        parsed_params=st.session_state.parsed_params or {},
                        all_flights=st.session_state.all_flights,
                        selected_flights=st.session_state.selected_flights,
                        csv_data=st.session_state.csv_data_outbound,
                        token=st.session_state.token
                    )

                    # Note: Token will be marked as used AFTER the 15-second countdown
                    # (see countdown logic below where search_id is displayed)

                    # Also save return flight CSV to the same search
                    if st.session_state.csv_data_return:
                        db = SessionLocal()
                        try:
                            csv_record = FlightCSV(
                                session_id=st.session_state.session_id,
                                search_id=search_id,
                                csv_data=st.session_state.csv_data_return,
                                num_flights=len(st.session_state.all_return_flights),
                                num_selected=len(st.session_state.selected_return_flights)
                            )
                            db.add(csv_record)
                            db.commit()
                            print(f"✓ Saved return flight CSV to search {search_id}")
                        finally:
                            db.close()

                    st.session_state.csv_generated = True
                    st.session_state.search_id = search_id
                    st.session_state.countdown_started = True  # Start countdown phase
                    st.success("✅ Rankings saved to database!")
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"⚠️ Failed to save rankings to database: {str(e)}")
                    st.code(traceback.format_exc())
                    st.session_state.csv_generated = True
                    st.session_state.db_save_error = str(e)
                    st.rerun()

        else:
            # SINGLE PANEL LAYOUT: Outbound only
            st.markdown('<div id="top-of-page"></div>', unsafe_allow_html=True)
            st.markdown('<div id="outbound-flights"></div>', unsafe_allow_html=True)
            st.markdown("## 🛫 Outbound Flights")

            col_flights, col_ranking = st.columns([2, 1])

            with col_flights:
                st.markdown("#### All Available Flights")

                # Show filter status if filters are active
                if len(filtered_outbound) < len(st.session_state.all_flights):
                    st.info(f"🔍 Filters applied: Showing {len(filtered_outbound)} of {len(st.session_state.all_flights)} flights")

                # Sort buttons with direction arrows
                col_sort1, col_sort2 = st.columns(2)
                with col_sort1:
                    arrow = "↑" if st.session_state.sort_price_dir_single == 'asc' else "↓"
                    if st.button(f"💰 Sort by Price {arrow}", key="sort_price_single", use_container_width=True):
                        # Toggle direction
                        reverse = st.session_state.sort_price_dir_single == 'desc'
                        st.session_state.all_flights = sorted(st.session_state.all_flights, key=lambda x: x['price'], reverse=reverse)
                        st.session_state.sort_price_dir_single = 'desc' if st.session_state.sort_price_dir_single == 'asc' else 'asc'
                        st.rerun()
                with col_sort2:
                    arrow = "↑" if st.session_state.sort_duration_dir_single == 'asc' else "↓"
                    if st.button(f"⏱️ Sort by Duration {arrow}", key="sort_duration_single", use_container_width=True):
                        # Toggle direction
                        reverse = st.session_state.sort_duration_dir_single == 'desc'
                        st.session_state.all_flights = sorted(st.session_state.all_flights, key=lambda x: x['duration_min'], reverse=reverse)
                        st.session_state.sort_duration_dir_single = 'desc' if st.session_state.sort_duration_dir_single == 'asc' else 'asc'
                        st.rerun()

                # Detect codeshares in outbound flights (on filtered list)
                outbound_codeshare_map = detect_codeshares(filtered_outbound)

                # Display all flights with checkboxes
                for idx, flight in enumerate(filtered_outbound):
                    # Generate unique_id for display
                    unique_id = f"{flight['origin']}_{flight['destination']}{idx + 1}"

                    # Add unique index to flight for tracking
                    flight['_ui_index'] = idx

                    # Check if selected using _ui_index
                    is_selected = any(f.get('_ui_index') == idx for f in st.session_state.selected_flights)

                    col1, col2 = st.columns([1, 5])

                    with col1:
                        # Use flight index for checkbox key
                        checkbox_key = f"chk_single_{idx}_v{st.session_state.checkbox_version}"
                        selected = st.checkbox(
                            "Select flight",
                            value=is_selected,
                            key=checkbox_key,
                            label_visibility="collapsed",
                            disabled=(not is_selected and len(st.session_state.selected_flights) >= 5)
                        )

                        if selected and not is_selected:
                            # Add to selected flights
                            if len(st.session_state.selected_flights) < 5:
                                st.session_state.selected_flights.append(flight)
                        elif not selected and is_selected:
                            # Remove from selected flights
                            st.session_state.selected_flights = [
                                f for f in st.session_state.selected_flights
                                if f.get('_ui_index') != idx
                            ]

                    with col2:
                        # Show flight metrics as requested
                        dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                        arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                        dept_time_display = dept_dt.strftime("%I:%M %p")
                        arr_time_display = arr_dt.strftime("%I:%M %p")
                        dept_date_display = dept_dt.strftime("%a, %b %d")  # e.g., "Fri, Jan 5"

                        # Format duration as "X hr Y min"
                        duration_hours = flight['duration_min'] // 60
                        duration_mins = flight['duration_min'] % 60
                        duration_display = f"{duration_hours} hr {duration_mins} min" if duration_hours > 0 else f"{duration_mins} min"

                        # Get full airline name
                        airline_name = get_airline_name(flight['airline'])

                        # Check if this flight is a codeshare
                        codeshare_label = ""
                        if outbound_codeshare_map.get(idx, False):
                            codeshare_label = '<span style="font-size: 0.75em; color: #666; font-style: italic;"> (Codeshare)</span>'

                        # Format stops for display
                        stops_text = "Direct" if flight['stops'] == 0 else f"{flight['stops']} stop{'s' if flight['stops'] > 1 else ''}"

                        # Add neon glow to first flight only
                        neon_class = "metric-neon" if idx == 0 else ""

                        st.markdown(f"""
                        <div style="line-height: 1.4; margin: 0; padding: 0.4rem 0; border-bottom: 1px solid #eee;">
                        <div style="font-size: 1.1em; margin-bottom: 0.2rem;">
                            <span class="{neon_class}" style="font-weight: 700;">{format_price(flight['price'])}</span> •
                            <span class="{neon_class}" style="font-weight: 600;">{duration_display}</span> •
                            <span class="{neon_class}" style="font-weight: 500;">{stops_text}</span> •
                            <span class="{neon_class}" style="font-weight: 500;">{dept_time_display} - {arr_time_display}</span>
                        </div>
                        <div style="font-size: 0.9em; color: #666;">
                            <span class="{neon_class}">{airline_name} {flight['flight_number']}{codeshare_label}</span> | <span class="{neon_class}">{flight['origin']} → {flight['destination']}</span> | <span class="{neon_class}">{dept_date_display}</span>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)

            with col_ranking:
                st.markdown("#### 📋 Your Top 5 (Drag to Rank)")
                st.markdown(f"**{len(st.session_state.selected_flights)}/5 selected**")

                if st.session_state.selected_flights:
                    # Enforce 5-flight limit
                    if len(st.session_state.selected_flights) > 5:
                        st.session_state.selected_flights = st.session_state.selected_flights[:5]
                        st.rerun()

                    # Create draggable items with X buttons inline
                    draggable_items = []
                    for i, flight in enumerate(st.session_state.selected_flights):
                        airline_name = get_airline_name(flight['airline'])

                        # Format flight data same way as main list
                        from datetime import datetime
                        dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                        arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                        dept_time_display = dept_dt.strftime("%I:%M %p")
                        arr_time_display = arr_dt.strftime("%I:%M %p")
                        dept_date_display = dept_dt.strftime("%a, %b %d")

                        duration_hours = flight['duration_min'] // 60
                        duration_mins = flight['duration_min'] % 60
                        duration_display = f"{duration_hours}h {duration_mins}m" if duration_hours > 0 else f"{duration_mins}m"

                        stops = int(flight.get('stops', 0))
                        stops_text = "Nonstop" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"

                        # Create draggable item string
                        item = f"""#{i+1}: {format_price(flight['price'])} • {duration_display} • {stops_text}
{dept_time_display} - {arr_time_display}
{airline_name} {flight['flight_number']}
{flight['origin']} → {flight['destination']} | {dept_date_display}"""
                        draggable_items.append(item)

                    # Sortable list
                    sorted_items = sort_items(
                        draggable_items,
                        multi_containers=False,
                        direction='vertical',
                        key=f'single_sort_v{st.session_state.single_sort_version}_n{len(st.session_state.selected_flights)}'
                    )

                    # Update order if dragged
                    if sorted_items and sorted_items != draggable_items and len(sorted_items) == len(draggable_items):
                        new_order = []
                        for sorted_item in sorted_items:
                            rank = int(sorted_item.split(':')[0].replace('#', '')) - 1
                            if rank < len(st.session_state.selected_flights):
                                new_order.append(st.session_state.selected_flights[rank])
                        if len(new_order) == len(st.session_state.selected_flights):
                            st.session_state.selected_flights = new_order
                            st.session_state.single_sort_version += 1
                            st.rerun()

                    # X buttons below
                    st.markdown("---")
                    cols = st.columns(5)
                    for i, flight in enumerate(st.session_state.selected_flights):
                        with cols[i]:
                            if st.button("✖", key=f"remove_single_{i}_{flight['id']}", help=f"Remove #{i+1}"):
                                flight_unique_key = f"{flight['id']}_{flight['departure_time']}"

                                # Remove the flight from selected list
                                st.session_state.selected_flights = [
                                    f for f in st.session_state.selected_flights
                                    if f"{f['id']}_{f['departure_time']}" != flight_unique_key
                                ]

                                # Increment version to force checkbox recreation
                                st.session_state.checkbox_version += 1

                                # Clear the checkbox widget state
                                checkbox_key = f"chk_{flight_unique_key}".replace(':', '').replace('-', '').replace('+', '')
                                if checkbox_key in st.session_state:
                                    del st.session_state[checkbox_key]

                                st.rerun()

                    # Submit button
                    if len(st.session_state.selected_flights) == 5 and not st.session_state.outbound_submitted:
                        if st.button("✅ Submit Rankings", key="submit_single", type="primary", use_container_width=True):
                            print(f"[DEBUG] Submit button clicked - Preparing for review")
                            print(f"[DEBUG] Session ID: {st.session_state.session_id}")
                            print(f"[DEBUG] Selected flights: {len(st.session_state.selected_flights)}")

                            # Generate CSV
                            csv_data = generate_flight_csv(
                                st.session_state.all_flights,
                                st.session_state.selected_flights,
                                k=5
                            )
                            print(f"[DEBUG] CSV generated: {len(csv_data)} bytes")

                            # Set flags to trigger review section (don't save to DB yet)
                            st.session_state.csv_data_outbound = csv_data
                            st.session_state.outbound_submitted = True
                            st.session_state.csv_generated = True
                            print(f"[DEBUG] Flags set - will show review section")
                            st.rerun()
                    elif st.session_state.outbound_submitted:
                        st.success("✅ Rankings submitted")
                    else:
                        st.info(f"Select {5 - len(st.session_state.selected_flights)} more flights")
                else:
                    st.info("Check boxes on the left to select flights")

#             # Subway-line navigation on the left side (simplified for one-way)
#             nav_items = ['How to Use', 'Outbound']
#             nav_ids = ['how-to-use', 'outbound-flights']
# 
#             st.markdown(f"""
#                 <style>
#                     .subway-nav {{
#                         position: fixed;
#                         left: 30px;
#                         top: 50%;
#                         transform: translateY(-50%);
#                         z-index: 1000;
#                         padding: 0;
#                         transition: left 0.3s ease;
#                     }}
#                     /* Adjust position when sidebar is open */
#                     [data-testid="stSidebar"]:not([aria-hidden="true"]) ~ div .subway-nav {{
#                         left: 340px;  /* 280px sidebar + 30px margin + 30px spacing = 340px */
#                     }}
#                     .subway-nav ul {{
#                         list-style: none;
#                         padding: 0;
#                         margin: 0;
#                         position: relative;
#                     }}
#                     /* Vertical line connecting stations */
#                     .subway-nav ul::before {{
#                         content: '';
#                         position: absolute;
#                         left: 12px;
#                         top: 20px;
#                         bottom: 20px;
#                         width: 2px;
#                         background-color: rgba(150, 150, 150, 0.3);
#                         z-index: 0;
#                     }}
#                     .subway-nav li {{
#                         position: relative;
#                         margin: 40px 0;
#                     }}
#                     .subway-nav li:first-child {{
#                         margin-top: 0;
#                     }}
#                     .subway-nav li:last-child {{
#                         margin-bottom: 0;
#                     }}
#                     /* Station circles */
#                     .subway-nav a {{
#                         display: flex;
#                         align-items: center;
#                         text-decoration: none;
#                         position: relative;
#                         z-index: 1;
#                     }}
#                     .subway-nav a .station-circle {{
#                         width: 20px;
#                         height: 20px;
#                         border-radius: 50%;
#                         background-color: rgba(255, 255, 255, 0.7);
#                         border: 4px solid #FF6B35;
#                         position: relative;
#                         transition: all 0.3s ease;
#                         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
#                         flex-shrink: 0;
#                     }}
#                     .subway-nav a:hover .station-circle {{
#                         background-color: #FF6B35;
#                         border-color: #FF6B35;
#                         box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.2), 0 2px 6px rgba(0, 0, 0, 0.25);
#                         transform: scale(1.15);
#                     }}
#                     .subway-nav a.active .station-circle {{
#                         background-color: #FF6B35;
#                         border-color: #FF6B35;
#                         box-shadow: 0 0 0 3px rgba(255, 107, 53, 0.3), 0 2px 6px rgba(0, 0, 0, 0.25);
#                         transform: scale(1.2);
#                     }}
#                     /* Station labels */
#                     .subway-nav a .station-label {{
#                         position: absolute;
#                         left: 35px;
#                         white-space: nowrap;
#                         background-color: rgba(0, 0, 0, 0.85);
#                         color: white;
#                         padding: 6px 12px;
#                         border-radius: 6px;
#                         font-size: 13px;
#                         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
#                         font-weight: 500;
#                         opacity: 0;
#                         pointer-events: none;
#                         transition: opacity 0.2s ease;
#                         box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
#                     }}
#                     .subway-nav a:hover .station-label {{
#                         opacity: 1;
#                     }}
#                     .subway-nav a.active .station-label {{
#                         opacity: 1;
#                         background-color: rgba(255, 107, 53, 0.95);
#                     }}
#                 </style>
#                 <div class="subway-nav">
#                     <ul>
#                         {''.join(f'<li><a href="#{nav_ids[i]}"><div class="station-circle"></div><div class="station-label">{nav_items[i]}</div></a></li>' for i in range(len(nav_items)))}
#                     </ul>
#                 </div>
#                 <script>
#                     // Update active state based on scroll position
#                     function updateSubwayNav() {{
#                         const sections = {nav_ids};
#                         const navLinks = document.querySelectorAll('.subway-nav a');
# 
#                         let currentSection = '';
#                         sections.forEach((sectionId, index) => {{
#                             const section = document.getElementById(sectionId);
#                             if (section) {{
#                                 const rect = section.getBoundingClientRect();
#                                 if (rect.top <= window.innerHeight / 3) {{
#                                     currentSection = sectionId;
#                                 }}
#                             }}
#                         }});
# 
#                         navLinks.forEach(link => {{
#                             link.classList.remove('active');
#                             if (link.getAttribute('href') === '#' + currentSection) {{
#                                 link.classList.add('active');
#                             }}
#                         }});
#                     }}
# 
#                     window.addEventListener('scroll', updateSubwayNav);
#                     setTimeout(updateSubwayNav, 200);
#                 </script>
#             """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Built for flight ranking research • Data collected for algorithm evaluation • Contact: listen.cornell@gmail.com")
