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
from components.interactive_demo import init_demo_mode, start_demo
from components.static_demo_page import render_static_demo_page

load_dotenv()

# Helper function to format price display
def format_price(price):
    """Format price for display, showing 'N/A' for 0 or None."""
    if price is None or price == 0:
        return "N/A"
    return f"${price:.0f}"

# LILO Chat UI Helper Functions
def render_chat_message(message, is_bot=True):
    """Render a single chat message with bot/user styling."""
    import html
    # Escape HTML to prevent rendering issues
    safe_message = html.escape(message)

    if is_bot:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="background: #f0f2f6; padding: 12px 16px; border-radius: 18px; max-width: 70%;">
                🤖 {safe_message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
            <div style="background: #0068c9; color: white; padding: 12px 16px; border-radius: 18px; max-width: 70%;">
                {safe_message}
            </div>
        </div>
        """, unsafe_allow_html=True)

def get_feature_color(value, min_val, max_val, reverse=False):
    """Get color for feature comparison (green=better, red=worse)."""
    if max_val == min_val:
        return "#808080"  # Gray if same

    # Normalize to 0-1
    normalized = (value - min_val) / (max_val - min_val)

    # Reverse for things where lower is better (price, duration)
    if reverse:
        normalized = 1 - normalized

    # Interpolate between red and green
    if normalized < 0.5:
        # Red to yellow
        r, g, b = 255, int(255 * (normalized * 2)), 0
    else:
        # Yellow to green
        r, g, b = int(255 * (1 - (normalized - 0.5) * 2)), 255, 0

    return f"rgb({r},{g},{b})"

def render_flight_comparison(flight_a, flight_b, label_a="Flight A", label_b="Flight B"):
    """Render two flights side-by-side with color-coded features."""
    col1, col2 = st.columns(2)

    # Extract features for comparison
    features = {
        'Price': (flight_a.get('price', 0), flight_b.get('price', 0), True),  # True = lower is better
        'Duration': (flight_a.get('duration_min', 0), flight_b.get('duration_min', 0), True),
        'Stops': (flight_a.get('stops', 0), flight_b.get('stops', 0), True),
    }

    with col1:
        st.markdown(f"### {label_a}")
        st.markdown(f"**{flight_a.get('airline', 'Unknown')}** {flight_a.get('flight_number', '')}")
        st.markdown(f"{flight_a.get('origin', '')} → {flight_a.get('destination', '')}")
        st.markdown(f"**Departs:** {flight_a.get('departure_time', 'N/A')}")
        st.markdown(f"**Arrives:** {flight_a.get('arrival_time', 'N/A')}")

        st.markdown("---")
        for feature_name, (val_a, val_b, reverse) in features.items():
            color = get_feature_color(val_a, min(val_a, val_b), max(val_a, val_b), reverse)
            if feature_name == 'Price':
                display_val = f"${val_a:.0f}"
            elif feature_name == 'Duration':
                display_val = f"{val_a//60}h {val_a%60}m"
            else:
                display_val = str(val_a)

            st.markdown(f"""
            <div style="background: {color}; padding: 8px; border-radius: 5px; margin: 5px 0;">
                <strong>{feature_name}:</strong> {display_val}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"### {label_b}")
        st.markdown(f"**{flight_b.get('airline', 'Unknown')}** {flight_b.get('flight_number', '')}")
        st.markdown(f"{flight_b.get('origin', '')} → {flight_b.get('destination', '')}")
        st.markdown(f"**Departs:** {flight_b.get('departure_time', 'N/A')}")
        st.markdown(f"**Arrives:** {flight_b.get('arrival_time', 'N/A')}")

        st.markdown("---")
        for feature_name, (val_a, val_b, reverse) in features.items():
            color = get_feature_color(val_b, min(val_a, val_b), max(val_a, val_b), reverse)
            if feature_name == 'Price':
                display_val = f"${val_b:.0f}"
            elif feature_name == 'Duration':
                display_val = f"{val_b//60}h {val_b%60}m"
            else:
                display_val = str(val_b)

            st.markdown(f"""
            <div style="background: {color}; padding: 8px; border-radius: 5px; margin: 5px 0;">
                <strong>{feature_name}:</strong> {display_val}
            </div>
            """, unsafe_allow_html=True)

# Initialize database (create tables if they don't exist)
try:
    from backend.db import init_db
    init_db()
except Exception as e:
    # Don't crash the app if database setup fails, just log it
    print(f"Database initialization: {str(e)}")

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
query_params = st.experimental_get_query_params()
token_from_url = query_params.get('id', [None])[0]
if token_from_url:
    # Validate token (checks database to see if it's been used)
    from backend.db import validate_token
    token_status = validate_token(token_from_url)

    st.session_state.token = token_from_url
    st.session_state.token_valid = token_status['valid']
    st.session_state.token_message = token_status['message']

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
if 'lilo_completed' not in st.session_state:
    st.session_state.lilo_completed = False
if 'lilo_round' not in st.session_state:
    st.session_state.lilo_round = 0  # 0 = round 1, 1 = round 2
if 'lilo_optimizer' not in st.session_state:
    st.session_state.lilo_optimizer = None
if 'lilo_round1_flights' not in st.session_state:
    st.session_state.lilo_round1_flights = []
if 'lilo_round2_flights' not in st.session_state:
    st.session_state.lilo_round2_flights = []
if 'lilo_round1_selected' not in st.session_state:
    st.session_state.lilo_round1_selected = []
if 'lilo_round2_selected' not in st.session_state:
    st.session_state.lilo_round2_selected = []
if 'lilo_questions' not in st.session_state:
    st.session_state.lilo_questions = []
if 'lilo_answers' not in st.session_state:
    st.session_state.lilo_answers = {}
if 'cross_validation_completed' not in st.session_state:
    st.session_state.cross_validation_completed = False
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

# TEMPORARY COMMENT: Old session state for algorithm-based ranking
# if 'interleaved_results' not in st.session_state:
#     st.session_state.interleaved_results = []
# if 'shortlist' not in st.session_state:
#     st.session_state.shortlist = []

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
# Special handling for DEMO token
if st.session_state.token and st.session_state.token.upper() == "DEMO":
    # Bypass validation for DEMO token
    st.session_state.token_valid = True
    st.info(f"🎯 **Demo Mode** - Explore the flight search tool freely!")
elif not st.session_state.token:
    st.error("❌ **Access Denied: No Token Provided**")
    st.warning("This study requires a unique access token. Please use the link provided to you by the researchers, or use `?id=DEMO` for demo mode.")
    st.stop()
elif not st.session_state.token_valid:
    st.error(f"❌ **Access Denied: {st.session_state.token_message}**")
    if 'already used' in st.session_state.token_message.lower():
        st.warning("This token has already been used. To request a new token for an additional session, please reach out to the research team.")
    else:
        st.warning("Please check your access link and try again, or contact the researchers if you believe this is an error.")
    st.stop()
else:
    # Show success message for valid token
    st.success(f"✅ Access granted! Token: {st.session_state.token}")

# Admin button at top right - plain text
st.markdown("""
<style>
.admin-btn-fixed {
    position: fixed !important;
    top: 10px !important;
    right: 20px !important;
    z-index: 9999 !important;
}
</style>
<div class="admin-btn-fixed">
""", unsafe_allow_html=True)

if st.button("Admin", type="secondary", key="admin_top_btn"):
    # Inline admin page instead of separate page
    st.session_state.show_admin = True
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Show inline admin if requested
if st.session_state.get('show_admin', False):
    st.markdown("### 🔧 Admin Panel - Complete Research Data")

    # Import admin utilities
    from admin_utils import get_all_sessions_summary, get_complete_session_detail, export_session_csv

    # Back button
    if st.button("← Back to Main"):
        st.session_state.show_admin = False
        st.session_state.selected_session_token = None
        st.rerun()

    st.markdown("---")

    # Check if viewing specific session detail
    if st.session_state.get('selected_session_token'):
        token = st.session_state.selected_session_token

        # Back to list button
        if st.button("← Back to Sessions List"):
            st.session_state.selected_session_token = None
            st.rerun()

        st.markdown("---")

        # Get detailed session data
        with st.spinner("Loading complete session data..."):
            detail = get_complete_session_detail(token)

        if not detail:
            st.error("❌ Session not found")
        else:
            # Session header
            status_badge = "✅ Completed" if detail['is_completed'] else "⏳ In Progress"
            st.markdown(f"## Session: `{detail['completion_token']}` {status_badge}")
            st.caption(f"Session ID: {detail['session_id']}")
            date_label = "Completed" if detail['is_completed'] else "Started"
            st.caption(f"{date_label}: {detail['completed_at'].strftime('%Y-%m-%d %H:%M:%S')}")

            st.markdown("---")

            # 1. FLIGHT SEARCH DATA
            if detail['search']:
                st.markdown("### ✈️ Flight Search")
                search = detail['search']

                # Extract origin/destination from JSON fields
                origin_str = str(search['origin'][0]) if search['origin'] and isinstance(search['origin'], list) and len(search['origin']) > 0 else str(search['origin']) if search['origin'] else 'N/A'
                dest_str = str(search['destination'][0]) if search['destination'] and isinstance(search['destination'], list) and len(search['destination']) > 0 else str(search['destination']) if search['destination'] else 'N/A'

                st.metric("Route", f"{origin_str} → {dest_str}")

                with st.expander("Search Details"):
                    st.markdown(f"**User Prompt:** {search['prompt']}")
                    st.text(f"Search ID: {search['search_id']}")
                    st.text(f"Departure: {search.get('departure_date', 'N/A')}")
                    st.text(f"Searched at: {search['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")

                st.markdown("---")

            # 2. USER RANKINGS
            if detail['user_rankings']:
                st.markdown(f"### 🎯 User Manual Rankings ({len(detail['user_rankings'])} flights)")
                with st.expander("View All User Rankings"):
                    for ranking in detail['user_rankings'][:10]:
                        st.text(
                            f"Rank #{ranking['rank']}: Flight ID {ranking['flight_id']} | "
                            f"Ranked at: {ranking['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    if len(detail['user_rankings']) > 10:
                        st.caption(f"... and {len(detail['user_rankings']) - 10} more")

                st.markdown("---")

            # 3. CROSS-VALIDATION
            if detail['cross_validation']:
                st.markdown(f"### ✅ Cross-Validation ({len(detail['cross_validation'])} reviews)")
                cv_total = len(detail['cross_validation'])

                st.metric("Total Cross-Validation Reviews", cv_total)

                with st.expander("View Cross-Validation Details"):
                    for i, cv in enumerate(detail['cross_validation'], 1):
                        st.markdown(f"**Review #{i}**")
                        st.text(f"Reviewed Session: {cv['reviewed_session_id'][:30]}...")
                        st.text(f"Reviewed Prompt: {cv['reviewed_prompt'][:80]}...")
                        st.text(f"Selected {len(cv.get('selected_flight_ids', []))} flights")
                        st.text(f"Reviewed at: {cv['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.markdown("---")

                st.markdown("---")

            # 4. SURVEY RESPONSE
            if detail['survey']:
                st.markdown("### 📝 Survey Response")
                survey = detail['survey']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Satisfaction", f"{survey['satisfaction']}/5")
                with col2:
                    st.metric("Ease of Use", f"{survey['ease_of_use']}/5")
                with col3:
                    st.metric("Would Use Again", survey['would_use_again'])

                with st.expander("Full Survey Responses"):
                    st.markdown(f"**Search Method:** {survey['search_method']}")
                    st.markdown(f"**Understood Ranking:** {survey['understood_ranking']}/5")
                    st.markdown(f"**Flights Matched Expectations:** {survey['flights_matched']}/5")
                    st.markdown(f"**Compared to Other Tools:** {survey['compared_to_others']}/5")
                    if survey['helpful_features']:
                        st.markdown(f"**Helpful Features:** {', '.join(survey['helpful_features'])}")
                    if survey['encountered_issues'] == 'Yes' and survey['issues_description']:
                        st.markdown(f"**Issues:** {survey['issues_description']}")
                    if survey['confusing_frustrating']:
                        st.markdown(f"**Confusing/Frustrating:** {survey['confusing_frustrating']}")
                    if survey['missing_features']:
                        st.markdown(f"**Missing Features:** {survey['missing_features']}")
                    if survey['would_use_again_reason']:
                        st.markdown(f"**Reason:** {survey['would_use_again_reason']}")
                    if survey['additional_comments']:
                        st.markdown(f"**Additional Comments:** {survey['additional_comments']}")

                st.markdown("---")

            # 5. LILO DATA
            if detail['lilo']:
                lilo = detail['lilo']
                st.markdown("### 🤖 LILO (Language-Informed Latent Optimization)")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Iterations", lilo['num_iterations'])
                with col2:
                    st.metric("Questions/Round", lilo['questions_per_round'])
                with col3:
                    status = "✅ Complete" if lilo['completed_at'] else "⏳ In Progress"
                    st.metric("Status", status)

                # Utility Statistics
                if lilo['rankings']['utility_stats']:
                    st.markdown("#### 📊 Learned Utility Function")
                    stats = lilo['rankings']['utility_stats']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Max", f"{stats['max']:.4f}")
                    with col2:
                        st.metric("Min", f"{stats['min']:.4f}")
                    with col3:
                        st.metric("Avg", f"{stats['avg']:.4f}")
                    with col4:
                        st.metric("Range", f"{stats['range']:.4f}")

                # Chat Transcript
                if lilo['chat_transcript']:
                    st.markdown("#### 💬 LILO Chat Transcript")
                    for round_num in sorted(lilo['chat_transcript'].keys()):
                        with st.expander(f"Round {round_num} ({len(lilo['chat_transcript'][round_num])} messages)"):
                            for msg in lilo['chat_transcript'][round_num]:
                                speaker = "🤖" if msg['is_bot'] else "👤"
                                st.markdown(f"{speaker} {msg['text']}")

                # Top Rankings
                if lilo['rankings']['top_10']:
                    st.markdown(f"#### 🏆 Top 10 Flights by LILO Utility ({lilo['rankings']['total']} total)")
                    import pandas as pd
                    top_data = []
                    for r in lilo['rankings']['top_10']:
                        flight = r['flight_data']
                        top_data.append({
                            'Rank': r['rank'],
                            'Utility': f"{r['utility_score']:.4f}",
                            'Price': flight.get('price'),
                            'Duration': flight.get('duration_min'),
                            'Stops': flight.get('stops'),
                            'Airline': flight.get('airline', 'N/A')[:30]
                        })
                    st.dataframe(pd.DataFrame(top_data), use_container_width=True, hide_index=True)

            # Export buttons
            st.markdown("---")
            st.markdown("### 📥 Export Data")

            from admin_utils import (
                export_manual_rankings_csv,
                export_cross_validation_csv,
                export_survey_csv,
                export_lilo_full_csv
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if detail.get('user_rankings'):
                    csv_data = export_manual_rankings_csv(token)
                    if csv_data:
                        st.download_button(
                            label="📊 Manual Rankings",
                            data=csv_data,
                            file_name=f"manual_rankings_{detail['completion_token']}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.button("📊 Manual Rankings", disabled=True, use_container_width=True)

            with col2:
                if detail.get('cross_validation'):
                    csv_data = export_cross_validation_csv(token)
                    if csv_data:
                        st.download_button(
                            label="✅ Cross-Validation",
                            data=csv_data,
                            file_name=f"cross_validation_{detail['completion_token']}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.button("✅ Cross-Validation", disabled=True, use_container_width=True)

            with col3:
                if detail.get('survey'):
                    csv_data = export_survey_csv(token)
                    if csv_data:
                        st.download_button(
                            label="📝 Survey",
                            data=csv_data,
                            file_name=f"survey_{detail['completion_token']}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.button("📝 Survey", disabled=True, use_container_width=True)

            with col4:
                if detail.get('lilo'):
                    csv_data = export_lilo_full_csv(token)
                    if csv_data:
                        st.download_button(
                            label="🤖 LILO Full Data",
                            data=csv_data,
                            file_name=f"lilo_full_{detail['completion_token']}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.button("🤖 LILO Full Data", disabled=True, use_container_width=True)
                else:
                    st.button("🤖 LILO Full Data", disabled=True, use_container_width=True)

    else:
        # Show list of all sessions
        st.markdown("## All Research Sessions")

        with st.spinner("Loading sessions..."):
            sessions = get_all_sessions_summary()

        if not sessions:
            st.info("No sessions found in database.")
        else:
            st.markdown(f"**Total Sessions:** {len(sessions)}")
            st.markdown("---")

            # Display each session as a card
            for sess in sessions:
                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        # Session header
                        st.markdown(f"### 📊 Token: `{sess['completion_token']}`")
                        st.caption(f"Completed: {sess['completed_at'].strftime('%Y-%m-%d %H:%M')}")

                        # Show what data exists
                        data_indicators = []
                        if sess['has_search']:
                            # Extract origin/destination from JSON fields
                            origin_str = str(sess['origin'][0]) if sess['origin'] and isinstance(sess['origin'], list) and len(sess['origin']) > 0 else str(sess['origin'])[:20] if sess['origin'] else 'N/A'
                            dest_str = str(sess['destination'][0]) if sess['destination'] and isinstance(sess['destination'], list) and len(sess['destination']) > 0 else str(sess['destination'])[:20] if sess['destination'] else 'N/A'
                            data_indicators.append(f"✈️ Search ({origin_str} → {dest_str})")
                        if sess['has_survey']:
                            data_indicators.append(f"📝 Survey (satisfaction: {sess['survey_satisfaction']}/5)")
                        if sess['has_cv']:
                            data_indicators.append(f"✅ Cross-Val ({sess['cv_count']} reviews)")
                        if sess['has_lilo']:
                            lilo_status = "✅" if sess['lilo_completed'] else "⏳"
                            data_indicators.append(f"{lilo_status} LILO ({sess['lilo_rankings']} flights ranked)")

                        for indicator in data_indicators:
                            st.text(indicator)

                        if sess['search_prompt']:
                            st.caption(f"Prompt: {sess['search_prompt'][:80]}...")

                    with col2:
                        # View details button - use session_id for in-progress, token for completed
                        identifier = sess['completion_token'] if sess['is_completed'] else sess['session_id']
                        if st.button("View Details", key=f"view_{sess['session_id']}"):
                            st.session_state.selected_session_token = identifier
                            st.rerun()

                    st.markdown("---")

    st.stop()

# Initialize interactive demo/tutorial mode
init_demo_mode()

# Show static demo page if active (completely separate from real app)
if st.session_state.get('demo_active', False):
    # Check for tutorial navigation via query params
    try:
        query_params = st.experimental_get_query_params()
        action = query_params.get('tutorial_action', [None])[0]
        if action:
            if action == 'next':
                if st.session_state.demo_step < 6:
                    st.session_state.demo_step += 1
                else:
                    st.session_state.demo_active = False
                    st.session_state.demo_step = 0
            elif action == 'back' and st.session_state.demo_step > 0:
                st.session_state.demo_step -= 1
            elif action == 'exit':
                st.session_state.demo_active = False
                st.session_state.demo_step = 0
            # Clear query param
            st.experimental_set_query_params()
            st.rerun()
    except:
        pass

    # OLD TUTORIAL CODE - Commented out in favor of new guided tutorial
    # render_static_demo_page(st.session_state.demo_step)
    # from components.tutorial_card import show_tutorial_card
    # show_tutorial_card(st.session_state.demo_step)
    # st.stop()

    # NEW GUIDED TUTORIAL - COMMENTED OUT PER USER REQUEST
    # from components.guided_tutorial import show_guided_tutorial
    #
    # # Pre-populate demo state with sample flight data
    # if 'demo_initialized' not in st.session_state:
    #     st.session_state.demo_initialized = True
    #     # Set up a populated state (as if user already searched)
    #     st.session_state.search_prompt = "I want to fly from JFK to LAX on December 26th. I prefer cheap flights but if a flight is longer than 12 hours I'd prefer to pay a bit more."
    #     st.session_state.search_complete = True
    #     # Add more demo state initialization as needed
    #
    # # Show the guided tutorial overlay
    # show_guided_tutorial(st.session_state.demo_step)
    #
    # # Continue rendering the real app below (but with interactions disabled in demo mode)
    pass

# # Interactive Demo/Tutorial Mode (COMMENTED OUT - REPLACED WITH NEW TOUR ABOVE)
# if 'demo_mode' not in st.session_state:
#     st.session_state.demo_mode = False
# if 'demo_step' not in st.session_state:
#     st.session_state.demo_step = 0
#
# st.markdown("---")
# if not st.session_state.demo_mode:
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         if st.button("🎓 Start Interactive Tutorial", use_container_width=True, type="secondary"):
#             st.session_state.demo_mode = True
#             st.session_state.demo_step = 0
#             st.rerun()
# else:
#     # Demo mode active - show slideshow
#     demo_steps = [
#         {
#             "title": "Step 1: Write Your Prompt",
#             "description": "Describe your ideal flight in natural language. Include your route, dates, and preferences.",
#             "example": "I need to fly from NYC to LAX on December 25th. I prefer nonstop flights and want the cheapest option under $400.",
#             "visual": "📝"
#         },
#         {
#             "title": "Step 2: Search for Flights",
#             "description": "Click 'Search Flights' to see available options. You can filter results by price, airlines, stops, and times.",
#             "example": "The system will show you all matching flights with detailed information about price, duration, and stops.",
#             "visual": "🔍"
#         },
#         {
#             "title": "Step 3: Select Top 5 Flights",
#             "description": "Check the boxes next to your 5 favorite flights that best match your needs.",
#             "example": "Review each flight's details (price, duration, stops) and select your top choices.",
#             "visual": "☑️"
#         },
#         {
#             "title": "Step 4: Rank Your Selections",
#             "description": "Drag and drop your selected flights to rank them from most to least preferred.",
#             "example": "Your #1 choice should be at the top, and your #5 choice at the bottom.",
#             "visual": "🎯"
#         },
#         {
#             "title": "Step 5: Complete Survey",
#             "description": "After submitting your rankings, complete a brief survey about your experience.",
#             "example": "Share your feedback to help us improve the flight search system.",
#             "visual": "📋"
#         }
#     ]
#
#     step = demo_steps[st.session_state.demo_step]
#
#     st.markdown(f"""
#     <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                 padding: 30px; border-radius: 10px; color: white; text-align: center;">
#         <div style="font-size: 60px; margin-bottom: 15px;">{step['visual']}</div>
#         <h2 style="color: white; margin: 0;">{step['title']}</h2>
#         <p style="font-size: 18px; margin-top: 10px; opacity: 0.9;">Tutorial Step {st.session_state.demo_step + 1} of {len(demo_steps)}</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     st.markdown(f"### {step['description']}")
#     st.info(f"**Example:** {step['example']}")
#
#     # Navigation buttons
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col1:
#         if st.session_state.demo_step > 0:
#             if st.button("⬅️ Previous", use_container_width=True):
#                 st.session_state.demo_step -= 1
#                 st.rerun()
#     with col2:
#         if st.button("❌ Exit Tutorial", use_container_width=True):
#             st.session_state.demo_mode = False
#             st.rerun()
#     with col3:
#         if st.session_state.demo_step < len(demo_steps) - 1:
#             if st.button("Next ➡️", use_container_width=True, type="primary"):
#                 st.session_state.demo_step += 1
#                 st.rerun()
#         else:
#             if st.button("✅ Finish Tutorial", use_container_width=True, type="primary"):
#                 st.session_state.demo_mode = False
#                 st.success("Tutorial complete! You're ready to search for flights.")
#                 st.rerun()
#
#     st.markdown("---")

# How to Use section
st.markdown('<div id="how-to-use"></div>', unsafe_allow_html=True)
st.markdown("### 📖 How to Use")

# Container to hide content on LILO/survey pages
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

# Always show animated placeholder with carousel controls
placeholder_html = r"""
<style>
    body {
        margin: 0;
        padding: 0;
        background: white;
    }
    .carousel-container {
        position: relative;
    }
    #animBox {
        border: 1px solid rgb(204, 204, 204);
        border-radius: 0.5rem;
        background: white;
        height: 150px;
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
        transition: opacity 0.5s ease-out;
        pointer-events: none;
    }
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
</style>
<div class="carousel-container">
    <div id="animBox">
        <button class="carousel-nav carousel-prev" onclick="prevPrompt()">‹</button>
        <button class="carousel-nav carousel-next" onclick="nextPrompt()">›</button>
        <div id="animPlaceholder"></div>
        <div class="carousel-indicator">
            <span class="indicator-dot active" onclick="goToPrompt(0)"></span>
            <span class="indicator-dot" onclick="goToPrompt(1)"></span>
        </div>
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
    let autoPlay = true;
    let typingTimeout = null;

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
            typing = true;
            placeholder.style.opacity = '1';
            updateIndicators();
            type();
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
            typing = true;
            placeholder.style.opacity = '1';
            updateIndicators();
            type();
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
            typing = true;
            placeholder.style.opacity = '1';
            updateIndicators();
            type();
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

# Search buttons
st.markdown('<div id="demo-search-btn">', unsafe_allow_html=True)
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    regular_search = st.button("🔍 Search Flights", type="primary", use_container_width=True)

with col_btn2:
    ai_search = st.button("🔎 Search Flights with AI Personalization", type="secondary", use_container_width=True)

# Custom CSS and JavaScript for black AI button
st.markdown("""
<style>
/* Make the AI search button black with white text */
button.ai-search-black {
    background-color: #000000 !important;
    color: white !important;
    border: 1px solid #333333 !important;
}
button.ai-search-black:hover {
    background-color: #1a1a1a !important;
    border: 1px solid #4a4a4a !important;
}
</style>
<script>
// Add class to AI search button (secondary button containing AI Personalization text)
setTimeout(function() {
    const buttons = document.querySelectorAll('button[kind="secondary"]');
    buttons.forEach(function(btn) {
        if (btn.textContent.includes('AI Personalization')) {
            btn.classList.add('ai-search-black');
        }
    });
}, 100);
</script>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # Close demo-search-btn

# Close hideable-survey-content container
st.markdown('</div>', unsafe_allow_html=True)

# Rate limiting for AI search (prevent quota exhaustion)
import time
if 'last_ai_search_time' not in st.session_state:
    st.session_state.last_ai_search_time = 0

if ai_search:
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_ai_search_time
    cooldown_seconds = 45  # 45 second cooldown between AI searches (allows ~5 iterations at 6s each + buffer)

    if time_since_last < cooldown_seconds:
        remaining = int(cooldown_seconds - time_since_last)
        st.error(f"⏳ Please wait {remaining} seconds before searching again (Gemini API rate limit protection)")
        st.stop()

    st.session_state.last_ai_search_time = current_time

if ai_search or regular_search:
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

    # COMMENTED OUT - No longer collecting name/email
    # if not st.session_state.user_name or not st.session_state.user_name.strip():
    #     validation_errors.append("Please enter your full name")
    # elif len(st.session_state.user_name.split()) < 2:
    #     validation_errors.append("Please enter both first and last name")
    #
    # if not st.session_state.user_email or not st.session_state.user_email.strip():
    #     validation_errors.append("Please enter your email address")
    # elif '@' not in st.session_state.user_email or '.' not in st.session_state.user_email:
    #     validation_errors.append("Please enter a valid email address")

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

        with st.spinner("✨ Parsing your request and searching flights..."):
            try:
                # Store original prompt
                st.session_state.original_prompt = prompt

                # Parse prompt with LLM
                st.info("🤖 Parsing your request with Gemini...")
                parsed = parse_flight_prompt_with_llm(prompt)
                st.session_state.parsed_params = parsed

                # Debug: show parsed results
                with st.expander("🔍 Debug: Parsed Parameters"):
                    st.json(parsed)

                if not parsed.get('origins') or not parsed.get('destinations'):
                    st.error("Could not extract origin and destination. Please specify airports or cities.")
                    st.stop()

                # Show what we understood
                st.success("✅ Understood your request!")

                # Check if return flight is present
                return_dates = parsed.get('return_dates', [])
                # Fallback to single return_date for backward compatibility
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

                # Get departure dates (list)
                departure_dates = parsed.get('departure_dates', [])
                if not departure_dates:
                    st.error("No departure dates found. Please specify when you want to fly.")
                    st.stop()

                for origin_code in parsed['origins']:  # Search all origins (e.g., JFK, EWR, LGA for NYC)
                    for dest_code in parsed['destinations']:  # Search all destinations
                        # Check test API compatibility
                        origin, origin_warning = get_test_api_fallback(origin_code)
                        dest, dest_warning = get_test_api_fallback(dest_code)

                        if origin_warning:
                            st.warning(origin_warning)
                        if dest_warning:
                            st.warning(dest_warning)

                        # Search flights for ALL departure dates
                        for departure_date in departure_dates:
                            st.info(f"Searching: {origin} → {dest} on {departure_date}")

                            # Search outbound flights
                            results = flight_client.search_flights(
                                origin=origin,
                                destination=dest,
                                departure_date=departure_date,
                                adults=1,
                                max_results=250  # Get ALL available flights (increased from 50)
                            )

                            # Debug: show raw results
                            with st.expander(f"🔍 Debug: {provider_name} Response ({origin}→{dest} on {departure_date})"):
                                st.write(f"Type: {type(results)}")
                                if isinstance(results, dict):
                                    st.write(f"Keys: {results.keys()}")
                                    if 'data' in results:
                                        st.write(f"Number of flights: {len(results['data'])}")
                                elif isinstance(results, list):
                                    st.write(f"Number of flights: {len(results)}")
                                st.json(results if isinstance(results, (dict, list)) else str(results))

                            # Parse results using unified client
                            if isinstance(results, list):
                                flight_offers = results
                            elif isinstance(results, dict) and 'data' in results:
                                flight_offers = results['data']
                            else:
                                flight_offers = []

                            for offer in flight_offers:
                                # Use unified client's parse method
                                flight_info = flight_client.parse_flight_offer(offer)
                                if flight_info:
                                    all_flights.append(flight_info)

                        # If return flight requested, search return flights for ALL return dates
                        if has_return:
                            return_dates = parsed.get('return_dates', [])
                            # Fallback to single return_date for backward compatibility
                            if not return_dates and parsed.get('return_date'):
                                return_dates = [parsed.get('return_date')]

                            for return_date in return_dates:
                                st.info(f"✈️ Searching return flights: {dest} → {origin} on {return_date}")

                                return_results = flight_client.search_flights(
                                    origin=dest,  # Swap: destination becomes origin
                                    destination=origin,  # Swap: origin becomes destination
                                    departure_date=return_date,
                                    adults=1,
                                    max_results=250
                                )

                                # Debug: show return flight results
                                with st.expander(f"🔍 Debug: Return Flight Response ({dest}→{origin} on {return_date})"):
                                    st.write(f"Type: {type(return_results)}")
                                    if isinstance(return_results, dict):
                                        st.write(f"Keys: {return_results.keys()}")
                                        if 'data' in return_results:
                                            st.write(f"Number of flights: {len(return_results['data'])}")
                                    elif isinstance(return_results, list):
                                        st.write(f"Number of flights: {len(return_results)}")
                                    st.json(return_results if isinstance(return_results, (dict, list)) else str(return_results))

                                # Parse return flight results
                                if isinstance(return_results, list):
                                    return_flight_offers = return_results
                                elif isinstance(return_results, dict) and 'data' in return_results:
                                    return_flight_offers = return_results['data']
                                else:
                                    return_flight_offers = []

                                for offer in return_flight_offers:
                                    # Use unified client's parse method
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

                # Apply AI ranking if AI search button was pressed
                if ai_search:
                    st.write("---")
                    st.write(f"### 🤖 LISTEN-U AI Personalization")
                    st.write(f"**Debug Info:**")
                    st.write(f"- Outbound flights to rank: {len(all_flights)}")
                    st.write(f"- Your prompt: *{prompt}*")

                    import traceback
                    import time
                    try:
                        from backend.listen_main_wrapper import rank_flights_with_listen_main

                        preferences = parsed.get('preferences', {})
                        preferences['origins'] = parsed.get('origins', [])
                        preferences['destinations'] = parsed.get('destinations', [])

                        # Show progress with simulated progress bar
                        n_iters = 5  # Reduced from 25 to stay within Gemini free tier quota (15 req/min)
                        expected_time = n_iters * 6  # ~6 seconds per iteration with Gemini free tier safe limit (10 req/min)

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Run LISTEN in background while updating progress bar
                        import threading

                        result_container = {'flights': None, 'error': None}

                        def run_listen():
                            try:
                                result_container['flights'] = rank_flights_with_listen_main(
                                    flights=all_flights,
                                    user_prompt=prompt,
                                    user_preferences=preferences,
                                    n_iterations=n_iters
                                )
                            except Exception as e:
                                result_container['error'] = e

                        thread = threading.Thread(target=run_listen)
                        thread.start()

                        start_time = time.time()

                        # Update progress bar while LISTEN runs
                        while thread.is_alive():
                            elapsed = time.time() - start_time
                            progress = min(elapsed / expected_time, 0.99)  # Cap at 99% until actually done
                            estimated_iter = int(progress * n_iters)

                            progress_bar.progress(progress)
                            status_text.info(f"⏳ **LISTEN-U iteration ~{estimated_iter}/{n_iters}** (Learning your preferences...)")
                            time.sleep(0.5)

                        thread.join()

                        # Check for errors
                        if result_container['error']:
                            progress_bar.empty()
                            status_text.empty()
                            raise result_container['error']

                        all_flights = result_container['flights']
                        elapsed = time.time() - start_time

                        progress_bar.progress(1.0)
                        status_text.success(f"✅ **LISTEN-U completed!**")
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()

                        st.success(f"✅ **LISTEN-U completed in {elapsed:.1f} seconds!**")
                        st.write(f"**Results:**")
                        st.write(f"- Ranked {len(all_flights)} flights by learned utility function")
                        st.write(f"- Top flight price: {format_price(all_flights[0]['price'])}")

                        # Rank return flights if present
                        if has_return and all_return_flights:
                            return_progress = st.empty()
                            return_progress.info("⏳ **Ranking return flights with LISTEN-U...**")

                            start_return = time.time()
                            all_return_flights = rank_flights_with_listen_main(
                                flights=all_return_flights,
                                user_prompt=prompt,
                                user_preferences=preferences,
                                n_iterations=5  # Reduced from 25 to stay within Gemini free tier quota
                            )
                            elapsed_return = time.time() - start_return
                            return_progress.empty()

                            st.success(f"✅ **Return flights ranked in {elapsed_return:.1f} seconds!**")
                            st.write(f"- Top return flight price: {format_price(all_return_flights[0]['price'])}")

                        st.write("---")
                    except Exception as e:
                        st.error(f"⚠️ **LISTEN-U AI PERSONALIZATION FAILED**")
                        st.error(f"**Error:** {str(e)}")
                        with st.expander("Show Full Error Details"):
                            st.code(traceback.format_exc())
                        st.error("**CANNOT CONTINUE - LISTEN-U IS REQUIRED FOR AI SEARCH**")
                        st.info("Please use the regular '🔍 Search Flights' button instead, or contact support if this error persists.")
                        st.stop()  # STOP execution - do NOT show flights if LISTEN failed

                # Store all flights
                st.session_state.all_flights = all_flights
                st.session_state.all_return_flights = all_return_flights
                # Store for LILO optimizer
                st.session_state.all_flights_data = all_flights

                # REMOVED: Pre-generation was blocking flight display (15-30+ seconds!)
                # LILO questions will be generated when user reaches LILO section
                # This makes flights appear immediately after Amadeus API responds

                if has_return:
                    st.success(f"✅ Found {len(all_flights)} outbound flights and {len(all_return_flights)} return flights!")
                else:
                    st.success(f"✅ Found {len(all_flights)} flights!")
                # Don't rerun here - it causes infinite loop

                # TEMPORARY COMMENT: Algorithm-based ranking code (old version)
                # preferences = parsed.get('preferences', {})
                # preferences['origins'] = parsed.get('origins', [])
                # preferences['destinations'] = parsed.get('destinations', [])
                #
                # # Algorithm 1: Cheapest
                # cheapest_ranked = sorted(all_flights, key=lambda x: x['price'])[:10]
                #
                # # Algorithm 2: Fastest
                # fastest_ranked = sorted(all_flights, key=lambda x: x['duration_min'])[:10]
                #
                # # Algorithm 3: LISTEN-U (using LISTEN's main.py framework)
                # try:
                #     from backend.listen_main_wrapper import rank_flights_with_listen_main
                #     st.info("🤖 Running LISTEN-U algorithm via main.py (25 iterations, may take 2-3 minutes)...")
                #     listen_u_ranked = rank_flights_with_listen_main(
                #         flights=all_flights,
                #         user_prompt=prompt,
                #         user_preferences=preferences
                #     )
                #     st.success("✅ LISTEN-U complete! Used utility algorithm to rank flights.")
                # except Exception as e:
                #     st.warning(f"⚠️ LISTEN-U failed ({str(e)}), using fallback preference-aware ranking")
                #     def preference_score(flight):
                #         score = 0
                #         if preferences.get('prefer_cheap'):
                #             score += flight['price'] / 1000
                #         if preferences.get('prefer_fast'):
                #             score += flight['duration_min'] / 300
                #         if preferences.get('prefer_direct'):
                #             score += flight['stops'] * 0.5
                #         return score if score > 0 else flight['price'] / 500 + flight['duration_min'] / 300
                #     listen_u_ranked = sorted(all_flights, key=preference_score)[:10]
                #
                # # Interleave results (round-robin)
                # interleaved = []
                # for i in range(10):
                #     if i < len(cheapest_ranked):
                #         interleaved.append({'flight': cheapest_ranked[i], 'algorithm': 'Cheapest', 'rank': i + 1})
                #     if i < len(fastest_ranked):
                #         interleaved.append({'flight': fastest_ranked[i], 'algorithm': 'Fastest', 'rank': i + 1})
                #     if i < len(listen_u_ranked):
                #         interleaved.append({'flight': listen_u_ranked[i], 'algorithm': 'LISTEN-U', 'rank': i + 1})
                # st.session_state.interleaved_results = interleaved

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
                            # Save LISTEN-ranked flights (the order shown to user)
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

            # ============================================================================
            # LILO PREFERENCE LEARNING SECTION (between initial ranking and cross-validation)
            # ============================================================================
            # Only show LILO if not completed AND not showing animation
            if not st.session_state.get('lilo_completed') and not st.session_state.get('show_evaluation_animation'):
                # DEBUG: Check session state at start of LILO section
                print(f"[LILO DEBUG] Entering LILO section - all_flights exists: {bool(st.session_state.get('all_flights'))}, count: {len(st.session_state.get('all_flights', []))}")
                print(f"[LILO DEBUG] Entering LILO section - lilo_round: {st.session_state.get('lilo_round', 'NOT SET')}, search_id: {st.session_state.get('search_id', 'NOT SET')}")

                # Hide survey content on LILO page
                st.markdown("""
                <style>
                    .hideable-survey-content {
                        display: none !important;
                    }
                </style>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("# 🧠 LILO: AI-Powered Preference Learning")
                st.markdown("*Help the AI understand your flight preferences through interactive questioning*")
                st.markdown("---")
                st.markdown('<div class="lilo-section">', unsafe_allow_html=True)

                # Initialize LILO session using real language_bo_code
                if 'lilo_bridge' not in st.session_state:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        from lilo_integration import StreamlitLILOBridge

                        # Step 1: Create bridge
                        status_text.text("Loading ML libraries (PyTorch, BoTorch)...")
                        progress_bar.progress(20)
                        bridge = StreamlitLILOBridge()
                        st.session_state.lilo_bridge = bridge

                        # Step 2: Create LILO session
                        status_text.text("Creating LILO session...")
                        progress_bar.progress(50)
                        flights_data = st.session_state.get('all_flights_data', [])
                        if not flights_data:
                            st.warning("⚠️ No flight data available for LILO. Skipping LILO section.")
                            st.session_state.lilo_completed = True
                            st.rerun()

                        session = bridge.create_session(
                            session_id=f"user_{st.session_state.get('user_id', 'default')}",
                            flights_data=flights_data
                        )
                        st.session_state.lilo_session_id = session.session_id

                        # Save LILO session to database (non-critical - won't block LILO startup)
                        try:
                            from backend.db import save_lilo_session
                            lilo_db_id = save_lilo_session(
                                session_id=session.session_id,
                                search_id=st.session_state.get('search_id'),
                                completion_token=st.session_state.get('token'),
                                num_iterations=2,
                                questions_per_round=2
                            )
                            st.session_state.lilo_db_session_id = lilo_db_id
                        except Exception as db_err:
                            print(f"⚠️ Non-critical: Failed to save LILO session to database: {db_err}")
                            st.session_state.lilo_db_session_id = None

                        # Step 3: Initialize state
                        status_text.text("Initializing conversation...")
                        progress_bar.progress(75)
                        st.session_state.lilo_round = 0
                        st.session_state.lilo_chat_history = []  # Chat message history with round tracking
                        st.session_state.lilo_current_question_idx = 0  # Current question index
                        st.session_state.lilo_round1_flights = []
                        st.session_state.lilo_round2_flights = []

                        # Step 4: Generate initial questions
                        status_text.text("Generating initial questions...")
                        progress_bar.progress(90)
                        initial_questions = bridge.get_initial_questions(session.session_id)
                        st.session_state.lilo_questions = initial_questions
                        st.session_state.lilo_answers = {}  # Store answers

                        # Complete
                        progress_bar.progress(100)
                        status_text.text("✅ Ready!")
                        st.success("✅ LILO initialized successfully!")

                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"⚠️ Error initializing LILO: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        if st.button("Skip LILO and Continue", type="primary"):
                            st.session_state.lilo_completed = True
                            st.rerun()
                        st.stop()

                # LILO: Continuous chat interface (single page, no reloads between rounds)
                st.markdown("### 💬 Chat with LILO")

                # Debug info: Show current round
                current_round_display = st.session_state.get('lilo_round', 0)
                total_rounds = 3  # Round 0, 1, 2
                st.caption(f"Round {current_round_display + 1} of {total_rounds}")

                st.markdown("---")

                # Add LILO-specific CSS
                st.markdown("""
                <style>
                    /* Prevent auto-scroll on checkbox click */
                    .stCheckbox {
                        overflow-anchor: none !important;
                    }

                    /* Position chat input above footer */
                    .stChatInput {
                        position: fixed !important;
                        bottom: 60px !important;
                        left: 0 !important;
                        right: 0 !important;
                        z-index: 999 !important;
                        background: white !important;
                        padding: 10px !important;
                        box-shadow: 0 -2px 10px rgba(0,0,0,0.1) !important;
                    }

                    /* Hide content between "How to Use" and "Search Flights" during LILO */
                    .hideable-survey-content {
                        display: none !important;
                    }
                </style>
                """, unsafe_allow_html=True)

                # Initialize chat state if needed
                if 'lilo_chat_history' not in st.session_state:
                    st.session_state.lilo_chat_history = []
                if 'lilo_current_question_idx' not in st.session_state:
                    st.session_state.lilo_current_question_idx = 0
                if 'lilo_answers' not in st.session_state:
                    st.session_state.lilo_answers = {}
                if 'lilo_round' not in st.session_state:
                    st.session_state.lilo_round = 0

                # Display ALL chat history (persists across rounds)
                for msg_idx, msg in enumerate(st.session_state.lilo_chat_history):
                    if msg.get('flights'):
                        # Show flight comparison
                        render_flight_comparison(msg['flights'][0], msg['flights'][1], "Option A", "Option B")

                    # For user messages, add an edit button
                    if not msg.get('is_bot'):
                        col1, col2 = st.columns([20, 1])
                        with col1:
                            render_chat_message(msg['text'], is_bot=False)
                        with col2:
                            # Check if this message is being edited
                            edit_key = f"editing_msg_{msg_idx}"
                            if edit_key not in st.session_state:
                                st.session_state[edit_key] = False

                            if st.button("✏️", key=f"edit_btn_{msg_idx}", help="Edit this answer"):
                                st.session_state[edit_key] = not st.session_state[edit_key]
                                st.rerun()

                        # Show edit box if editing
                        if st.session_state.get(edit_key, False):
                            edited_text = st.text_area(
                                "Edit your answer:",
                                value=msg['text'],
                                key=f"edit_area_{msg_idx}",
                                height=100
                            )
                            col_save, col_cancel = st.columns([1, 1])
                            with col_save:
                                if st.button("💾 Save", key=f"save_{msg_idx}", use_container_width=True, type="primary"):
                                    # Update the message in chat history
                                    st.session_state.lilo_chat_history[msg_idx]['text'] = edited_text

                                    # Also update in lilo_answers if this was a question response
                                    # Find which question this was an answer to
                                    question_num = None
                                    for i in range(msg_idx - 1, -1, -1):
                                        if st.session_state.lilo_chat_history[i].get('is_bot') and '?' in st.session_state.lilo_chat_history[i]['text']:
                                            # Count how many questions before this one
                                            question_count = sum(1 for m in st.session_state.lilo_chat_history[:i+1] if m.get('is_bot') and '?' in m['text'])
                                            question_num = question_count - 1
                                            break

                                    if question_num is not None and f"q{question_num}" in st.session_state.lilo_answers:
                                        st.session_state.lilo_answers[f"q{question_num}"] = edited_text

                                    st.session_state[edit_key] = False
                                    st.rerun()
                            with col_cancel:
                                if st.button("❌ Cancel", key=f"cancel_{msg_idx}", use_container_width=True):
                                    st.session_state[edit_key] = False
                                    st.rerun()
                    else:
                        render_chat_message(msg['text'], is_bot=True)

                # Determine what to show next based on state
                questions = st.session_state.get('lilo_questions', [])
                current_idx = st.session_state.lilo_current_question_idx
                current_round = st.session_state.lilo_round

                # Check if we need to run iteration (all questions answered for current round)
                if current_idx >= len(questions) and current_round < 2:
                    # DEBUG: Check session state before iteration
                    print(f"[LILO DEBUG] Before iteration - all_flights exists: {bool(st.session_state.get('all_flights'))}, count: {len(st.session_state.get('all_flights', []))}")
                    print(f"[LILO DEBUG] Before iteration - all_flights_data exists: {bool(st.session_state.get('all_flights_data'))}, count: {len(st.session_state.get('all_flights_data', []))}")

                    # Add loading message to chat
                    loading_msg = "Analyzing your preferences..." if current_round == 0 else "Refining flight options..."
                    render_chat_message(loading_msg, is_bot=True)
                    st.session_state.lilo_chat_history.append({'text': loading_msg, 'is_bot': True, 'round': current_round})

                    # Run LILO iteration
                    try:
                        with st.spinner("🤖 Running LILO algorithm..."):
                            flights, next_questions = st.session_state.lilo_bridge.run_iteration(
                                st.session_state.lilo_session_id,
                                st.session_state.lilo_answers
                            )

                            # Store results
                            if current_round == 0:
                                st.session_state.lilo_round1_flights = flights
                            else:
                                st.session_state.lilo_round2_flights = flights

                            # Check if we got valid questions
                            if not next_questions or len(next_questions) == 0:
                                st.error("⚠️ LILO failed to generate questions for next round. Using fallback.")
                                # Use fallback questions
                                if current_round == 0:
                                    next_questions = [
                                        "Between the two flight options shown, which one better matches your needs and why?",
                                        "What trade-offs are you willing to make between price, duration, and convenience?"
                                    ]
                                else:
                                    next_questions = [
                                        "Looking at the refined options, which aspects are most important to you?",
                                        "Is there anything you'd like to adjust in your preferences?"
                                    ]

                            # Update state for next round
                            st.session_state.lilo_questions = next_questions
                            st.session_state.lilo_current_question_idx = 0

                            # Save iteration data to database (non-blocking - errors won't break LILO)
                            try:
                                from backend.db import save_lilo_iteration
                                if st.session_state.get('lilo_db_session_id'):
                                    save_lilo_iteration(
                                        lilo_session_id=st.session_state.lilo_db_session_id,
                                        iteration_number=current_round,
                                        user_responses=st.session_state.lilo_answers,
                                        flights_shown=flights,
                                        utility_params=None,  # Will be extracted later
                                        acquisition_values=None
                                    )
                            except Exception as db_err:
                                print(f"⚠️ Non-critical: Failed to save iteration to database: {db_err}")
                                # Continue anyway - database save is not critical for user experience

                            st.session_state.lilo_answers = {}
                            st.session_state.lilo_round += 1

                            # Add flights to chat if available
                            if len(flights) >= 2:
                                comparison_msg = "Here are the options I found:" if current_round == 0 else "Here are refined options:"
                                st.session_state.lilo_chat_history.append({
                                    'text': comparison_msg,
                                    'is_bot': True,
                                    'flights': [flights[0], flights[1]],
                                    'round': current_round + 1  # +1 because we just incremented
                                })

                            # DEBUG: Check session state before rerun
                            print(f"[LILO DEBUG] Before rerun - all_flights exists: {bool(st.session_state.get('all_flights'))}, count: {len(st.session_state.get('all_flights', []))}")
                            print(f"[LILO DEBUG] Before rerun - lilo_round: {st.session_state.lilo_round}, search_id: {st.session_state.get('search_id')}")

                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                # Check if LILO is complete (Round 2 questions all answered)
                elif current_idx >= len(questions) and current_round == 2:
                    # Save final iteration and compute rankings (do this ONCE)
                    if not st.session_state.get('lilo_data_saved'):
                        # Wrap all database operations in try-except to prevent breaking LILO
                        try:
                            from backend.db import save_lilo_iteration, save_lilo_final_rankings, save_lilo_chat_transcript, complete_lilo_session

                            lilo_db_id = st.session_state.get('lilo_db_session_id')
                            if lilo_db_id:
                                # Save final iteration (round 2)
                                save_lilo_iteration(
                                    lilo_session_id=lilo_db_id,
                                    iteration_number=2,
                                    user_responses=st.session_state.lilo_answers,
                                    flights_shown=[],
                                    utility_params=None,
                                    acquisition_values=None
                                )

                                # Save complete chat transcript
                                # Add metadata for rounds/indices
                                chat_with_metadata = []
                                msg_idx = 0
                                for msg in st.session_state.lilo_chat_history:
                                    chat_with_metadata.append({
                                        'text': msg.get('text', ''),
                                        'is_bot': msg.get('is_bot', False),
                                        'round': msg.get('round', 0),  # Use round from message, default to 0
                                        'index': msg_idx,
                                        'flight_a': msg.get('flights', [None])[0] if msg.get('flights') else None,
                                        'flight_b': msg.get('flights', [None, None])[1] if msg.get('flights') and len(msg.get('flights')) > 1 else None
                                    })
                                    msg_idx += 1

                                save_lilo_chat_transcript(lilo_db_id, chat_with_metadata)

                                # Compute final rankings
                                with st.spinner("Computing final flight rankings..."):
                                    all_flights = st.session_state.get('all_flights_data', [])
                                    ranked_flights = st.session_state.lilo_bridge.compute_final_rankings(
                                        st.session_state.lilo_session_id,
                                        all_flights
                                    )

                                    # Save rankings and get CSV
                                    csv_data = save_lilo_final_rankings(lilo_db_id, ranked_flights)
                                    st.session_state.lilo_rankings_csv = csv_data

                                    # Mark LILO session as complete
                                    complete_lilo_session(lilo_db_id)

                                    print(f"✓ LILO data saved: session {lilo_db_id}, {len(ranked_flights)} flights ranked")

                        except Exception as e:
                            print(f"⚠️ Non-critical: Failed to save LILO data to database: {e}")
                            import traceback
                            traceback.print_exc()
                            # Continue anyway - LILO can complete without database saves

                        st.session_state.lilo_data_saved = True

                    # Only add completion message once
                    completion_msg = "Perfect! I've learned your preferences. LILO is complete! ✅"
                    if not st.session_state.lilo_chat_history or st.session_state.lilo_chat_history[-1].get('text') != completion_msg:
                        st.session_state.lilo_chat_history.append({'text': completion_msg, 'is_bot': True, 'round': current_round})

                    render_chat_message(completion_msg, is_bot=True)

                    # Complete LILO and move to cross-validation (skip animation for cleaner transition)
                    st.markdown("---")
                    st.info("🎯 Thank you! Moving to the next section...")

                    # Mark LILO as completed and skip animation
                    st.session_state.lilo_completed = True
                    st.session_state.show_evaluation_animation = False

                    # Small delay then rerun for smooth transition
                    import time
                    time.sleep(1)
                    st.rerun()

                # Show current question
                elif current_idx < len(questions):
                    current_question = questions[current_idx]
                    render_chat_message(current_question, is_bot=True)

                    # Chat input with Enter key support (back to original)
                    answer = st.chat_input(
                        "Type your answer and press Enter...",
                        key=f"lilo_chat_input_{current_round}_{current_idx}"
                    )

                    if answer and answer.strip():
                        # Add Q&A to chat history with round tracking
                        st.session_state.lilo_chat_history.append({'text': current_question, 'is_bot': True, 'round': current_round})
                        st.session_state.lilo_chat_history.append({'text': answer, 'is_bot': False, 'round': current_round})

                        # Save answer
                        st.session_state.lilo_answers[f"q{current_idx}"] = answer

                        # Move to next question
                        st.session_state.lilo_current_question_idx += 1
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

            # Evaluation Animation - Show between LILO completion and cross-validation
            if st.session_state.get('show_evaluation_animation'):
                st.markdown("""
                <style>
                    /* Full-screen animation overlay */
                    .evaluation-animation {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100vw;
                        height: 100vh;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        z-index: 9999;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        overflow: hidden;
                    }

                    /* Person A watching */
                    .observer {
                        position: absolute;
                        bottom: 20px;
                        left: 50px;
                        width: 120px;
                        height: 140px;
                        opacity: 0;
                        animation: fadeInObserver 1s ease-in forwards;
                    }

                    @keyframes fadeInObserver {
                        to { opacity: 1; }
                    }

                    /* Flight path container */
                    .flight-container {
                        position: relative;
                        width: 70%;
                        height: 60%;
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 20px;
                        backdrop-filter: blur(10px);
                        border: 2px solid rgba(255, 255, 255, 0.2);
                        overflow: hidden;
                    }

                    /* Animated flight path */
                    .flight-path {
                        position: absolute;
                        top: 50%;
                        left: 10%;
                        width: 0;
                        height: 3px;
                        background: linear-gradient(90deg, #4CAF50, #2196F3, #9C27B0);
                        animation: drawPath 3s ease-in-out forwards;
                        animation-delay: 0.5s;
                    }

                    @keyframes drawPath {
                        to { width: 80%; }
                    }

                    /* Airplane icon */
                    .airplane {
                        position: absolute;
                        top: calc(50% - 15px);
                        left: 10%;
                        font-size: 30px;
                        animation: flyPlane 3s ease-in-out forwards;
                        animation-delay: 0.5s;
                    }

                    @keyframes flyPlane {
                        0% { left: 10%; }
                        100% { left: 90%; }
                    }

                    /* Telemetry overlays */
                    .telemetry {
                        position: absolute;
                        right: 30px;
                        top: 50%;
                        transform: translateY(-50%);
                        display: flex;
                        flex-direction: column;
                        gap: 20px;
                        opacity: 0;
                        animation: fadeInTelemetry 1s ease-in forwards;
                        animation-delay: 1.5s;
                    }

                    @keyframes fadeInTelemetry {
                        to { opacity: 1; }
                    }

                    .telemetry-item {
                        background: rgba(255, 255, 255, 0.9);
                        padding: 15px 20px;
                        border-radius: 10px;
                        min-width: 200px;
                        animation: pulse 2s infinite;
                    }

                    @keyframes pulse {
                        0%, 100% { transform: scale(1); box-shadow: 0 0 10px rgba(76, 175, 80, 0.3); }
                        50% { transform: scale(1.05); box-shadow: 0 0 20px rgba(76, 175, 80, 0.6); }
                    }

                    .telemetry-label {
                        font-size: 12px;
                        color: #666;
                        text-transform: uppercase;
                        margin-bottom: 5px;
                    }

                    .telemetry-value {
                        font-size: 24px;
                        font-weight: bold;
                        color: #2196F3;
                    }

                    /* Ranking scale */
                    .ranking-scale {
                        position: absolute;
                        bottom: 100px;
                        left: 50%;
                        transform: translateX(-50%);
                        display: flex;
                        gap: 10px;
                        opacity: 0;
                        animation: fadeInScale 1s ease-in forwards;
                        animation-delay: 2.5s;
                    }

                    @keyframes fadeInScale {
                        to { opacity: 1; }
                    }

                    .ranking-bar {
                        width: 40px;
                        background: rgba(255, 255, 255, 0.3);
                        border-radius: 5px 5px 0 0;
                        position: relative;
                        overflow: hidden;
                    }

                    .ranking-bar-fill {
                        position: absolute;
                        bottom: 0;
                        width: 100%;
                        background: linear-gradient(180deg, #4CAF50, #2196F3);
                        animation: fillBar 1.5s ease-out forwards;
                        animation-delay: 3s;
                    }

                    @keyframes fillBar {
                        from { height: 0; }
                    }

                    .bar-label {
                        position: absolute;
                        bottom: -25px;
                        width: 100%;
                        text-align: center;
                        color: white;
                        font-size: 11px;
                    }

                    /* Score assembly */
                    .score-container {
                        position: absolute;
                        top: 30px;
                        left: 50%;
                        transform: translateX(-50%);
                        opacity: 0;
                        animation: fadeInScore 1s ease-in forwards;
                        animation-delay: 4.5s;
                    }

                    @keyframes fadeInScore {
                        to { opacity: 1; }
                    }

                    .score-box {
                        background: white;
                        padding: 30px 50px;
                        border-radius: 15px;
                        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                        text-align: center;
                    }

                    .score-label {
                        font-size: 14px;
                        color: #666;
                        margin-bottom: 10px;
                    }

                    .score-value {
                        font-size: 64px;
                        font-weight: bold;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        animation: countUp 2s ease-out forwards;
                    }

                    @keyframes countUp {
                        from { opacity: 0; transform: scale(0.5); }
                        to { opacity: 1; transform: scale(1); }
                    }

                    /* Playback controls */
                    .playback-controls {
                        position: absolute;
                        bottom: 30px;
                        left: 50%;
                        transform: translateX(-50%);
                        display: flex;
                        gap: 15px;
                        opacity: 0;
                        animation: fadeInControls 1s ease-in forwards;
                        animation-delay: 1s;
                    }

                    @keyframes fadeInControls {
                        to { opacity: 1; }
                    }

                    .control-btn {
                        background: rgba(255, 255, 255, 0.2);
                        border: 2px solid rgba(255, 255, 255, 0.4);
                        color: white;
                        padding: 10px 20px;
                        border-radius: 25px;
                        cursor: pointer;
                        transition: all 0.3s;
                        backdrop-filter: blur(5px);
                    }

                    .control-btn:hover {
                        background: rgba(255, 255, 255, 0.3);
                        transform: scale(1.05);
                    }

                    /* Continue button */
                    .continue-btn {
                        position: absolute;
                        bottom: 50px;
                        right: 50px;
                        background: white;
                        color: #667eea;
                        padding: 15px 40px;
                        border-radius: 30px;
                        font-size: 18px;
                        font-weight: bold;
                        cursor: pointer;
                        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
                        opacity: 0;
                        animation: fadeInContinue 1s ease-in forwards;
                        animation-delay: 5.5s;
                        transition: all 0.3s;
                    }

                    .continue-btn:hover {
                        transform: scale(1.1);
                        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
                    }

                    @keyframes fadeInContinue {
                        to { opacity: 1; }
                    }

                    /* Confirmation checkmark */
                    .checkmark {
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%) scale(0);
                        width: 100px;
                        height: 100px;
                        border-radius: 50%;
                        background: white;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 60px;
                        animation: popCheckmark 0.5s ease-out forwards;
                        animation-delay: 6s;
                        z-index: 10000;
                    }

                    @keyframes popCheckmark {
                        0% { transform: translate(-50%, -50%) scale(0); }
                        50% { transform: translate(-50%, -50%) scale(1.2); }
                        100% { transform: translate(-50%, -50%) scale(1); }
                    }
                </style>

                <div class="evaluation-animation">
                    <!-- Observer (Person A) -->
                    <div class="observer">
                        <div style="font-size: 80px;">👨‍💼</div>
                        <div style="color: white; text-align: center; margin-top: 5px; font-size: 12px;">Person A</div>
                    </div>

                    <!-- Flight container -->
                    <div class="flight-container">
                        <!-- Flight path -->
                        <div class="flight-path"></div>

                        <!-- Airplane -->
                        <div class="airplane">✈️</div>

                        <!-- Telemetry overlays -->
                        <div class="telemetry">
                            <div class="telemetry-item">
                                <div class="telemetry-label">Altitude</div>
                                <div class="telemetry-value">35,000 ft</div>
                            </div>
                            <div class="telemetry-item" style="animation-delay: 0.2s;">
                                <div class="telemetry-label">Speed</div>
                                <div class="telemetry-value">550 mph</div>
                            </div>
                            <div class="telemetry-item" style="animation-delay: 0.4s;">
                                <div class="telemetry-label">Smoothness</div>
                                <div class="telemetry-value">98%</div>
                            </div>
                            <div class="telemetry-item" style="animation-delay: 0.6s;">
                                <div class="telemetry-label">Efficiency</div>
                                <div class="telemetry-value">95%</div>
                            </div>
                        </div>

                        <!-- Playback controls -->
                        <div class="playback-controls">
                            <div class="control-btn">⏮️ Replay</div>
                            <div class="control-btn">⏸️ Pause</div>
                            <div class="control-btn">⏭️ Skip</div>
                        </div>
                    </div>

                    <!-- Ranking scale -->
                    <div class="ranking-scale">
                        <div class="ranking-bar" style="height: 100px;">
                            <div class="ranking-bar-fill" style="--fill-height: 85%;"></div>
                            <div class="bar-label">Comfort</div>
                        </div>
                        <div class="ranking-bar" style="height: 120px;">
                            <div class="ranking-bar-fill" style="--fill-height: 92%; animation-delay: 3.2s;"></div>
                            <div class="bar-label">Speed</div>
                        </div>
                        <div class="ranking-bar" style="height: 110px;">
                            <div class="ranking-bar-fill" style="--fill-height: 88%; animation-delay: 3.4s;"></div>
                            <div class="bar-label">Price</div>
                        </div>
                        <div class="ranking-bar" style="height: 105px;">
                            <div class="ranking-bar-fill" style="--fill-height: 90%; animation-delay: 3.6s;"></div>
                            <div class="bar-label">Route</div>
                        </div>
                    </div>

                    <!-- Score assembly -->
                    <div class="score-container">
                        <div class="score-box">
                            <div class="score-label">COMPOSITE SCORE</div>
                            <div class="score-value">8.9</div>
                        </div>
                    </div>

                    <!-- Checkmark confirmation -->
                    <div class="checkmark">✓</div>
                </div>

                <script>
                    // Auto-advance to cross-validation after animation completes
                    setTimeout(function() {
                        // Find and click the hidden continue button
                        const continueBtn = document.querySelector('[data-testid="evaluation-continue"]');
                        if (continueBtn) {
                            continueBtn.click();
                        }
                    }, 7000); // 7 seconds total animation time
                </script>
                """, unsafe_allow_html=True)

                # Auto-advance using session state timer
                if 'animation_start_time' not in st.session_state:
                    import time
                    st.session_state.animation_start_time = time.time()

                # Check if 7 seconds have passed
                import time
                elapsed = time.time() - st.session_state.animation_start_time
                if elapsed >= 7:
                    # Clear timer and advance
                    del st.session_state.animation_start_time
                    st.session_state.lilo_completed = True
                    st.session_state.show_evaluation_animation = False
                    st.rerun()
                else:
                    # Show manual continue button
                    if st.button("Continue to Cross-Validation", key="evaluation-continue", help="Continue", type="primary"):
                        del st.session_state.animation_start_time
                        st.session_state.lilo_completed = True
                        st.session_state.show_evaluation_animation = False
                        st.rerun()
                    # Force rerun to check timer again
                    time.sleep(0.1)
                    st.rerun()

            # Cross-validation section (before survey) - only show after LILO is completed
            if st.session_state.get('lilo_completed') and not st.session_state.get('cross_validation_completed'):
                try:
                    from backend.db import get_previous_search_for_validation

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
                    st.markdown("### 🤝 Help Validate Another Search")
                    st.markdown("*Before completing your session, please help us by ranking flights for another user's search*")

                    # Fetch previous search for cross-validation
                    if 'cross_val_data' not in st.session_state:
                        st.session_state.cross_val_data = get_previous_search_for_validation(st.session_state.session_id)
                except ImportError:
                    # Function not available yet (old deployment) - skip cross-validation
                    st.session_state.cross_validation_completed = True
                    st.session_state.cross_val_data = None

                if st.session_state.get('cross_val_data') is not None:
                    if not st.session_state.cross_val_data:
                        # No previous search available - skip cross-validation
                        st.info("ℹ️ No previous searches available for validation. You're one of the first users!")
                        st.session_state.cross_validation_completed = True
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
                                    if st.button("✅ Submit Cross-Validation Rankings", key="cv_submit", type="primary", use_container_width=True):
                                        from backend.db import save_cross_validation
                                        success = save_cross_validation(
                                            reviewer_session_id=st.session_state.session_id,
                                            reviewed_session_id=cross_val['session_id'],
                                            reviewed_search_id=cross_val['search_id'],
                                            reviewed_prompt=cross_val['prompt'],
                                            reviewed_flights=cross_val['flights'],
                                            selected_flights=st.session_state.cross_val_selected_flights,
                                            reviewer_token=st.session_state.get('token')
                                        )
                                        if success:
                                            st.session_state.cross_validation_completed = True
                                            st.success("✅ Thank you for helping validate!")
                                            st.rerun()
                                        else:
                                            st.error("⚠️ Failed to save. Please try again.")
                            else:
                                st.caption("No flights selected yet")

                    # Close cross-validation div
                    st.markdown('</div>', unsafe_allow_html=True)

            # Survey section (after cross-validation)
            if st.session_state.get('cross_validation_completed') and not st.session_state.get('survey_completed'):
                # Hide content above survey and auto-scroll to top
                st.markdown("""
                <style>
                    /* Fade out previous content */
                    .stApp > header,
                    [data-testid="stSidebar"],
                    .main > div:not(:has(.survey-section)) {
                        opacity: 0.1;
                        pointer-events: none;
                    }
                    .survey-section {
                        background: white;
                        position: relative;
                        z-index: 1000;
                        padding-top: 20px;
                    }
                    /* Hide content between "How to Use" and "Search Flights" on survey page */
                    .hideable-survey-content {
                        display: none !important;
                    }

                    /* Make survey radio buttons bigger and easier to click */
                    .survey-section .stRadio > div {
                        gap: 0.75rem !important;
                    }
                    .survey-section .stRadio label {
                        background-color: #f0f2f6 !important;
                        padding: 1rem 1.5rem !important;
                        border-radius: 8px !important;
                        border: 2px solid #e0e0e0 !important;
                        cursor: pointer !important;
                        transition: all 0.2s ease !important;
                        font-size: 1rem !important;
                        min-width: 120px !important;
                        text-align: center !important;
                        display: inline-block !important;
                    }
                    .survey-section .stRadio label:hover {
                        background-color: #e8eaf0 !important;
                        border-color: #4F8BF9 !important;
                        transform: translateY(-2px) !important;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
                    }
                    .survey-section .stRadio label[data-selected="true"] {
                        background-color: #4F8BF9 !important;
                        color: white !important;
                        border-color: #4F8BF9 !important;
                        font-weight: 600 !important;
                    }

                    /* Make submit button bigger */
                    .survey-section button[kind="primary"] {
                        padding: 1rem 2rem !important;
                        font-size: 1.2rem !important;
                        min-height: 60px !important;
                    }
                </style>
                <script>
                    // Auto-scroll to top when survey loads
                    window.scrollTo(0, 0);
                </script>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown('<div id="survey-step-1" class="survey-section">', unsafe_allow_html=True)
                st.markdown("### 📋 Quick Feedback Survey")
                st.markdown("*Please take 2-3 minutes to help us improve the tool*")

                # Initialize survey responses in session state
                if 'survey_data' not in st.session_state:
                    st.session_state.survey_data = {}

                # Q1: Satisfaction
                st.markdown("**1. How satisfied are you with the flight recommendations you received?**")
                satisfaction = st.radio(
                    "Satisfaction",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"][x-1],
                    key="q1_satisfaction",
                    label_visibility="collapsed",
                    horizontal=True,
                    index=None
                )

                # Q2: Ease of use
                st.markdown("**2. How easy was it to use the flight search and ranking system?**")
                ease_of_use = st.radio(
                    "Ease of use",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: ["Very Difficult", "Difficult", "Neutral", "Easy", "Very Easy"][x-1],
                    key="q2_ease",
                    label_visibility="collapsed",
                    horizontal=True,
                    index=None
                )

                # Q3: Technical issues
                st.markdown("**3. Did you encounter any technical issues or errors during your session?**")
                encountered_issues = st.radio(
                    "Issues",
                    options=["No", "Yes"],
                    key="q3_issues",
                    label_visibility="collapsed",
                    horizontal=True,
                    index=None
                )
                issues_description = None
                if encountered_issues == "Yes":
                    issues_description = st.text_area(
                        "Please describe the issues:",
                        key="q3_issues_desc",
                        placeholder="e.g., The page froze when I clicked submit..."
                    )

                # Q4: Search method
                st.markdown("**4. Which search method did you use?**")
                search_method = st.radio(
                    "Search method",
                    options=["Regular Search", "AI Search with LISTEN", "Both"],
                    key="q4_method",
                    label_visibility="collapsed",
                    index=None
                )

                # Q5: Understanding AI ranking
                st.markdown("**5. Did you understand how the AI ranked your flights?**")
                understood_ranking = st.radio(
                    "Understanding",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: ["Not at all", "Slightly", "Somewhat", "Mostly", "Completely"][x-1],
                    key="q5_understood",
                    label_visibility="collapsed",
                    horizontal=True,
                    index=None
                )

                # Q6: Helpful features
                st.markdown("**6. Which features were most helpful in making your decision?** *(Select all that apply)*")
                helpful_features = []
                if st.checkbox("Top 5 ranked flights sidebar", key="q6_sidebar"):
                    helpful_features.append("Top 5 sidebar")
                if st.checkbox("Drag-to-rerank functionality", key="q6_drag"):
                    helpful_features.append("Drag-to-rerank")
                if st.checkbox("Flight filtering (price, airline, stops, etc.)", key="q6_filter"):
                    helpful_features.append("Filtering")
                if st.checkbox("AI-generated rankings", key="q6_ai"):
                    helpful_features.append("AI rankings")
                if st.checkbox("Detailed flight metrics (duration, stops, times, etc.)", key="q6_metrics"):
                    helpful_features.append("Flight metrics")

                # Q7: Flights matched expectations
                st.markdown("**7. Did the top-ranked flights match what you were looking for?**")
                flights_matched = st.radio(
                    "Match",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: ["Not at all", "Slightly", "Somewhat", "Mostly", "Perfectly"][x-1],
                    key="q7_matched",
                    label_visibility="collapsed",
                    horizontal=True,
                    index=None
                )

                # Q8: Confusion/frustration
                st.markdown("**8. Was there anything confusing or frustrating about the experience?**")
                confusing_frustrating = st.text_area(
                    "Your feedback:",
                    key="q8_confusing",
                    placeholder="e.g., I didn't understand why flight X was ranked higher than Y...",
                    label_visibility="collapsed"
                )

                # Q9: Missing features
                st.markdown("**9. Was there any information or feature missing that would have helped you make a better decision?**")
                missing_features = st.text_area(
                    "Your feedback:",
                    key="q9_missing",
                    placeholder="e.g., I wanted to see baggage fees, seat availability...",
                    label_visibility="collapsed"
                )

                # Q10: Would use again
                st.markdown("**10. Would you use this tool again for future flight searches?**")
                would_use_again = st.radio(
                    "Use again",
                    options=["Yes", "Maybe", "No"],
                    key="q10_again",
                    label_visibility="collapsed",
                    horizontal=True,
                    index=None
                )
                would_use_again_reason = st.text_input(
                    "Why or why not? (optional)",
                    key="q10_reason",
                    placeholder="e.g., It saved me time compared to other sites..."
                )

                # Q11: Comparison to others
                st.markdown("**11. Compared to other flight search tools (Google Flights, Kayak, etc.), how would you rate this experience?**")
                compared_to_others = st.radio(
                    "Comparison",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: ["Much Worse", "Worse", "About the Same", "Better", "Much Better"][x-1],
                    key="q11_compared",
                    label_visibility="collapsed",
                    horizontal=True,
                    index=None
                )

                # Q12: Additional comments
                st.markdown("**12. Any other comments or suggestions?** *(optional)*")
                additional_comments = st.text_area(
                    "Your feedback:",
                    key="q12_comments",
                    placeholder="Any other thoughts you'd like to share...",
                    label_visibility="collapsed"
                )

                # Submit survey button
                if st.button("📨 Submit Survey", type="primary", use_container_width=True):
                    # Validate required fields
                    missing_fields = []
                    if satisfaction is None:
                        missing_fields.append("Question 1 (Satisfaction)")
                    if ease_of_use is None:
                        missing_fields.append("Question 2 (Ease of use)")
                    if encountered_issues is None:
                        missing_fields.append("Question 3 (Technical issues)")
                    if search_method is None:
                        missing_fields.append("Question 4 (Search method)")
                    if understood_ranking is None:
                        missing_fields.append("Question 5 (Understanding AI ranking)")
                    if flights_matched is None:
                        missing_fields.append("Question 7 (Flights matched expectations)")
                    if would_use_again is None:
                        missing_fields.append("Question 10 (Would use again)")
                    if compared_to_others is None:
                        missing_fields.append("Question 11 (Comparison to others)")

                    if missing_fields:
                        st.error(f"⚠️ Please answer all required questions: {', '.join(missing_fields)}")
                    else:
                        # Collect all survey data
                        survey_data = {
                            'satisfaction': satisfaction,
                            'ease_of_use': ease_of_use,
                            'encountered_issues': encountered_issues,
                            'issues_description': issues_description if encountered_issues == "Yes" else None,
                            'search_method': search_method,
                            'understood_ranking': understood_ranking,
                            'helpful_features': helpful_features if helpful_features else None,
                            'flights_matched': flights_matched,
                            'confusing_frustrating': confusing_frustrating if confusing_frustrating else None,
                            'missing_features': missing_features if missing_features else None,
                            'would_use_again': would_use_again,
                            'would_use_again_reason': would_use_again_reason if would_use_again_reason else None,
                            'compared_to_others': compared_to_others,
                            'additional_comments': additional_comments if additional_comments else None
                        }

                        # Save to database (REQUIRED - don't allow continuation if save fails)
                        try:
                            from backend.db import save_survey_response
                            success = save_survey_response(
                                session_id=st.session_state.session_id,
                                survey_data=survey_data,
                                completion_token=st.session_state.token
                            )

                            if success:
                                st.session_state.survey_completed = True
                                st.success("✅ Thank you for your feedback!")
                                st.rerun()
                            else:
                                st.error("⚠️ Failed to save survey response to database. Please try again.")
                                st.info("If this persists, please contact the research team.")
                        except Exception as e:
                            # Show actual error for debugging
                            st.error(f"⚠️ Database Error: {str(e)}")
                            import traceback
                            error_details = traceback.format_exc()
                            print(f"[SURVEY ERROR] {error_details}")
                            # Show error details in expandable section
                            with st.expander("Technical Details (for debugging)"):
                                st.code(error_details)
                            st.info("Please contact the research team with the error above.")

                # Close survey div
                st.markdown('</div>', unsafe_allow_html=True)

            # Show completion page with CSV download after survey
            if (st.session_state.get('cross_validation_completed') and
                st.session_state.get('survey_completed') and
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

                st.markdown("### 📥 Download Your Complete Session Data")
                st.markdown("You can download a comprehensive CSV file containing all your session data:")

                # Generate comprehensive CSV
                token = st.session_state.get('token')
                if token:
                    try:
                        from export_session_data import export_session_to_csv
                        import io

                        # Generate CSV in memory
                        csv_file = export_session_to_csv(token, output_file=None)

                        # Read the CSV file
                        if csv_file:
                            with open(csv_file, 'r', encoding='utf-8') as f:
                                csv_data = f.read()

                            # Provide download button
                            st.download_button(
                                label="📄 Download Complete Session Data (CSV)",
                                data=csv_data,
                                file_name=f"session_data_{token}.csv",
                                mime="text/csv",
                                use_container_width=True
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

                # Mark token as used NOW (unless it's the DEMO token)
                if st.session_state.token != "DEMO":
                    from backend.db import mark_token_used
                    mark_token_used(st.session_state.token)

                    # Mark countdown as completed
                    st.session_state.countdown_completed = True

                    # Show session ended message
                    st.warning("🔒 Session ended. Thank you for your participation!")
                    st.info("This link can no longer be used. To participate again, please request a new token from the research team.")
                    st.stop()
                else:
                    # DEMO token - allow continued use
                    st.session_state.countdown_completed = True
                    st.success("✅ Thank you! You can continue using this demo link to submit more rankings.")
                    st.info("💡 This is a DEMO token - you can use it as many times as you want!")

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

        # Only show "What would you like to do next?" after LILO is completed
        if st.session_state.get('lilo_completed'):
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
            st.markdown("**Select your top 5 flights for EACH direction and drag to rank them →**")
        else:
            st.markdown(f"### ✈️ Found {len(st.session_state.all_flights)} Flights")
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
