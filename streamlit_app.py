"""
Flight Metrics Web Application - Streamlit Version
Algorithm-based flight ranking with interleaved results.
"""
import streamlit as st
import os
from datetime import datetime, timedelta
import pandas as pd
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import backend modules
from backend.amadeus_client import AmadeusClient
from backend.database import get_db, test_connection, get_session_local
from backend.utils.builtin_algorithms import BUILTIN_ALGORITHMS, get_algorithm, list_algorithms
from backend.utils.parse_duration import parse_duration_to_minutes
from backend.utils.nl_parser import parse_flight_query

# Page configuration
st.set_page_config(
    page_title="Flight Metrics App",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'flights' not in st.session_state:
    st.session_state.flights = []
if 'interleaved_results' not in st.session_state:
    st.session_state.interleaved_results = []
if 'parsed_query' not in st.session_state:
    st.session_state.parsed_query = None
if 'uploaded_algorithms' not in st.session_state:
    st.session_state.uploaded_algorithms = {}

# Initialize Amadeus client
@st.cache_resource
def get_amadeus_client():
    return AmadeusClient()

amadeus = get_amadeus_client()

# Database connection test
@st.cache_resource
def check_database():
    try:
        test_connection()
        return True
    except:
        return False

db_connected = check_database()

# ============================================================================
# HEADER
# ============================================================================
st.title("✈️ Flight Ranking with Algorithm Interleaving")
st.markdown("Search flights and compare results from two different ranking algorithms")

# Sidebar
with st.sidebar:
    st.header("System Status")
    if db_connected:
        st.success("✅ Database Connected")
    else:
        st.warning("⚠️ Database Disconnected")

    st.markdown("---")
    st.header("About")
    st.markdown("""
    This app demonstrates:
    - **Natural Language** flight search
    - **Multiple ranking algorithms** (LISTEN-U, LISTEN-T, simple heuristics)
    - **Interleaved results** from two algorithms
    - **Custom algorithm upload**
    """)

    st.markdown("---")
    st.header("Available Algorithms")

    # Group by category
    simple_algos = [name for name, info in BUILTIN_ALGORITHMS.items() if info['category'] == 'Simple']
    advanced_algos = [name for name, info in BUILTIN_ALGORITHMS.items() if info['category'] == 'Advanced']
    smart_algos = [name for name, info in BUILTIN_ALGORITHMS.items() if info['category'] == 'Smart']

    if simple_algos:
        st.markdown("**Simple:**")
        for name in simple_algos:
            st.markdown(f"• {name}")

    if smart_algos:
        st.markdown("**Smart:**")
        for name in smart_algos:
            st.markdown(f"• {name}")

    if advanced_algos:
        st.markdown("**Advanced:**")
        for name in advanced_algos:
            st.markdown(f"• {name}")

    if st.session_state.uploaded_algorithms:
        st.markdown("**Custom:**")
        for name in st.session_state.uploaded_algorithms.keys():
            st.markdown(f"• {name}")

# ============================================================================
# MAIN FLOW
# ============================================================================

# Step 1: Natural Language Input
st.header("🗣️ Step 1: Describe Your Flight Search")
st.markdown("*Examples: 'I want to go from Ithaca to San Francisco on 10/31' or 'Fly from NYC to LA in November as cheap as possible'*")

nl_query = st.text_area(
    "What flight are you looking for?",
    value="I want to go from New York to Los Angeles on November 15th",
    height=80,
    placeholder="I want to go from [origin] to [destination] on [date]..."
)

col_parse1, col_parse2 = st.columns([1, 3])
with col_parse1:
    if st.button("✨ Parse Request", type="secondary", use_container_width=True):
        parsed = parse_flight_query(nl_query)
        st.session_state.parsed_query = parsed

        if parsed['parsed_successfully']:
            st.rerun()

# Show parsed results if available
if st.session_state.parsed_query and st.session_state.parsed_query['parsed_successfully']:
    parsed = st.session_state.parsed_query

    st.success("✅ Successfully parsed your request!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Origin:** {parsed['origin']}")
        if 'original_origin' in parsed:
            st.caption(f"(Originally: {parsed['original_origin']})")
    with col2:
        st.info(f"**Destination:** {parsed['destination']}")
        if 'original_destination' in parsed:
            st.caption(f"(Originally: {parsed['original_destination']})")
    with col3:
        st.info(f"**Date:** {parsed['date']}")

    # Show warnings
    if parsed['warnings']:
        for warning in parsed['warnings']:
            st.warning(f"ℹ️ {warning}")

    # Show preferences
    if parsed['preferences']:
        st.markdown("**Detected preferences:**")
        prefs_text = []
        if parsed['preferences'].get('prefer_nonstop'):
            prefs_text.append("Direct flights preferred")
        if parsed['preferences'].get('prefer_cheap'):
            prefs_text.append("Budget-friendly")
        if parsed['preferences'].get('prefer_fast'):
            prefs_text.append("Fastest route")
        if parsed['preferences'].get('prefer_comfort'):
            prefs_text.append("Comfortable seating")
        if prefs_text:
            st.info(" • " + " • ".join(prefs_text))

st.markdown("---")

# Step 2: Algorithm Selection
st.header("🤖 Step 2: Choose Two Algorithms to Compare")
st.markdown("Results will be interleaved: A1, B1, A2, B2, A3, B3...")

# Get all available algorithms
all_algorithms = list(BUILTIN_ALGORITHMS.keys()) + list(st.session_state.uploaded_algorithms.keys())

col1, col2 = st.columns(2)
with col1:
    algo_a = st.selectbox(
        "Algorithm A",
        options=all_algorithms,
        index=0 if "Cheapest" in all_algorithms else 0,
        help="First ranking algorithm"
    )

    # Show algorithm description
    if algo_a in BUILTIN_ALGORITHMS:
        st.caption(f"📝 {BUILTIN_ALGORITHMS[algo_a]['description']}")
    elif algo_a in st.session_state.uploaded_algorithms:
        st.caption(f"📝 Custom uploaded algorithm")

with col2:
    algo_b = st.selectbox(
        "Algorithm B",
        options=all_algorithms,
        index=1 if len(all_algorithms) > 1 else 0,
        help="Second ranking algorithm"
    )

    # Show algorithm description
    if algo_b in BUILTIN_ALGORITHMS:
        st.caption(f"📝 {BUILTIN_ALGORITHMS[algo_b]['description']}")
    elif algo_b in st.session_state.uploaded_algorithms:
        st.caption(f"📝 Custom uploaded algorithm")

st.markdown("---")

# Step 3: Search Parameters (optional refinement)
with st.expander("🔧 Advanced Search Parameters (Optional)", expanded=False):
    st.markdown("These will override the natural language parsing above")

    col1, col2, col3, col4 = st.columns(4)

    # Use parsed values as defaults if available
    default_origin = "JFK"
    default_destination = "LAX"
    default_date = datetime.now() + timedelta(days=30)

    if st.session_state.parsed_query and st.session_state.parsed_query['parsed_successfully']:
        parsed = st.session_state.parsed_query
        default_origin = parsed.get('origin', 'JFK')
        default_destination = parsed.get('destination', 'LAX')
        if parsed.get('date'):
            try:
                default_date = datetime.strptime(parsed['date'], "%Y-%m-%d").date()
            except:
                pass

    with col1:
        origin = st.text_input("Origin (IATA Code)", value=default_origin, max_chars=3).upper()

    with col2:
        destination = st.text_input("Destination (IATA Code)", value=default_destination, max_chars=3).upper()

    with col3:
        departure_date = st.date_input("Departure Date", value=default_date)

    with col4:
        adults = st.number_input("Adults", min_value=1, max_value=9, value=1)

    max_results = st.slider("Max Results", min_value=5, max_value=50, value=10)

# Step 4: Search and Rank Button
if st.button("🚀 Search Flights & Show Interleaved Rankings", type="primary", use_container_width=True):
    with st.spinner("Searching flights and computing rankings..."):
        try:
            # Check API credentials
            api_key = os.getenv("AMADEUS_API_KEY") or st.secrets.get("AMADEUS_API_KEY", "")
            api_secret = os.getenv("AMADEUS_API_SECRET") or st.secrets.get("AMADEUS_API_SECRET", "")

            if not api_key or not api_secret:
                st.error("⚠️ Amadeus API credentials not found! Please add them to Streamlit secrets.")
                st.info("Go to Settings → Secrets and add AMADEUS_API_KEY and AMADEUS_API_SECRET")
                st.stop()

            # Call Amadeus API
            results = amadeus.search_flights(
                origin=origin,
                destination=destination,
                departure_date=departure_date.strftime("%Y-%m-%d"),
                adults=adults,
                max_results=max_results
            )

            # Handle both dict with 'data' key and direct list
            flight_offers = []
            if isinstance(results, dict) and 'data' in results:
                flight_offers = results['data']
            elif isinstance(results, list):
                flight_offers = results

            if flight_offers and len(flight_offers) > 0:
                # Parse flight data
                flights_data = []
                for offer in flight_offers[:max_results]:
                    itinerary = offer['itineraries'][0]
                    segments = itinerary['segments']

                    flight_info = {
                        'id': offer['id'],
                        'price': float(offer['price']['total']),
                        'currency': offer['price']['currency'],
                        'duration_min': parse_duration_to_minutes(itinerary['duration']),
                        'stops': len(segments) - 1,
                        'departure_time': segments[0]['departure']['at'],
                        'arrival_time': segments[-1]['arrival']['at'],
                        'airline': segments[0]['carrierCode'],
                        'flight_number': segments[0]['number'],
                        'origin': segments[0]['departure']['iataCode'],
                        'destination': segments[-1]['arrival']['iataCode']
                    }
                    flights_data.append(flight_info)

                st.session_state.flights = flights_data

                # Extract preferences for algorithms
                preferences = {
                    'original_query': nl_query,
                    'prefer_cheap': False,
                    'prefer_fast': False,
                    'prefer_nonstop': False,
                    'prefer_comfort': False
                }

                if st.session_state.parsed_query and st.session_state.parsed_query.get('preferences'):
                    preferences.update(st.session_state.parsed_query['preferences'])

                # Get algorithm functions
                algo_a_func = get_algorithm(algo_a) or st.session_state.uploaded_algorithms.get(algo_a)
                algo_b_func = get_algorithm(algo_b) or st.session_state.uploaded_algorithms.get(algo_b)

                if not algo_a_func or not algo_b_func:
                    st.error("❌ Selected algorithms not found!")
                    st.stop()

                # Run both algorithms
                ranked_a = algo_a_func(flights_data.copy(), preferences)
                ranked_b = algo_b_func(flights_data.copy(), preferences)

                # Interleave results (A1, B1, A2, B2, A3, B3...)
                interleaved = []
                max_len = max(len(ranked_a), len(ranked_b))

                for i in range(max_len):
                    if i < len(ranked_a):
                        interleaved.append({
                            'flight': ranked_a[i],
                            'source': 'A',
                            'algorithm': algo_a,
                            'position_in_algo': i + 1
                        })
                    if i < len(ranked_b):
                        interleaved.append({
                            'flight': ranked_b[i],
                            'source': 'B',
                            'algorithm': algo_b,
                            'position_in_algo': i + 1
                        })

                st.session_state.interleaved_results = interleaved
                st.session_state.algo_a_name = algo_a
                st.session_state.algo_b_name = algo_b

                st.success(f"✅ Found {len(flights_data)} flights and computed rankings!")
                st.rerun()

            else:
                st.error("No flights found. Try different search criteria.")
                st.info("Tips: Use valid IATA codes (JFK, LAX, etc.) and a future date (at least 1 day ahead)")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ============================================================================
# DISPLAY INTERLEAVED RESULTS
# ============================================================================
if st.session_state.interleaved_results:
    st.markdown("---")
    st.header("🏆 Interleaved Results")

    algo_a_name = st.session_state.get('algo_a_name', 'Algorithm A')
    algo_b_name = st.session_state.get('algo_b_name', 'Algorithm B')

    st.info(f"📊 Results alternating between **{algo_a_name}** and **{algo_b_name}**")

    # Create display table
    display_data = []
    for idx, item in enumerate(st.session_state.interleaved_results, 1):
        flight = item['flight']
        source_label = f"🔵 {item['algorithm']}" if item['source'] == 'A' else f"🟢 {item['algorithm']}"

        display_data.append({
            'Rank': idx,
            'Source': source_label,
            'Algo Position': f"#{item['position_in_algo']}",
            'Airline': flight['airline'],
            'Flight': flight['flight_number'],
            'Price': f"${flight['price']:.2f}",
            'Duration': f"{flight['duration_min']:.0f} min",
            'Stops': flight['stops'],
            'Departure': pd.to_datetime(flight['departure_time']).strftime('%H:%M'),
            'Arrival': pd.to_datetime(flight['arrival_time']).strftime('%H:%M')
        })

    # Display table
    st.dataframe(
        pd.DataFrame(display_data),
        use_container_width=True,
        hide_index=True,
        height=600
    )

    # Summary statistics
    st.markdown("### 📈 Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Results", len(st.session_state.interleaved_results))

    with col2:
        count_a = sum(1 for item in st.session_state.interleaved_results if item['source'] == 'A')
        st.metric(f"{algo_a_name} Results", count_a)

    with col3:
        count_b = sum(1 for item in st.session_state.interleaved_results if item['source'] == 'B')
        st.metric(f"{algo_b_name} Results", count_b)

    with col4:
        avg_price = sum(item['flight']['price'] for item in st.session_state.interleaved_results) / len(st.session_state.interleaved_results)
        st.metric("Avg Price", f"${avg_price:.2f}")

    # Show top 3 from each algorithm
    st.markdown("### 🥇 Top 3 Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{algo_a_name} - Top 3**")
        top_a = [item for item in st.session_state.interleaved_results if item['source'] == 'A'][:3]
        for i, item in enumerate(top_a, 1):
            f = item['flight']
            st.write(f"{i}. {f['airline']}{f['flight_number']} - ${f['price']:.2f} ({f['duration_min']:.0f} min, {f['stops']} stops)")

    with col2:
        st.markdown(f"**{algo_b_name} - Top 3**")
        top_b = [item for item in st.session_state.interleaved_results if item['source'] == 'B'][:3]
        for i, item in enumerate(top_b, 1):
            f = item['flight']
            st.write(f"{i}. {f['airline']}{f['flight_number']} - ${f['price']:.2f} ({f['duration_min']:.0f} min, {f['stops']} stops)")

# ============================================================================
# ALGORITHM UPLOAD SECTION
# ============================================================================
st.markdown("---")
st.header("📤 Upload Custom Algorithm")

with st.expander("ℹ️ Algorithm Interface Requirements", expanded=False):
    st.markdown("""
    ### Required Function Signature

    Your algorithm must implement this exact function signature:

    ```python
    def rank_flights(flights: List[Dict], preferences: Dict) -> List[Dict]:
        \"\"\"
        Rank flights based on your custom logic.

        Args:
            flights: List of flight dictionaries with these fields:
                - id (str): Unique flight identifier
                - price (float): Total price
                - currency (str): Currency code (e.g., 'USD')
                - duration_min (int): Flight duration in minutes
                - stops (int): Number of stops
                - departure_time (str): ISO datetime string
                - arrival_time (str): ISO datetime string
                - airline (str): Airline code (e.g., 'AA')
                - flight_number (str): Flight number
                - origin (str): Origin airport code
                - destination (str): Destination airport code

            preferences: Dictionary with these optional fields:
                - original_query (str): User's natural language query
                - prefer_cheap (bool): User prefers cheap flights
                - prefer_fast (bool): User prefers fast flights
                - prefer_nonstop (bool): User prefers direct flights
                - prefer_comfort (bool): User prefers comfortable flights
                - max_stops (int): Maximum acceptable stops

        Returns:
            List[Dict]: Same flights list, sorted by your ranking (best first)
        \"\"\"
        # Your ranking logic here
        return sorted(flights, key=lambda x: ...)
    ```

    ### Example Algorithms

    **Simple Price-based:**
    ```python
    def rank_flights(flights, preferences):
        return sorted(flights, key=lambda x: x['price'])
    ```

    **Multi-factor scoring:**
    ```python
    def rank_flights(flights, preferences):
        def score(flight):
            price_score = flight['price'] / 1000  # Normalize
            time_score = flight['duration_min'] / 600  # Normalize
            stop_penalty = flight['stops'] * 0.3
            return price_score + time_score + stop_penalty

        return sorted(flights, key=score)
    ```

    ### Upload Your Code

    1. Create a Python file with your `rank_flights` function
    2. Upload it below
    3. Give it a unique name
    4. It will be added to the algorithm list!
    """)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Python file (.py)",
        type=['py'],
        help="File must contain a 'rank_flights' function with the correct signature"
    )

with col2:
    algorithm_name = st.text_input(
        "Algorithm Name",
        placeholder="My Custom Ranker",
        help="Unique name for your algorithm"
    )

if uploaded_file and algorithm_name:
    if st.button("➕ Add Algorithm", type="primary"):
        try:
            # Read the uploaded file
            code = uploaded_file.read().decode('utf-8')

            # Create a namespace and execute the code
            namespace = {}
            exec(code, namespace)

            # Check if rank_flights function exists
            if 'rank_flights' not in namespace:
                st.error("❌ Error: File must contain a function named 'rank_flights'")
            else:
                # Validate function signature
                import inspect
                func = namespace['rank_flights']
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                if len(params) != 2:
                    st.error(f"❌ Error: rank_flights must take exactly 2 parameters, got {len(params)}")
                else:
                    # Store the algorithm
                    st.session_state.uploaded_algorithms[algorithm_name] = func
                    st.success(f"✅ Successfully added algorithm: {algorithm_name}")
                    st.info(f"You can now select '{algorithm_name}' from the algorithm dropdowns above!")
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Error loading algorithm: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Show currently uploaded algorithms
if st.session_state.uploaded_algorithms:
    st.markdown("### Your Uploaded Algorithms")
    for name in st.session_state.uploaded_algorithms.keys():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"✅ **{name}**")
        with col2:
            if st.button(f"Remove", key=f"remove_{name}"):
                del st.session_state.uploaded_algorithms[name]
                st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by Amadeus Flight API • Algorithm-Based Ranking System")
