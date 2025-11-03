"""
Flight Metrics Web Application - Streamlit Version
Combines flight search, LISTEN algorithms, and evaluation in one app.
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
from backend.models.flight import Flight, ListenRanking, TeamDraftResult, Rating
from backend.utils.listen_algorithms import ListenU, ListenT
from backend.utils.parse_duration import parse_duration_to_minutes

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
if 'selected_flights' not in st.session_state:
    st.session_state.selected_flights = []
if 'listen_u_results' not in st.session_state:
    st.session_state.listen_u_results = None
if 'listen_t_results' not in st.session_state:
    st.session_state.listen_t_results = None

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

# Title and header
st.title("✈️ Flight Metrics & LISTEN Evaluation")
st.markdown("Search flights, evaluate with LISTEN algorithms, and compare rankings")

# Sidebar for database status
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
    - **Flight Search** via Amadeus API
    - **LISTEN-U** (Utility Refinement)
    - **LISTEN-T** (Tournament Selection)
    - **Manual Ranking & Evaluation**
    """)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search Flights", "🤖 LISTEN Algorithms", "📊 Manual Ranking", "📈 Results"])

# ============================================================================
# TAB 1: FLIGHT SEARCH
# ============================================================================
with tab1:
    st.header("Search Flights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        origin = st.text_input("Origin (IATA Code)", value="JFK", max_chars=3).upper()

    with col2:
        destination = st.text_input("Destination (IATA Code)", value="LAX", max_chars=3).upper()

    with col3:
        default_date = datetime.now() + timedelta(days=30)
        departure_date = st.date_input("Departure Date", value=default_date)

    with col4:
        adults = st.number_input("Number of Adults", min_value=1, max_value=9, value=1)

    col5, col6 = st.columns(2)
    with col5:
        max_results = st.slider("Max Results", min_value=5, max_value=50, value=10)

    if st.button("🔍 Search Flights", type="primary", use_container_width=True):
        with st.spinner("Searching flights via Amadeus API..."):
            try:
                # Check API credentials
                import os
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

                # Debug: Show what we got back
                st.write("🔍 DEBUG INFO:")
                st.write(f"- API Response type: {type(results)}")
                if isinstance(results, dict):
                    st.write(f"- Response keys: {list(results.keys())}")
                    if 'errors' in results:
                        st.error(f"API Error: {results['errors']}")
                    if 'data' in results:
                        st.write(f"- Number of flights: {len(results['data'])}")
                        st.write(f"- First flight type: {type(results['data'][0])}")
                elif isinstance(results, list):
                    st.write(f"- Response is a list with {len(results)} items")
                    st.write(f"- First item type: {type(results[0])}")

                if results and 'data' in results and len(results['data']) > 0:
                    flights_data = []
                    for offer in results['data'][:max_results]:
                        # Parse flight data
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
                            'segments': len(segments)
                        }
                        flights_data.append(flight_info)

                    st.session_state.flights = flights_data
                    st.success(f"✅ Found {len(flights_data)} flights!")
                else:
                    st.error("No flights found. Try different search criteria.")
                    st.info("Tips: Use valid IATA codes (JFK, LAX, etc.) and a future date (at least 1 day ahead)")
                    if results:
                        st.write("Full API response:", results)

            except Exception as e:
                st.error(f"Error searching flights: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Display flights
    if st.session_state.flights:
        st.subheader(f"Found {len(st.session_state.flights)} Flights")

        # Convert to DataFrame for display
        df = pd.DataFrame(st.session_state.flights)

        # Format columns
        df['duration_hrs'] = df['duration_min'] / 60
        df['departure_time'] = pd.to_datetime(df['departure_time']).dt.strftime('%Y-%m-%d %H:%M')
        df['arrival_time'] = pd.to_datetime(df['arrival_time']).dt.strftime('%Y-%m-%d %H:%M')

        # Display table
        display_cols = ['airline', 'flight_number', 'price', 'currency', 'duration_hrs', 'stops', 'departure_time', 'arrival_time']
        st.dataframe(
            df[display_cols].style.format({
                'price': '${:.2f}',
                'duration_hrs': '{:.2f}h'
            }),
            use_container_width=True,
            hide_index=True
        )

        # Flight selection for LISTEN
        st.markdown("### Select Flights for LISTEN Evaluation")
        selected_indices = st.multiselect(
            "Choose flights to evaluate:",
            options=range(len(st.session_state.flights)),
            format_func=lambda x: f"Flight {x+1}: {st.session_state.flights[x]['airline']}{st.session_state.flights[x]['flight_number']} - ${st.session_state.flights[x]['price']}"
        )

        if selected_indices:
            st.session_state.selected_flights = [st.session_state.flights[i] for i in selected_indices]
            st.success(f"✅ Selected {len(selected_indices)} flights for evaluation")

# ============================================================================
# TAB 2: LISTEN ALGORITHMS
# ============================================================================
with tab2:
    st.header("LISTEN Algorithms")

    if not st.session_state.selected_flights:
        st.info("👈 Please search and select flights in the 'Search Flights' tab first")
    else:
        st.success(f"✅ {len(st.session_state.selected_flights)} flights selected for evaluation")

        # Algorithm selection
        algorithm = st.radio(
            "Choose LISTEN Algorithm:",
            ["LISTEN-U (Utility Refinement)", "LISTEN-T (Tournament Selection)"],
            horizontal=True
        )

        # Preference utterance
        st.markdown("### Enter Your Preferences")
        preference = st.text_area(
            "Describe what you're looking for in natural language:",
            value="I want the cheapest flight with minimal stops",
            height=100
        )

        col1, col2 = st.columns(2)

        if "LISTEN-U" in algorithm:
            with col1:
                max_iterations = st.slider("Max Iterations", min_value=1, max_value=10, value=3)

            if st.button("▶️ Run LISTEN-U", type="primary"):
                with st.spinner("Running LISTEN-U algorithm..."):
                    try:
                        listen_u = ListenU()
                        results = listen_u.rank_flights(
                            st.session_state.selected_flights,
                            preference,
                            max_iterations
                        )
                        st.session_state.listen_u_results = results
                        st.success("✅ LISTEN-U completed!")
                    except Exception as e:
                        st.error(f"Error running LISTEN-U: {str(e)}")

            # Display LISTEN-U results
            if st.session_state.listen_u_results:
                st.markdown("### LISTEN-U Results")

                # Show learned weights
                st.markdown("#### Learned Attribute Weights")
                weights_df = pd.DataFrame([st.session_state.listen_u_results['final_weights']])
                st.dataframe(weights_df.style.format("{:.3f}"), use_container_width=True)

                # Show ranked flights
                st.markdown("#### Ranked Flights")
                ranked_flights = st.session_state.listen_u_results['ranked_flights']
                ranked_df = pd.DataFrame(ranked_flights)

                if 'utility_score' in ranked_df.columns:
                    display_cols = ['airline', 'flight_number', 'price', 'duration_min', 'stops', 'utility_score']
                    st.dataframe(
                        ranked_df[display_cols].style.format({
                            'price': '${:.2f}',
                            'duration_min': '{:.0f} min',
                            'utility_score': '{:.4f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

        else:  # LISTEN-T
            with col1:
                num_rounds = st.slider("Number of Rounds", min_value=1, max_value=10, value=3)
            with col2:
                batch_size = st.slider("Batch Size", min_value=2, max_value=8, value=4)

            if st.button("▶️ Run LISTEN-T", type="primary"):
                with st.spinner("Running LISTEN-T tournament..."):
                    try:
                        listen_t = ListenT()
                        results = listen_t.rank_flights(
                            st.session_state.selected_flights,
                            preference,
                            num_rounds,
                            batch_size
                        )
                        st.session_state.listen_t_results = results
                        st.success("✅ LISTEN-T completed!")
                    except Exception as e:
                        st.error(f"Error running LISTEN-T: {str(e)}")

            # Display LISTEN-T results
            if st.session_state.listen_t_results:
                st.markdown("### LISTEN-T Results")

                # Show winner
                winner = st.session_state.listen_t_results['winner']
                st.markdown("#### 🏆 Tournament Winner")
                st.success(f"**{winner['airline']}{winner['flight_number']}** - ${winner['price']:.2f}")

                # Show champions
                st.markdown("#### Round Champions")
                champions = st.session_state.listen_t_results['champions']
                champions_df = pd.DataFrame(champions)
                display_cols = ['airline', 'flight_number', 'price', 'duration_min', 'stops']
                st.dataframe(
                    champions_df[display_cols].style.format({
                        'price': '${:.2f}',
                        'duration_min': '{:.0f} min'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

# ============================================================================
# TAB 3: MANUAL RANKING
# ============================================================================
with tab3:
    st.header("Manual Flight Ranking")

    if not st.session_state.selected_flights:
        st.info("👈 Please search and select flights in the 'Search Flights' tab first")
    else:
        st.markdown("Rank the flights by dragging them in your preferred order:")

        # Simple ranking with selectbox
        st.markdown("### Rank Your Flights (1 = Best)")

        rankings = {}
        for i, flight in enumerate(st.session_state.selected_flights):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{flight['airline']}{flight['flight_number']}** - ${flight['price']:.2f}, {flight['duration_min']:.0f} min, {flight['stops']} stops")
            with col2:
                rank = st.number_input(
                    f"Rank",
                    min_value=1,
                    max_value=len(st.session_state.selected_flights),
                    value=i+1,
                    key=f"rank_{i}"
                )
                rankings[i] = rank

        if st.button("💾 Save Rankings", type="primary"):
            st.success("✅ Rankings saved!")

            # Store rankings in session
            sorted_flights = sorted(
                zip(st.session_state.selected_flights, rankings.values()),
                key=lambda x: x[1]
            )
            st.session_state.manual_ranking = [f for f, r in sorted_flights]

# ============================================================================
# TAB 4: RESULTS COMPARISON
# ============================================================================
with tab4:
    st.header("Results & Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.listen_u_results:
            st.markdown("### LISTEN-U Top 3")
            top3 = st.session_state.listen_u_results['ranked_flights'][:3]
            for i, flight in enumerate(top3, 1):
                st.write(f"{i}. {flight['airline']}{flight['flight_number']} (${flight['price']:.2f})")
        else:
            st.info("No LISTEN-U results yet")

    with col2:
        if st.session_state.listen_t_results:
            st.markdown("### LISTEN-T Winner")
            winner = st.session_state.listen_t_results['winner']
            st.success(f"🏆 {winner['airline']}{winner['flight_number']}")
            st.write(f"${winner['price']:.2f}")
        else:
            st.info("No LISTEN-T results yet")

    with col3:
        if 'manual_ranking' in st.session_state:
            st.markdown("### Your Top Choice")
            top = st.session_state.manual_ranking[0]
            st.success(f"⭐ {top['airline']}{top['flight_number']}")
            st.write(f"${top['price']:.2f}")
        else:
            st.info("No manual ranking yet")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by Amadeus Flight API • LISTEN Algorithms")
