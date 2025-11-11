"""
Flight Evaluation Web App - Clean Suno-style Interface
Collect user feedback on algorithm-ranked flights for research.
"""
import streamlit as st
import os
import sys
import uuid
from datetime import datetime
from dotenv import load_dotenv
from streamlit_sortables import sort_items

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

# Import modules
from backend.amadeus_client import AmadeusClient
from backend.prompt_parser import parse_flight_prompt_with_llm, get_test_api_fallback
from backend.utils.parse_duration import parse_duration_to_minutes

load_dotenv()

# Initialize database (create tables if they don't exist)
try:
    from backend.db import init_db
    init_db()
except Exception as e:
    # Don't crash the app if database setup fails, just log it
    print(f"Database initialization: {str(e)}")

# Page config
st.set_page_config(
    page_title="Flight Ranker",
    page_icon="✈️",
    layout="wide"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'flights' not in st.session_state:
    st.session_state.flights = []
if 'interleaved_results' not in st.session_state:
    st.session_state.interleaved_results = []
if 'shortlist' not in st.session_state:
    st.session_state.shortlist = []
if 'parsed_params' not in st.session_state:
    st.session_state.parsed_params = None

# Initialize Amadeus (force fresh instance, don't cache with wrong credentials)
def get_amadeus():
    # Ensure .env is loaded fresh
    load_dotenv(override=True)
    return AmadeusClient()

# Don't cache initially to avoid credential issues
amadeus = get_amadeus()

# Custom CSS for Suno-like clean design
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">✈️ Flight Ranker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Describe your flight and get personalized rankings</div>', unsafe_allow_html=True)

# Main prompt input
prompt = st.text_area(
    "",
    value="",
    height=150,
    placeholder="Example: I need to fly from Ithaca NY to Washington DC on October 20th. I prefer direct flights and want to avoid early morning departures before 7am. I fly United and am not very price sensitive.",
    label_visibility="collapsed"
)

# Search button
if st.button("🔍 Search Flights", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("Please describe your flight needs")
    else:
        # Clear previous shortlist and results when starting new search
        st.session_state.shortlist = []
        st.session_state.interleaved_results = []

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

                col1, col2, col3 = st.columns(3)
                with col1:
                    origins_str = " or ".join(parsed['origins'])
                    st.info(f"**From:** {origins_str}")
                with col2:
                    dests_str = " or ".join(parsed['destinations'])
                    st.info(f"**To:** {dests_str}")
                with col3:
                    st.info(f"**Date:** {parsed.get('departure_date', 'Not specified')}")

                # Search flights from all origin/destination combinations
                all_flights = []

                st.info("✈️ Searching flights from Amadeus API...")

                for origin_code in parsed['origins'][:1]:  # Limit to first origin for speed
                    for dest_code in parsed['destinations'][:1]:  # Limit to first dest
                        # Check test API compatibility
                        origin, origin_warning = get_test_api_fallback(origin_code)
                        dest, dest_warning = get_test_api_fallback(dest_code)

                        if origin_warning:
                            st.warning(origin_warning)
                        if dest_warning:
                            st.warning(dest_warning)

                        st.info(f"Searching: {origin} → {dest} on {parsed['departure_date']}")

                        # Search flights
                        results = amadeus.search_flights(
                            origin=origin,
                            destination=dest,
                            departure_date=parsed['departure_date'],
                            adults=1,
                            max_results=50  # Get more to have options
                        )

                        # Debug: show raw results
                        with st.expander("🔍 Debug: Amadeus API Response"):
                            st.write(f"Type: {type(results)}")
                            if isinstance(results, dict):
                                st.write(f"Keys: {results.keys()}")
                                if 'data' in results:
                                    st.write(f"Number of flights: {len(results['data'])}")
                            elif isinstance(results, list):
                                st.write(f"Number of flights: {len(results)}")
                            st.json(results if isinstance(results, (dict, list)) else str(results))

                        # Parse results
                        if isinstance(results, list):
                            flight_offers = results
                        elif isinstance(results, dict) and 'data' in results:
                            flight_offers = results['data']
                        else:
                            flight_offers = []

                        for offer in flight_offers:
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
                            all_flights.append(flight_info)

                if not all_flights:
                    st.error("No flights found. Try different dates or airports.")
                    st.stop()

                st.session_state.flights = all_flights

                # Run algorithms
                preferences = parsed.get('preferences', {})

                # Add origins/destinations to preferences for LISTEN-U
                preferences['origins'] = parsed.get('origins', [])
                preferences['destinations'] = parsed.get('destinations', [])

                # Algorithm 1: Cheapest
                cheapest_ranked = sorted(all_flights, key=lambda x: x['price'])[:10]

                # Algorithm 2: Fastest
                fastest_ranked = sorted(all_flights, key=lambda x: x['duration_min'])[:10]

                # Algorithm 3: LISTEN-U (using LISTEN's main.py framework)
                try:
                    from backend.listen_main_wrapper import rank_flights_with_listen_main

                    st.info("🤖 Running LISTEN-U algorithm via main.py (this may take 30-60 seconds)...")

                    listen_u_ranked = rank_flights_with_listen_main(
                        flights=all_flights,
                        user_prompt=prompt,
                        user_preferences=preferences,
                        n_iterations=5  # 5 comparison batches
                    )

                    st.success("✅ LISTEN-U complete!")

                except Exception as e:
                    st.warning(f"⚠️ LISTEN-U failed ({str(e)}), using fallback preference-aware ranking")

                    # Fallback to simple preference-aware
                    def preference_score(flight):
                        score = 0
                        if preferences.get('prefer_cheap'):
                            score += flight['price'] / 1000
                        if preferences.get('prefer_fast'):
                            score += flight['duration_min'] / 300
                        if preferences.get('prefer_direct'):
                            score += flight['stops'] * 0.5
                        return score if score > 0 else flight['price'] / 500 + flight['duration_min'] / 300

                    listen_u_ranked = sorted(all_flights, key=preference_score)[:10]

                # Interleave results (round-robin)
                interleaved = []
                for i in range(10):
                    if i < len(cheapest_ranked):
                        interleaved.append({
                            'flight': cheapest_ranked[i],
                            'algorithm': 'Cheapest',
                            'rank': i + 1
                        })
                    if i < len(fastest_ranked):
                        interleaved.append({
                            'flight': fastest_ranked[i],
                            'algorithm': 'Fastest',
                            'rank': i + 1
                        })
                    if i < len(listen_u_ranked):
                        interleaved.append({
                            'flight': listen_u_ranked[i],
                            'algorithm': 'LISTEN-U',
                            'rank': i + 1
                        })

                st.session_state.interleaved_results = interleaved
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Display results
if st.session_state.interleaved_results:
    st.markdown("---")

    # Header with export button
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(f"### ✈️ Found {len(st.session_state.interleaved_results)} Flights")
        st.markdown("**Drag your top 5 choices to the shortlist on the right →**")
    with header_col2:
        import json
        # Prepare export data
        export_data = {
            'search_params': st.session_state.parsed_params,
            'flights': [item['flight'] for item in st.session_state.interleaved_results],
            'algorithms': {
                'cheapest': [item for item in st.session_state.interleaved_results if item['algorithm'] == 'Cheapest'],
                'fastest': [item for item in st.session_state.interleaved_results if item['algorithm'] == 'Fastest'],
                'listen_u': [item for item in st.session_state.interleaved_results if item['algorithm'] == 'LISTEN-U']
            }
        }
        export_json = json.dumps(export_data, indent=2, default=str)

        st.download_button(
            label="📥 Export Flights",
            data=export_json,
            file_name=f"flight_options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    col_flights, col_shortlist = st.columns([2, 1])

    with col_flights:
        st.markdown("#### All Flights")

        for idx, item in enumerate(st.session_state.interleaved_results):
            flight = item['flight']
            algo = item['algorithm']
            rank = item['rank']

            # Create unique key
            flight_key = f"{flight['id']}_{idx}"

            # Check if in shortlist
            in_shortlist = flight_key in [f['key'] for f in st.session_state.shortlist]

            col1, col2 = st.columns([1, 4])

            with col1:
                if st.button("+" if not in_shortlist else "✓",
                             key=f"btn_{flight_key}",
                             disabled=in_shortlist or len(st.session_state.shortlist) >= 5):
                    st.session_state.shortlist.append({
                        'key': flight_key,
                        'flight': flight,
                        'algorithm': algo,
                        'rank': rank
                    })
                    st.rerun()

            with col2:
                st.markdown(f"""
                <div style="line-height: 1.2; margin: 0; padding: 0;">
                <strong>{flight['airline']}{flight['flight_number']}</strong> • {flight['origin']} → {flight['destination']}<br>
                <span style="font-size: 0.9em;">${flight['price']:.0f} • {flight['duration_min']//60}h {flight['duration_min']%60}m • {flight['stops']} stops</span><br>
                <em style="font-size: 0.85em; color: #666;">From {algo} (#{rank})</em>
                </div>
                """, unsafe_allow_html=True)

    with col_shortlist:
        st.markdown("#### 📋 Your Shortlist (Drag to Reorder)")
        st.markdown(f"**{len(st.session_state.shortlist)}/5 selected**")

        if st.session_state.shortlist:
            # Create list of flight labels for sorting
            flight_labels = []
            for i, item in enumerate(st.session_state.shortlist):
                flight = item['flight']
                label = f"#{i+1}: {flight['airline']}{flight['flight_number']} - ${flight['price']:.0f}"
                flight_labels.append(label)

            # Display sortable list
            sorted_labels = sort_items(
                flight_labels,
                multi_containers=False,
                direction='vertical'
            )

            # If order changed, update the shortlist (no rerun to avoid flash)
            if sorted_labels != flight_labels:
                # Build new order based on sorted labels
                new_order = []
                for sorted_label in sorted_labels:
                    # Extract original position from label (after #)
                    original_pos = int(sorted_label.split(':')[0].replace('#', '')) - 1
                    new_order.append(st.session_state.shortlist[original_pos])

                st.session_state.shortlist = new_order

            # Submit button
            st.markdown("---")
            if len(st.session_state.shortlist) == 5:
                if st.button("✅ Submit Rankings", type="primary", use_container_width=True):
                    # Save to database
                    try:
                        from backend.db import save_search_and_rankings

                        search_id = save_search_and_rankings(
                            session_id=st.session_state.session_id,
                            user_prompt=st.session_state.get('original_prompt', prompt),
                            parsed_params=st.session_state.parsed_params or {},
                            interleaved_results=st.session_state.interleaved_results,
                            user_shortlist=st.session_state.shortlist
                        )

                        st.success(f"✅ Thank you! Your rankings have been saved (Search ID: {search_id})")
                        st.balloons()

                    except Exception as e:
                        st.error(f"Failed to save rankings: {str(e)}")
                        st.info("Your rankings were collected but could not be saved to the database.")
            else:
                st.info(f"Select {5 - len(st.session_state.shortlist)} more flights to submit")

        else:
            st.info("Click + to add flights from the list")

# Footer
st.markdown("---")
st.caption("Built for flight ranking research • Data collected for algorithm evaluation")
