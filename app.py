"""
Flight Evaluation Web App - Clean Suno-style Interface
Collect user feedback on algorithm-ranked flights for research.
"""
import streamlit as st
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

    # Create a set of selected flight IDs for quick lookup
    selected_ids = {f['id']: (idx + 1) for idx, f in enumerate(selected_flights)}

    for idx, flight in enumerate(all_flights):
        # Determine if this flight was ranked
        is_best = flight['id'] in selected_ids
        rank = selected_ids.get(flight['id'], unranked_value)

        # Generate unique_id
        unique_id = f"{flight['origin']}_{flight['destination']}{idx + 1}"

        # Get airline name (carrier code for now, could map to full names)
        name = flight['airline']

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

        # Build row
        row = {
            'unique_id': unique_id,
            'is_best': is_best,
            'rank': rank,
            'name': name,
            'origin': flight['origin'],
            'destination': flight['destination'],
            'departure_time': dept_time_str,
            'arrival_time': arr_time_str,
            'duration': f"{flight['duration_min']//60} hr {flight['duration_min']%60} min",
            'stops': float(flight['stops']),
            'price': float(flight['price']),
            'dis_from_origin': 0.0,
            'dis_from_dest': 0.0,
            'departure_dt': dept_dt_formatted,
            'departure_seconds': dept_seconds,
            'arrival_dt': arr_dt_formatted,
            'arrival_seconds': arr_seconds,
            'duration_min': flight['duration_min']
        }
        csv_rows.append(row)

    # Convert to DataFrame and then to CSV
    df = pd.DataFrame(csv_rows)
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
if 'all_flights' not in st.session_state:  # Changed from 'flights' to 'all_flights'
    st.session_state.all_flights = []
if 'selected_flights' not in st.session_state:  # Changed from 'shortlist'
    st.session_state.selected_flights = []
if 'parsed_params' not in st.session_state:
    st.session_state.parsed_params = None
if 'csv_generated' not in st.session_state:
    st.session_state.csv_generated = False

# TEMPORARY COMMENT: Old session state for algorithm-based ranking
# if 'interleaved_results' not in st.session_state:
#     st.session_state.interleaved_results = []
# if 'shortlist' not in st.session_state:
#     st.session_state.shortlist = []

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
        # Clear previous selections when starting new search
        st.session_state.selected_flights = []
        st.session_state.all_flights = []
        st.session_state.csv_generated = False

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
                            max_results=250  # Get ALL available flights (increased from 50)
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

                # Store all flights (no algorithm ranking)
                st.session_state.all_flights = all_flights
                st.success(f"✅ Found {len(all_flights)} flights!")
                st.rerun()

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
    st.markdown("---")
    st.markdown(f"### ✈️ Found {len(st.session_state.all_flights)} Flights")
    st.markdown("**Select your top 5 flights and drag to rank them →**")

    col_flights, col_ranking = st.columns([2, 1])

    with col_flights:
        st.markdown("#### All Available Flights")

        # Display all flights with checkboxes
        for idx, flight in enumerate(st.session_state.all_flights):
            # Generate unique_id for display
            unique_id = f"{flight['origin']}_{flight['destination']}{idx + 1}"

            # Check if already selected
            is_selected = flight['id'] in [f['id'] for f in st.session_state.selected_flights]

            col1, col2 = st.columns([1, 5])

            with col1:
                # Checkbox to select/deselect
                selected = st.checkbox(
                    "✓" if is_selected else "",
                    value=is_selected,
                    key=f"select_{flight['id']}_{idx}",
                    label_visibility="collapsed",
                    disabled=(not is_selected and len(st.session_state.selected_flights) >= 5)
                )

                if selected and not is_selected:
                    # Add to selected flights
                    st.session_state.selected_flights.append(flight)
                    st.rerun()
                elif not selected and is_selected:
                    # Remove from selected flights
                    st.session_state.selected_flights = [
                        f for f in st.session_state.selected_flights if f['id'] != flight['id']
                    ]
                    st.rerun()

            with col2:
                # Show flight metrics as requested
                dept_dt = datetime.fromisoformat(flight['departure_time'].replace('Z', '+00:00'))
                arr_dt = datetime.fromisoformat(flight['arrival_time'].replace('Z', '+00:00'))
                dept_time_display = dept_dt.strftime("%I:%M %p")
                arr_time_display = arr_dt.strftime("%I:%M %p")

                st.markdown(f"""
                <div style="line-height: 1.3; margin: 0; padding: 0.3rem 0;">
                <strong>{unique_id}</strong> | <strong>{flight['airline']}</strong> {flight['flight_number']}<br>
                <span style="font-size: 0.95em;">{flight['origin']} → {flight['destination']} | {dept_time_display} - {arr_time_display}</span><br>
                <span style="font-size: 0.9em; color: #555;">${flight['price']:.0f} | {flight['duration_min']} min | {flight['stops']} stops</span>
                </div>
                """, unsafe_allow_html=True)

    with col_ranking:
        st.markdown("#### 📋 Your Top 5 (Drag to Rank)")
        st.markdown(f"**{len(st.session_state.selected_flights)}/5 selected**")

        if st.session_state.selected_flights:
            # Create list of flight labels for sorting
            flight_labels = []
            for i, flight in enumerate(st.session_state.selected_flights):
                label = f"#{i+1}: {flight['airline']}{flight['flight_number']} - ${flight['price']:.0f}"
                flight_labels.append(label)

            # Display sortable list
            sorted_labels = sort_items(
                flight_labels,
                multi_containers=False,
                direction='vertical'
            )

            # If order changed, update the selected_flights
            if sorted_labels != flight_labels:
                new_order = []
                for sorted_label in sorted_labels:
                    original_pos = int(sorted_label.split(':')[0].replace('#', '')) - 1
                    new_order.append(st.session_state.selected_flights[original_pos])
                st.session_state.selected_flights = new_order

            # Submit button
            st.markdown("---")
            if len(st.session_state.selected_flights) == 5:
                if st.button("✅ Submit Rankings", type="primary", use_container_width=True):
                    # Generate CSV
                    csv_data = generate_flight_csv(
                        st.session_state.all_flights,
                        st.session_state.selected_flights,
                        k=5
                    )

                    # Save to database
                    try:
                        from backend.db import save_search_and_csv

                        search_id = save_search_and_csv(
                            session_id=st.session_state.session_id,
                            user_prompt=st.session_state.get('original_prompt', ''),
                            parsed_params=st.session_state.parsed_params or {},
                            all_flights=st.session_state.all_flights,
                            selected_flights=st.session_state.selected_flights,
                            csv_data=csv_data
                        )

                        st.session_state.csv_data = csv_data
                        st.session_state.csv_generated = True
                        st.session_state.search_id = search_id
                        st.success(f"✅ Rankings saved! (Search ID: {search_id})")
                        st.balloons()
                        st.rerun()

                    except Exception as e:
                        st.error(f"Failed to save rankings: {str(e)}")
                        # Still allow CSV download even if DB save fails
                        st.session_state.csv_data = csv_data
                        st.session_state.csv_generated = True
                        st.rerun()
            else:
                st.info(f"Select {5 - len(st.session_state.selected_flights)} more flights")
        else:
            st.info("Check boxes on the left to select flights")

    # Show CSV download button if rankings were submitted
    if st.session_state.csv_generated and hasattr(st.session_state, 'csv_data'):
        st.markdown("---")
        st.markdown("### 📥 Download Your Results")
        st.download_button(
            label="Download CSV File",
            data=st.session_state.csv_data,
            file_name=f"flight_rankings_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )

# Footer
st.markdown("---")
st.caption("Built for flight ranking research • Data collected for algorithm evaluation")
