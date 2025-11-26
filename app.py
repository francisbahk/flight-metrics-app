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
if 'csv_data_outbound' not in st.session_state:
    st.session_state.csv_data_outbound = None
if 'csv_data_return' not in st.session_state:
    st.session_state.csv_data_return = None

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

# How to Use section
st.markdown("---")
st.markdown("### 📖 How to Use")
st.markdown("""
1. **Describe your flight** - Enter your travel details in natural language (origin, destination, dates, preferences)
2. **Review results** - Browse all available flights ~~sorted by your criteria~~
3. **Select top 5** - Check the boxes next to your 5 favorite flights (for both outbound and return if applicable)
4. **Drag to rank** - Reorder your selections by dragging them in the right panel
5. **Submit & download** - Click submit to save your rankings and download as CSV
""")
st.markdown("---")

# Main prompt input
prompt = st.text_area(
    "",
    value="",
    height=150,
    placeholder="",
    label_visibility="collapsed"
)

# Animated placeholder CSS and JavaScript (must be after text_area)
st.markdown("""
<style>
    .stTextArea {
        position: relative;
    }
    .animated-placeholder {
        position: absolute;
        top: 12px;
        left: 12px;
        right: 12px;
        color: #94a3b8;
        font-family: "Source Code Pro", monospace;
        font-size: 14px;
        line-height: 1.5;
        pointer-events: none;
        white-space: pre-wrap;
        z-index: 1;
        opacity: 1;
        transition: opacity 0.3s ease;
    }
    .animated-placeholder.hidden {
        opacity: 0;
        display: none;
    }
    .stTextArea textarea {
        position: relative;
        z-index: 2;
        background-color: transparent !important;
    }
</style>
<script>
    (function() {
        const prompts = [
            `I would like to take a trip from Chicago to New York City with my brother the weekend of October 11, 2025. Time is of the essence, so I prefer to maximize my time there. I will be leaving from Times Square area, so I can fly from any of the three major airports. I heavily prefer to fly into ORD.
I do not feel the need to strictly minimize cost; however, I would prefer to keep the fare to under 400 dollars. Obviously, if different flights meet my requirements, I prefer the cheaper one. I prefer direct flights.
I would like to maximize my time in NYC on Sunday. It would be ideal to leave on the second-to-last flight leaving from the departure airport to Chicago, in case of delays and cancellations. Worst case, I would like there to be an early Monday morning departure to Chicago from the airport, in case of cancellations.
I have no preference for airline. I would prefer to not leave NYC before 5 PM. I am okay with an early morning departure, as long as I arrive in Chicago by around 9 AM, as I will need to go to work. The earlier the arrival Monday morning, the better.`,
            `On November 3rd I need to fly from where I live, in Ithaca NY, to a conference in Reston VA. The conference starts the next day (November 4th) at 9am. I'd like to sleep well but if my travel plans are disrupted and I arrive late, it's ok. I'll either fly out of Ithaca, Syracuse, Elmira, or Binghamton.
I'll fly to DCA or IAD. For all my flights, I don't like having to get up before 7am to be on time to my flight.  I'd like to avoid the amount of time I need to spend driving / taking Ubers / taking transit to airports both at home and at my destination.
I prefer flying out of my local airport in Ithaca rather than driving or taking an Uber to a nearby airport in Syracuse, Elmira, or Binghamton.
I want to avoid extra connections because they take more time and increase the chance of missing a connection. I can move pretty quickly
through airports so connections longer than 45 min are fine but for connections that are tighter than that I worry about missing my flight
if there is a delay. If the connection is earlier in the day and there are lots of other ways to get to my destination in the event of a missed
connection, then a 30 min connection is fine.
I prefer to avoid long layovers. Under 60 minutes is fine, under 90 minutes is not a huge deal, and over 90 minutes starts to get annoying.
I feel that flying late in the day or connecting through airports with poor on-time performance like EWR increases my chance of a delay.
I don't like JFK because the food choices are poor (except for Shake Shack). When I fly to Europe from the US, I don't like taking a redeye but I know I'll usually have to take one. When I do take a redeye, I don't like to have a long layover early in the morning — I prefer to just arrive at my destination.  If I do have a layover, I prefer to land later in the morning so that I can get some sleep on the plane.
I prefer to fly United because I'm a frequent flyer with them. When I fly for work, my travel is usually reimbursed from federal grants. Because of this, I must comply with the Fly America Act. This requires me to fly on a US carrier unless there are no other options. Even if I'm allowed to reimburse a trip on a non-US carrier, I don't want to because it creates extra paperwork.
For longer trips, I am happy to return to an airport that is different from the one I left from because I probably wouldn't drive my car in any case. When I do this, I'll take an Uber, rent a car, or get a ride. For shorter trips, however, I do prefer to return to the airport I left from so that I can drive to the airport, unless it saves me a lot of trouble.
I am not very price sensitive. It is ok to pay 20% more than the cheapest fare if the itinerary is more convenient. But if the fare is outrageous then that's problematic.
I usually don't check bags except on very long trips.`
        ];

        let currentPromptIndex = 0;
        let currentCharIndex = 0;
        let isHolding = false;
        let placeholderDiv = null;
        const typingSpeed = 20; // ms per character
        const holdDuration = 3000; // ms to hold completed text
        const fadeOutDuration = 500; // ms to fade out

        function typeWriter() {
            if (!placeholderDiv || placeholderDiv.classList.contains('hidden')) {
                return;
            }

            if (isHolding) {
                return;
            }

            if (currentCharIndex < prompts[currentPromptIndex].length) {
                placeholderDiv.textContent = prompts[currentPromptIndex].substring(0, currentCharIndex + 1);
                currentCharIndex++;
                setTimeout(typeWriter, typingSpeed);
            } else {
                // Finished typing, hold for a moment
                isHolding = true;
                setTimeout(() => {
                    // Fade out
                    placeholderDiv.style.opacity = '0';
                    setTimeout(() => {
                        // Move to next prompt
                        currentPromptIndex = (currentPromptIndex + 1) % prompts.length;
                        currentCharIndex = 0;
                        isHolding = false;
                        placeholderDiv.style.opacity = '1';
                        typeWriter();
                    }, fadeOutDuration);
                }, holdDuration);
            }
        }

        function setupAnimatedPlaceholder() {
            const textAreaContainer = document.querySelector('.stTextArea');
            const textArea = textAreaContainer ? textAreaContainer.querySelector('textarea') : null;

            if (!textArea || textAreaContainer.querySelector('.animated-placeholder')) {
                if (!textArea) {
                    setTimeout(setupAnimatedPlaceholder, 100);
                }
                return;
            }

            // Create and inject placeholder div
            placeholderDiv = document.createElement('div');
            placeholderDiv.className = 'animated-placeholder';
            textAreaContainer.insertBefore(placeholderDiv, textArea);

            // Start typing animation
            setTimeout(typeWriter, 500);

            // Hide placeholder on focus or input
            textArea.addEventListener('focus', () => {
                if (placeholderDiv) {
                    placeholderDiv.classList.add('hidden');
                }
            });

            textArea.addEventListener('input', () => {
                if (placeholderDiv && textArea.value.length > 0) {
                    placeholderDiv.classList.add('hidden');
                }
            });

            // Show placeholder again if textarea becomes empty and unfocused
            textArea.addEventListener('blur', () => {
                if (placeholderDiv && textArea.value.length === 0) {
                    placeholderDiv.classList.remove('hidden');
                    currentCharIndex = 0;
                    isHolding = false;
                    setTimeout(typeWriter, 100);
                }
            });
        }

        // Start setup
        setupAnimatedPlaceholder();
    })();
</script>
""", unsafe_allow_html=True)

# Search button
if st.button("🔍 Search Flights", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("Please describe your flight needs")
    else:
        # Clear previous selections when starting new search
        st.session_state.selected_flights = []
        st.session_state.all_flights = []
        st.session_state.selected_return_flights = []
        st.session_state.all_return_flights = []
        st.session_state.has_return = False
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

                st.info("✈️ Searching outbound flights from Amadeus API...")

                # Get departure dates (list)
                departure_dates = parsed.get('departure_dates', [])
                if not departure_dates:
                    st.error("No departure dates found. Please specify when you want to fly.")
                    st.stop()

                for origin_code in parsed['origins'][:1]:  # Limit to first origin for speed
                    for dest_code in parsed['destinations'][:1]:  # Limit to first dest
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
                            results = amadeus.search_flights(
                                origin=origin,
                                destination=dest,
                                departure_date=departure_date,
                                adults=1,
                                max_results=250  # Get ALL available flights (increased from 50)
                            )

                            # Debug: show raw results
                            with st.expander(f"🔍 Debug: Amadeus API Response ({origin}→{dest} on {departure_date})"):
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

                        # If return flight requested, search return flights for ALL return dates
                        if has_return:
                            return_dates = parsed.get('return_dates', [])
                            # Fallback to single return_date for backward compatibility
                            if not return_dates and parsed.get('return_date'):
                                return_dates = [parsed.get('return_date')]

                            for return_date in return_dates:
                                st.info(f"✈️ Searching return flights: {dest} → {origin} on {return_date}")

                                return_results = amadeus.search_flights(
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
                                    all_return_flights.append(flight_info)

                if not all_flights:
                    st.error("No outbound flights found. Try different dates or airports.")
                    st.stop()

                # Look up airline names using Amadeus API
                all_airline_codes = [f['airline'] for f in all_flights]
                if has_return and all_return_flights:
                    all_airline_codes.extend([f['airline'] for f in all_return_flights])

                unique_airlines = list(set(all_airline_codes))
                airline_name_map = amadeus.get_airline_names(unique_airlines)
                st.session_state.airline_names = airline_name_map

                # Store all flights (no algorithm ranking)
                st.session_state.all_flights = all_flights
                st.session_state.all_return_flights = all_return_flights

                if has_return:
                    st.success(f"✅ Found {len(all_flights)} outbound flights and {len(all_return_flights)} return flights!")
                else:
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
    # Calculate submission progress
    has_return = st.session_state.has_return and st.session_state.all_return_flights
    num_required = 2 if has_return else 1
    num_completed = (1 if st.session_state.outbound_submitted else 0) + (1 if st.session_state.return_submitted else 0)
    all_submitted = (num_completed == num_required)

    # Progress bar (simple, no sticky)
    progress_percent = (num_completed / num_required)
    st.progress(progress_percent)
    st.caption(f"{num_completed} / {num_required} submitted")

    st.markdown("---")

    # Check if all submissions are complete
    if all_submitted:
        # COMPLETION SCREEN
        st.success("✅ All rankings submitted successfully!")
        st.markdown("### What would you like to do next?")

        # Create columns based on whether we have return flights
        if has_return:
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2)

        with col1:
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
                st.rerun()

        with col2:
            if st.session_state.csv_data_outbound:
                st.download_button(
                    label="📥 Download Outbound CSV",
                    data=st.session_state.csv_data_outbound,
                    file_name=f"outbound_rankings_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="secondary"
                )

        if has_return:
            with col3:
                if st.session_state.csv_data_return:
                    st.download_button(
                        label="📥 Download Return CSV",
                        data=st.session_state.csv_data_return,
                        file_name=f"return_rankings_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="secondary"
                    )

        if hasattr(st.session_state, 'search_id'):
            st.info(f"Your rankings have been saved (Search ID: {st.session_state.search_id})")

    else:
        # FLIGHT SELECTION INTERFACE
        # Check if we have return flights
        has_return = st.session_state.has_return and st.session_state.all_return_flights

        if has_return:
            st.markdown(f"### ✈️ Found {len(st.session_state.all_flights)} Outbound Flights and {len(st.session_state.all_return_flights)} Return Flights")
            st.markdown("**Select your top 5 flights for EACH direction and drag to rank them →**")
        else:
            st.markdown(f"### ✈️ Found {len(st.session_state.all_flights)} Flights")
            st.markdown("**Select your top 5 flights and drag to rank them →**")

        # CONDITIONAL UI: Show single or dual panels based on has_return
        if has_return:
            # DUAL PANEL LAYOUT: Outbound + Return
            st.markdown("---")
            st.markdown("## 🛫 Outbound Flights")

            col_flights_out, col_ranking_out = st.columns([2, 1])

            with col_flights_out:
                st.markdown("#### All Outbound Flights")

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

                st.markdown("---")

                # Display all outbound flights with checkboxes
                for idx, flight in enumerate(st.session_state.all_flights):
                    unique_id = f"{flight['origin']}_{flight['destination']}{idx + 1}"

                    # Create unique key using ID + departure time to handle duplicate IDs across dates
                    flight_unique_key = f"{flight['id']}_{flight['departure_time']}"
                    is_selected = any(f"{f['id']}_{f['departure_time']}" == flight_unique_key for f in st.session_state.selected_flights)

                    col1, col2 = st.columns([1, 5])

                    with col1:
                        # Use flight unique key for checkbox (sanitized for Streamlit)
                        checkbox_key = f"chk_out_{flight_unique_key}".replace(':', '').replace('-', '').replace('+', '')
                        selected = st.checkbox(
                            "✓" if is_selected else "",
                            value=is_selected,
                            key=checkbox_key,
                            label_visibility="collapsed",
                            disabled=(not is_selected and len(st.session_state.selected_flights) >= 5)
                        )

                        if selected and not is_selected:
                            st.session_state.selected_flights.append(flight)
                            st.rerun()
                        elif not selected and is_selected:
                            st.session_state.selected_flights = [
                                f for f in st.session_state.selected_flights
                                if f"{f['id']}_{f['departure_time']}" != flight_unique_key
                            ]
                            st.rerun()

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

                        st.markdown(f"""
                        <div style="line-height: 1.3; margin: 0; padding: 0.3rem 0;">
                        <strong>{unique_id}</strong> | <strong>{airline_name}</strong> {flight['flight_number']}<br>
                        <span style="font-size: 0.95em;">{flight['origin']} → {flight['destination']} | <strong>{dept_date_display}</strong> | {dept_time_display} - {arr_time_display}</span><br>
                        <span style="font-size: 0.9em; color: #555;">${flight['price']:.0f} | {duration_display} | {flight['stops']} stops</span>
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

                    # Create flight labels WITH RANK NUMBERS
                    flight_labels = []
                    for i, flight in enumerate(st.session_state.selected_flights):
                        airline_name = get_airline_name(flight['airline'])
                        label = f"#{i+1}: {airline_name} {flight['flight_number']} - ${flight['price']:.0f}"
                        flight_labels.append(label)

                    # Sortable list with version AND length in key to refresh on both drag and addition
                    sorted_labels = sort_items(
                        flight_labels,
                        multi_containers=False,
                        direction='vertical',
                        key=f'outbound_sort_v{st.session_state.outbound_sort_version}_n{len(st.session_state.selected_flights)}'
                    )

                    # Update order ONLY if user dragged (same length, different order)
                    if sorted_labels != flight_labels and len(sorted_labels) == len(flight_labels):
                        new_order = []
                        for sorted_label in sorted_labels:
                            # Extract original rank from label
                            rank = int(sorted_label.split(':')[0].replace('#', '')) - 1
                            if rank < len(st.session_state.selected_flights):
                                new_order.append(st.session_state.selected_flights[rank])
                        if len(new_order) == len(st.session_state.selected_flights):
                            st.session_state.selected_flights = new_order
                            st.session_state.outbound_sort_version += 1
                            st.rerun()

                    # Submit button for outbound
                    st.markdown("---")
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

            # RETURN FLIGHTS SECTION
            st.markdown("---")
            st.markdown("## 🛬 Return Flights")

            col_flights_ret, col_ranking_ret = st.columns([2, 1])

            with col_flights_ret:
                st.markdown("#### All Return Flights")

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

                st.markdown("---")

                # Display all return flights with checkboxes
                for idx, flight in enumerate(st.session_state.all_return_flights):
                    unique_id = f"{flight['origin']}_{flight['destination']}{idx + 1}"

                    # Create unique key using ID + departure time to handle duplicate IDs
                    flight_unique_key = f"{flight['id']}_{flight['departure_time']}"
                    is_selected = any(f"{f['id']}_{f['departure_time']}" == flight_unique_key for f in st.session_state.selected_return_flights)

                    col1, col2 = st.columns([1, 5])

                    with col1:
                        # Use flight unique key for checkbox (sanitized for Streamlit)
                        checkbox_key = f"chk_ret_{flight_unique_key}".replace(':', '').replace('-', '').replace('+', '')
                        selected = st.checkbox(
                            "✓" if is_selected else "",
                            value=is_selected,
                            key=checkbox_key,
                            label_visibility="collapsed",
                            disabled=(not is_selected and len(st.session_state.selected_return_flights) >= 5)
                        )

                        if selected and not is_selected:
                            st.session_state.selected_return_flights.append(flight)
                            st.rerun()
                        elif not selected and is_selected:
                            st.session_state.selected_return_flights = [
                                f for f in st.session_state.selected_return_flights
                                if f"{f['id']}_{f['departure_time']}" != flight_unique_key
                            ]
                            st.rerun()

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

                        st.markdown(f"""
                        <div style="line-height: 1.3; margin: 0; padding: 0.3rem 0;">
                        <strong>{unique_id}</strong> | <strong>{airline_name}</strong> {flight['flight_number']}<br>
                        <span style="font-size: 0.95em;">{flight['origin']} → {flight['destination']} | <strong>{dept_date_display}</strong> | {dept_time_display} - {arr_time_display}</span><br>
                        <span style="font-size: 0.9em; color: #555;">${flight['price']:.0f} | {duration_display} | {flight['stops']} stops</span>
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

                    # Create flight labels WITH RANK NUMBERS
                    flight_labels = []
                    for i, flight in enumerate(st.session_state.selected_return_flights):
                        airline_name = get_airline_name(flight['airline'])
                        label = f"#{i+1}: {airline_name} {flight['flight_number']} - ${flight['price']:.0f}"
                        flight_labels.append(label)

                    # Sortable list with version AND length in key to refresh on both drag and addition
                    sorted_labels = sort_items(
                        flight_labels,
                        multi_containers=False,
                        direction='vertical',
                        key=f'return_sort_v{st.session_state.return_sort_version}_n{len(st.session_state.selected_return_flights)}'
                    )

                    # Update order ONLY if user dragged (same length, different order)
                    if sorted_labels != flight_labels and len(sorted_labels) == len(flight_labels):
                        new_order = []
                        for sorted_label in sorted_labels:
                            # Extract original rank from label
                            rank = int(sorted_label.split(':')[0].replace('#', '')) - 1
                            if rank < len(st.session_state.selected_return_flights):
                                new_order.append(st.session_state.selected_return_flights[rank])
                        if len(new_order) == len(st.session_state.selected_return_flights):
                            st.session_state.selected_return_flights = new_order
                            st.session_state.return_sort_version += 1
                            st.rerun()

                    # Submit button for return
                    st.markdown("---")
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
                    from backend.db import save_search_and_csv

                    # Save outbound as primary
                    search_id = save_search_and_csv(
                        session_id=st.session_state.session_id,
                        user_prompt=st.session_state.get('original_prompt', ''),
                        parsed_params=st.session_state.parsed_params or {},
                        all_flights=st.session_state.all_flights,
                        selected_flights=st.session_state.selected_flights,
                        csv_data=st.session_state.csv_data_outbound
                    )

                    st.session_state.csv_generated = True
                    st.session_state.search_id = search_id
                    st.balloons()
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to save rankings: {str(e)}")
                    st.session_state.csv_generated = True
                    st.rerun()

        else:
            # SINGLE PANEL LAYOUT: Outbound only
            col_flights, col_ranking = st.columns([2, 1])

            with col_flights:
                st.markdown("#### All Available Flights")

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

                st.markdown("---")

                # Display all flights with checkboxes
                for idx, flight in enumerate(st.session_state.all_flights):
                    # Generate unique_id for display
                    unique_id = f"{flight['origin']}_{flight['destination']}{idx + 1}"

                    # Create unique key using ID + departure time to handle duplicate IDs across dates
                    flight_unique_key = f"{flight['id']}_{flight['departure_time']}"
                    is_selected = any(f"{f['id']}_{f['departure_time']}" == flight_unique_key for f in st.session_state.selected_flights)

                    col1, col2 = st.columns([1, 5])

                    with col1:
                        # Use flight unique key for checkbox (sanitized for Streamlit)
                        checkbox_key = f"chk_single_{flight_unique_key}".replace(':', '').replace('-', '').replace('+', '')
                        selected = st.checkbox(
                            "✓" if is_selected else "",
                            value=is_selected,
                            key=checkbox_key,
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
                                f for f in st.session_state.selected_flights
                                if f"{f['id']}_{f['departure_time']}" != flight_unique_key
                            ]
                            st.rerun()

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

                        st.markdown(f"""
                        <div style="line-height: 1.3; margin: 0; padding: 0.3rem 0;">
                        <strong>{unique_id}</strong> | <strong>{airline_name}</strong> {flight['flight_number']}<br>
                        <span style="font-size: 0.95em;">{flight['origin']} → {flight['destination']} | <strong>{dept_date_display}</strong> | {dept_time_display} - {arr_time_display}</span><br>
                        <span style="font-size: 0.9em; color: #555;">${flight['price']:.0f} | {duration_display} | {flight['stops']} stops</span>
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

                    # Create flight labels WITH RANK NUMBERS
                    flight_labels = []
                    for i, flight in enumerate(st.session_state.selected_flights):
                        airline_name = get_airline_name(flight['airline'])
                        label = f"#{i+1}: {airline_name} {flight['flight_number']} - ${flight['price']:.0f}"
                        flight_labels.append(label)

                    # Sortable list with version AND length in key to refresh on both drag and addition
                    sorted_labels = sort_items(
                        flight_labels,
                        multi_containers=False,
                        direction='vertical',
                        key=f'single_sort_v{st.session_state.single_sort_version}_n{len(st.session_state.selected_flights)}'
                    )

                    # Update order ONLY if user dragged (same length, different order)
                    if sorted_labels and sorted_labels != flight_labels and len(sorted_labels) == len(flight_labels):
                        new_order = []
                        for sorted_label in sorted_labels:
                            # Extract rank number from label (e.g., "#1: ..." -> 0)
                            rank = int(sorted_label.split(':')[0].replace('#', '')) - 1
                            if rank < len(st.session_state.selected_flights):
                                new_order.append(st.session_state.selected_flights[rank])
                        if len(new_order) == len(st.session_state.selected_flights):
                            st.session_state.selected_flights = new_order
                            st.session_state.single_sort_version += 1
                            st.rerun()

                    # Submit button
                    st.markdown("---")
                    if len(st.session_state.selected_flights) == 5 and not st.session_state.outbound_submitted:
                        if st.button("✅ Submit Rankings", key="submit_single", type="primary", use_container_width=True):
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

                                st.session_state.csv_data_outbound = csv_data
                                st.session_state.outbound_submitted = True
                                st.session_state.csv_generated = True
                                st.session_state.search_id = search_id
                                st.balloons()
                                st.rerun()

                            except Exception as e:
                                st.error(f"Failed to save rankings: {str(e)}")
                                # Still allow CSV download even if DB save fails
                                st.session_state.csv_data_outbound = csv_data
                                st.session_state.outbound_submitted = True
                                st.session_state.csv_generated = True
                                st.rerun()
                    elif st.session_state.outbound_submitted:
                        st.success("✅ Rankings submitted")
                    else:
                        st.info(f"Select {5 - len(st.session_state.selected_flights)} more flights")
                else:
                    st.info("Check boxes on the left to select flights")

# Footer
st.markdown("---")
st.caption("Built for flight ranking research • Data collected for algorithm evaluation")
