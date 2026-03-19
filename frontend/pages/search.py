"""
Search section: How to Use guide, example prompt carousel, search form, and search execution.
"""
import streamlit as st
import streamlit.components.v1 as components

from frontend.styles import TEXTAREA_CSS, PROMPT_SPACING_CSS


CAROUSEL_HTML = r"""
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
    .carousel-prev { left: 10px; }
    .carousel-next { right: 10px; }
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
    .indicator-dot.active { background: rgba(0, 0, 0, 0.35); }
    .indicator-dot:hover { background: rgba(0, 0, 0, 0.25); }
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
    #animBox.scrollable { overflow-y: auto; }
    #animBox.scrollable::-webkit-scrollbar { width: 6px; }
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
    <button class="carousel-nav carousel-prev" onclick="prevPrompt()">&#8249;</button>
    <button class="carousel-nav carousel-next" onclick="nextPrompt()">&#8250;</button>
    <div class="carousel-indicator">
        <span class="indicator-dot active" onclick="goToPrompt(0)"></span>
        <span class="indicator-dot" onclick="goToPrompt(1)"></span>
    </div>
    <button class="toggle-btn" id="toggleBtn" onclick="toggleTypewriter()" title="Pause animation">&#9646;&#9646;</button>
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

    function toggleTypewriter() {
        staticMode = !staticMode;
        autoPlay = false;
        if (typingTimeout) clearTimeout(typingTimeout);
        if (staticMode) {
            toggleBtn.classList.add('static-mode');
            animBox.classList.add('scrollable');
            toggleBtn.innerHTML = '&#9654;';
            toggleBtn.title = 'Enable animation';
            displayText = prompts[idx];
            placeholder.textContent = displayText;
        } else {
            toggleBtn.classList.remove('static-mode');
            animBox.classList.remove('scrollable');
            toggleBtn.innerHTML = '&#9646;&#9646;';
            toggleBtn.title = 'Pause animation';
            charIdx = 0;
            displayText = '';
            typing = true;
            type();
        }
    }

    function updateIndicators() {
        const dots = document.querySelectorAll('.indicator-dot');
        dots.forEach((dot, i) => { dot.classList.toggle('active', i === idx); });
    }

    function prevPrompt() {
        autoPlay = false;
        if (typingTimeout) clearTimeout(typingTimeout);
        placeholder.style.opacity = '0';
        setTimeout(() => {
            idx = (idx - 1 + prompts.length) % prompts.length;
            charIdx = 0; displayText = '';
            placeholder.style.opacity = '1';
            updateIndicators();
            animBox.scrollTop = 0;
            if (staticMode) { displayText = prompts[idx]; placeholder.textContent = displayText; }
            else { typing = true; type(); }
        }, fadeTime);
    }

    function nextPrompt() {
        autoPlay = false;
        if (typingTimeout) clearTimeout(typingTimeout);
        placeholder.style.opacity = '0';
        setTimeout(() => {
            idx = (idx + 1) % prompts.length;
            charIdx = 0; displayText = '';
            placeholder.style.opacity = '1';
            updateIndicators();
            animBox.scrollTop = 0;
            if (staticMode) { displayText = prompts[idx]; placeholder.textContent = displayText; }
            else { typing = true; type(); }
        }, fadeTime);
    }

    function goToPrompt(index) {
        if (index === idx) return;
        autoPlay = false;
        if (typingTimeout) clearTimeout(typingTimeout);
        placeholder.style.opacity = '0';
        setTimeout(() => {
            idx = index; charIdx = 0; displayText = '';
            placeholder.style.opacity = '1';
            updateIndicators();
            animBox.scrollTop = 0;
            if (staticMode) { displayText = prompts[idx]; placeholder.textContent = displayText; }
            else { typing = true; type(); }
        }, fadeTime);
    }

    function trimToFit(text) {
        placeholder.textContent = text;
        if (placeholder.scrollHeight <= animBox.clientHeight) return text;
        let trimmed = text;
        const sentencePattern = /\.\s+/g;
        let match;
        let lastSentenceEnd = 0;
        while (placeholder.scrollHeight > animBox.clientHeight && (match = sentencePattern.exec(text))) {
            lastSentenceEnd = match.index + match[0].length;
            trimmed = text.substring(lastSentenceEnd);
            placeholder.textContent = trimmed;
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
                            charIdx = 0; displayText = ''; typing = true;
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


def render_how_to_use():
    """Render the 'How to Use' section and tips expander."""
    st.markdown('<div id="how-to-use"></div>', unsafe_allow_html=True)
    st.markdown("### 📖 How to Use")

    st.markdown('<div class="hideable-survey-content">', unsafe_allow_html=True)

    st.markdown("""
1. **Describe your flight** - Enter your travel details in natural language, as if you are telling a flight itinerary manager how to book your ideal trip. What would you want them to know?
""")

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
2. **Review results** - Browse all available flights and use the filter sidebar on the left to narrow down options.
3. **Select top 5** - Check the boxes next to your 5 favorite flights
4. **Drag to rank** - Reorder your selections by dragging them in the right panel
5. **Submit** - Click submit to save your rankings
""")


def render_search_section(static_route_day_options, flight_client, static_flights):
    """
    Render the How to Use guide, example prompt carousel, search form, and
    execute search on button press. Updates st.session_state.all_flights.

    Args:
        static_route_day_options: List of "ORIG → DEST (Day)" option strings.
        flight_client: FlightSearchClient instance for AI searches.
        static_flights: List of all static flight dicts.
    """
    from backend.prompt_parser import parse_flight_prompt_with_llm, get_test_api_fallback

    # Once flights are loaded the search is locked — show a read-only summary
    # instead of the full form so participants can't change their route/prompt.
    if st.session_state.get('all_flights'):
        return

    render_how_to_use()

    # CSS for textarea styling
    st.markdown(TEXTAREA_CSS, unsafe_allow_html=True)
    st.markdown(PROMPT_SPACING_CSS, unsafe_allow_html=True)

    # Example prompts carousel
    st.markdown("**Example prompts:**")
    components.html(CAROUSEL_HTML, height=178)

    regular_search = False
    manual_search_btn = False
    manual_prompt = ""
    manual_route = None

    st.markdown("Describe your trip in plain English — origin, destination, dates, and any preferences.")
    with st.form("ai_search_form", clear_on_submit=False):
        ai_prompt_input = st.text_area(
            "Your trip prompt",
            placeholder="e.g. Cheapest flight from New York to LA on March 1st, direct preferred, morning departure",
            height=100,
            key="ai_prompt_input",
            label_visibility="collapsed",
        )
        ai_search_submitted = st.form_submit_button(
            "Search", type="primary",
            use_container_width=True,
        )
    if ai_search_submitted:
        regular_search = True
        st.session_state.auto_search_prompt = ai_prompt_input

    # Close hideable-survey-content container
    st.markdown('</div>', unsafe_allow_html=True)

    # Determine search mode
    auto_search = st.session_state.get('auto_search_requested', False)
    if auto_search:
        st.session_state.auto_search_requested = False

    if manual_search_btn:
        st.session_state.search_mode = "manual"
    elif regular_search or auto_search:
        st.session_state.search_mode = "ai"

    if not (regular_search or manual_search_btn or auto_search):
        return

    # Reset previous results
    st.session_state.all_flights = []
    st.session_state.selected_flights = []
    st.session_state.outbound_submitted = False
    st.session_state.csv_data_outbound = None
    st.session_state.review_confirmed = False

    # Validate inputs
    validation_errors = []
    if st.session_state.search_mode == "manual":
        if not manual_route:
            validation_errors.append("Please select a route")
        if not manual_prompt or not manual_prompt.strip():
            validation_errors.append("Please describe your flight preferences before searching")
    else:
        prompt = st.session_state.get('auto_search_prompt') or ""
        if not prompt.strip():
            validation_errors.append("Please describe your flight needs")

    if validation_errors:
        for error in validation_errors:
            st.error(error)
        return

    # Clear and execute search
    st.session_state.selected_flights = []
    st.session_state.all_flights = []
    st.session_state.csv_generated = False
    st.session_state.outbound_submitted = False
    st.session_state.csv_data_outbound = None
    st.session_state.review_confirmed = False

    with st.spinner("✨ Searching flights..."):
        try:
            if st.session_state.search_mode == "manual":
                _r_origin, _r_rest = manual_route.split(" → ")
                _r_dest, _r_day = _r_rest.split(" (")
                _r_day = _r_day.rstrip(")")

                all_flights = [
                    f for f in static_flights
                    if f["origin"] == _r_origin
                    and f["destination"] == _r_dest
                    and f["day_of_week"] == _r_day
                ]

                st.session_state.original_prompt = (
                    manual_prompt.strip() if manual_prompt and manual_prompt.strip() else ""
                )
                st.session_state.selected_route = manual_route

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**From:** {_r_origin}")
                with col2:
                    st.info(f"**To:** {_r_dest}")
                with col3:
                    st.info(f"**Day:** {_r_day}")

            else:
                # AI mode
                prompt = st.session_state.get('auto_search_prompt', '')
                st.session_state.original_prompt = prompt

                from backend.db import save_prompt_attempt, update_prompt_attempt_result
                prolific_id = st.session_state.get('prolific_id', 'anonymous')

                st.info("🤖 Parsing your request...")
                parsed = parse_flight_prompt_with_llm(prompt)
                st.session_state.parsed_params = parsed

                # Determine result and feedback before saving, so both are stored together
                if not parsed.get('parsed_successfully', True):
                    rejection_msg = parsed.get(
                        'user_message',
                        "Please describe your flight — include where you're flying from, where you're going, and when."
                    )
                    attempt_num = save_prompt_attempt(prolific_id, prompt)
                    update_prompt_attempt_result(prolific_id, attempt_num, passed=False, llm_feedback=rejection_msg)
                    st.warning(rejection_msg)
                    st.stop()

                if not parsed.get('origins') or not parsed.get('destinations'):
                    rejection_msg = "Could not extract origin and destination. Please include both airports or cities in your prompt."
                    attempt_num = save_prompt_attempt(prolific_id, prompt)
                    update_prompt_attempt_result(prolific_id, attempt_num, passed=False, llm_feedback=rejection_msg)
                    st.error(rejection_msg)
                    st.stop()

                attempt_num = save_prompt_attempt(prolific_id, prompt)
                update_prompt_attempt_result(prolific_id, attempt_num, passed=True)

                st.success("✅ Understood your request!")

                departure_dates = parsed.get('departure_dates', [])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**From:** {' or '.join(parsed['origins'])}")
                with col2:
                    st.info(f"**To:** {' or '.join(parsed['destinations'])}")
                with col3:
                    st.info(f"**Depart:** {', '.join(departure_dates) if departure_dates else 'Not specified'}")

                # Save for locked summary display
                st.session_state.selected_route = (
                    f"{' / '.join(parsed['origins'])} → {' / '.join(parsed['destinations'])}"
                    + (f" ({', '.join(departure_dates)})" if departure_dates else "")
                )

                all_flights = []
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
                            flight_offers = results if isinstance(results, list) else results.get('data', [])
                            for offer in flight_offers:
                                flight_info = flight_client.parse_flight_offer(offer)
                                if flight_info:
                                    all_flights.append(flight_info)

            if not all_flights:
                st.error("No flights found. Try different dates or airports.")
                return

            # Look up airline names
            all_airline_codes = [f.get('airline') or f.get('carrier_code') for f in all_flights if f.get('airline') or f.get('carrier_code')]
            airline_name_map = flight_client.get_airline_names(list(set(all_airline_codes)))
            st.session_state.airline_names = airline_name_map

            st.session_state.all_flights = all_flights

            # Save session progress
            if st.session_state.get('token'):
                from backend.db import save_participant_progress
                _save_route_id = 'JFK-LAX-20260301' if st.session_state.search_mode == 'manual' else 'ai'
                save_participant_progress(
                    prolific_id=st.session_state.token,
                    session_id=st.session_state.session_id,
                    prompt=st.session_state.get('original_prompt') or st.session_state.get('user_prompt', ''),
                    route_id=_save_route_id,
                    all_flights=all_flights,
                    study_id=st.session_state.get('study_id'),
                )

            st.success(f"✅ Found {len(all_flights)} flights!")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
