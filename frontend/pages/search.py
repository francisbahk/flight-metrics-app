"""
Search section: How to Use guide, example prompt carousel, search form, and search execution.
"""
import re
from datetime import date, timedelta
from html import escape as _esc
import streamlit as st
import streamlit.components.v1 as components

from streamlit_searchbox import st_searchbox
from frontend.styles import TEXTAREA_CSS, PROMPT_SPACING_CSS
from frontend.utils import remove_codeshares
from backend.utils.airport_search import search_cities, get_airports_for_city, get_countries, get_regions


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
3. **Select flights to rank** - If you see 20 or fewer flights, check all of them. If you see more than 20, check your top 20.
4. **Drag to rank** - Reorder your selections by dragging them in the right panel
5. **Submit** - Click submit to save your rankings
""")


IMESSAGE_CSS = """
<style>
.imsg-container {
    border: 1.5px solid #c7c7cc;
    border-radius: 16px;
    padding: 14px 12px 8px 12px;
    background: #fff;
    margin-bottom: 10px;
}
.imsg-row {
    display: flex;
    margin-bottom: 6px;
    align-items: flex-end;
}
.imsg-bot-row { justify-content: flex-start; }
.imsg-user-row { justify-content: flex-end; }
.imsg-bubble {
    max-width: 78%;
    padding: 8px 12px;
    border-radius: 18px;
    font-size: 0.88em;
    line-height: 1.45;
    word-wrap: break-word;
}
.imsg-bot {
    background: #e9e9eb;
    color: #1c1c1e;
    border-bottom-left-radius: 4px;
}
.imsg-user {
    background: #0a84ff;
    color: #fff;
    border-bottom-right-radius: 4px;
}
</style>
"""

MIC_BTN_HTML = """
<button id="voiceBtn" onclick="toggleVoice()" title="Voice input" style="
    background:#f0f2f6;border:1px solid #ddd;border-radius:50%;
    width:38px;height:38px;cursor:pointer;font-size:16px;
    display:flex;align-items:center;justify-content:center;
    transition:all 0.2s;">🎙️</button>
<script>
let _rec = null, _isRec = false;
function setVal(el, v) {
  const s = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
  s.call(el, v); el.dispatchEvent(new Event('input', {bubbles:true}));
}
function toggleVoice() { _isRec ? stopRec() : startRec(); }
function startRec() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert('Speech recognition not supported in this browser'); return; }
  _rec = new SR(); _rec.continuous=true; _rec.interimResults=true; _rec.lang='en-US';
  const btn=document.getElementById('voiceBtn');
  _rec.onstart=()=>{ _isRec=true; btn.style.background='#ffebee'; btn.style.borderColor='#ef5350'; btn.textContent='⏹️'; };
  _rec.onresult=(e)=>{
    const ta=window.parent.document.querySelector('[data-testid="stTextArea"] textarea');
    if(!ta) return;
    let fin='';
    for(let i=e.resultIndex;i<e.results.length;i++) if(e.results[i].isFinal) fin+=e.results[i][0].transcript;
    if(fin) setVal(ta, (ta.value?ta.value+' ':'')+fin);
  };
  _rec.onerror=(e)=>{ alert('Voice error: '+e.error); stopRec(); };
  _rec.onend=()=>{ if(_isRec) _rec.start(); };
  _rec.start();
}
function stopRec() {
  if(_rec){ _isRec=false; _rec.stop(); _rec=null; }
  const btn=document.getElementById('voiceBtn');
  btn.style.background='#f0f2f6'; btn.style.borderColor='#ddd'; btn.textContent='🎙️';
}
</script>
"""


def _save_full_chat_history(prolific_id: str, attempt_num: int):
    try:
        from backend.db import save_chat_history
        save_chat_history(prolific_id, attempt_num, st.session_state.search_chat_messages)
    except Exception:
        pass


def _md_to_html(text: str) -> str:
    """Convert basic markdown (bold/italic) to HTML for bubble rendering."""
    text = _esc(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    return text


def _execute_chat_search(prompt: str, flight_client):
    """Run LLM parse + Amadeus search. Appends bot messages to search_chat_messages. Sets all_flights on success."""
    from backend.prompt_parser import parse_flight_prompt_with_llm, get_test_api_fallback
    from backend.db import save_prompt_attempt, update_prompt_attempt_result

    prolific_id = st.session_state.get('prolific_id', 'anonymous')

    def bot(text):
        st.session_state.search_chat_messages.append({'role': 'bot', 'text': text})

    # LLM parse
    try:
        parsed = parse_flight_prompt_with_llm(prompt)
    except Exception as e:
        bot("Something went wrong processing your request. Please try again.")
        return

    if not parsed.get('parsed_successfully', True):
        rejection = parsed.get('user_message',
            "Please describe your flight — include where you're flying from, where you're going, and when.")
        attempt_num = save_prompt_attempt(prolific_id, prompt)
        update_prompt_attempt_result(prolific_id, attempt_num, passed=False, llm_feedback=rejection)
        bot(rejection)
        _save_full_chat_history(prolific_id, attempt_num)
        return

    if not parsed.get('origins') or not parsed.get('destinations'):
        rejection = "Couldn't find an origin and destination. Please include both in your prompt."
        attempt_num = save_prompt_attempt(prolific_id, prompt)
        update_prompt_attempt_result(prolific_id, attempt_num, passed=False, llm_feedback=rejection)
        bot(rejection)
        _save_full_chat_history(prolific_id, attempt_num)
        return

    if not parsed.get('departure_dates'):
        rejection = "No departure date found. Please include when you want to fly."
        attempt_num = save_prompt_attempt(prolific_id, prompt)
        update_prompt_attempt_result(prolific_id, attempt_num, passed=False, llm_feedback=rejection)
        bot(rejection)
        _save_full_chat_history(prolific_id, attempt_num)
        return

    origins = parsed['origins']
    dests = parsed['destinations']
    dates = parsed['departure_dates']

    attempt_num = save_prompt_attempt(prolific_id, prompt)
    update_prompt_attempt_result(prolific_id, attempt_num, passed=True)
    _save_full_chat_history(prolific_id, attempt_num)

    bot(f"Got it! Searching **{' / '.join(origins)} → {' / '.join(dests)}** on **{', '.join(dates)}**...")

    # Amadeus search
    all_flights = []
    warnings_list = []
    for origin_code in origins:
        for dest_code in dests:
            origin, ow = get_test_api_fallback(origin_code)
            dest, dw = get_test_api_fallback(dest_code)
            if ow: warnings_list.append(ow)
            if dw: warnings_list.append(dw)
            for date in dates:
                try:
                    results = flight_client.search_flights(
                        origin=origin, destination=dest,
                        departure_date=date, adults=1, max_results=250
                    )
                    offers = results if isinstance(results, list) else results.get('data', [])
                    for offer in offers:
                        fi = flight_client.parse_flight_offer(offer)
                        if fi:
                            all_flights.append(fi)
                except Exception as e:
                    warnings_list.append(f"Search error for {origin}→{dest} on {date}: {str(e)}")

    if not all_flights:
        bot("No flights found for that route and date. Try different dates or airports.")
        return

    all_flights = remove_codeshares(all_flights)
    airline_codes = list(set(
        f.get('airline') or f.get('carrier_code')
        for f in all_flights if f.get('airline') or f.get('carrier_code')
    ))
    st.session_state.airline_names = flight_client.get_airline_names(airline_codes)
    st.session_state.original_prompt = prompt
    st.session_state.all_flights = all_flights
    st.session_state.selected_route = (
        f"{' / '.join(origins)} → {' / '.join(dests)}"
        + (f" ({', '.join(dates)})" if dates else "")
    )
    st.session_state.parsed_params = parsed
    st.session_state.search_mode = "ai"

    if st.session_state.get('token'):
        from backend.db import save_participant_progress
        save_participant_progress(
            prolific_id=st.session_state.token,
            session_id=st.session_state.session_id,
            prompt=prompt,
            route_id='ai',
            all_flights=all_flights,
            study_id=st.session_state.get('study_id'),
        )

    for w in warnings_list:
        bot(f"Note: {w}")

    bot(f"Found **{len(all_flights)} flights**! Scroll down to browse and rank them.")
    _save_full_chat_history(prolific_id, attempt_num)


def render_search_section(static_route_day_options, flight_client, static_flights):
    """
    Render the How to Use guide, example prompt carousel, chat search interface,
    and execute search on user message. Updates st.session_state.all_flights.
    """
    # Once flights are loaded the search is locked.
    if st.session_state.get('all_flights'):
        return

    render_how_to_use()

    st.markdown(TEXTAREA_CSS, unsafe_allow_html=True)
    st.markdown(IMESSAGE_CSS, unsafe_allow_html=True)

    # Example prompts carousel
    st.markdown("**Example prompts:**")
    components.html(CAROUSEL_HTML, height=178)

    st.markdown("---")

    # ── Optional location filters ────────────────────────────────────
    st.markdown("**Select origin and destination:**")
    st.caption("Search by city name or state/region (e.g. \"New York\", \"California\", \"London\"). "
               "If your city isn't listed, try searching by state or region to find nearby airports.")
    with st.expander("Filter by country / region (optional)"):
        countries = [""] + get_countries()
        fc1, fc2 = st.columns(2)
        with fc1:
            origin_country = st.selectbox(
                "Origin country", countries, index=0,
                format_func=lambda x: "All countries" if x == "" else x,
                key="origin_country_filter",
            )
        with fc2:
            dest_country = st.selectbox(
                "Destination country", countries, index=0,
                format_func=lambda x: "All countries" if x == "" else x,
                key="dest_country_filter",
            )

        origin_regions = [""] + get_regions(origin_country or None)
        dest_regions = [""] + get_regions(dest_country or None)
        fr1, fr2 = st.columns(2)
        with fr1:
            origin_region = st.selectbox(
                "Origin region", origin_regions, index=0,
                format_func=lambda x: "All regions" if x == "" else x,
                key="origin_region_filter",
            )
        with fr2:
            dest_region = st.selectbox(
                "Destination region", dest_regions, index=0,
                format_func=lambda x: "All regions" if x == "" else x,
                key="dest_region_filter",
            )

    # ── City searchboxes with filters applied ─────────────────────
    def _origin_search(query: str) -> list[tuple[str, str]]:
        return search_cities(
            query,
            country=origin_country or None,
            region=origin_region or None,
        )

    def _dest_search(query: str) -> list[tuple[str, str]]:
        return search_cities(
            query,
            country=dest_country or None,
            region=dest_region or None,
        )

    # Build unique keys so searchbox resets when filters change
    origin_filter_tag = f"{origin_country}_{origin_region}"
    dest_filter_tag = f"{dest_country}_{dest_region}"

    # Pre-populate dropdown when filters are set
    origin_has_filter = bool(origin_country or origin_region)
    dest_has_filter = bool(dest_country or dest_region)
    origin_defaults = search_cities("", country=origin_country or None, region=origin_region or None) if origin_has_filter else None
    dest_defaults = search_cities("", country=dest_country or None, region=dest_region or None) if dest_has_filter else None

    col_origin, col_dest = st.columns(2)
    with col_origin:
        origin_selection = st_searchbox(
            _origin_search,
            key=f"origin_searchbox_{origin_filter_tag}",
            placeholder="Enter a city…",
            label="Origin",
            clear_on_submit=False,
            default_options=origin_defaults,
        )
    with col_dest:
        dest_selection = st_searchbox(
            _dest_search,
            key=f"dest_searchbox_{dest_filter_tag}",
            placeholder="Enter a city…",
            label="Destination",
            clear_on_submit=False,
            default_options=dest_defaults,
        )

    # Show airports for the selected cities
    if origin_selection:
        airports = get_airports_for_city(origin_selection)
        st.session_state["origin_airports"] = airports
        labels = [
            a["iata"] if a["distance_mi"] == 0 else f"{a['iata']} ({a['distance_mi']:.0f} mi)"
            for a in airports
        ]
        st.caption(f"Airports: {', '.join(labels)}")

    if dest_selection:
        airports = get_airports_for_city(dest_selection)
        st.session_state["dest_airports"] = airports
        labels = [
            a["iata"] if a["distance_mi"] == 0 else f"{a['iata']} ({a['distance_mi']:.0f} mi)"
            for a in airports
        ]
        st.caption(f"Airports: {', '.join(labels)}")

    # ── Date selection ────────────────────────────────────────────
    SEARCH_YEAR = 2026
    min_date = max(date(SEARCH_YEAR, 1, 1), date.today() + timedelta(days=7))
    max_date = date(SEARCH_YEAR, 12, 31)

    st.markdown("**Select travel dates:**")
    st.caption("Pick a start and end date (up to 7 consecutive days). "
               "All dates between them will be searched.")
    d1, d2 = st.columns(2)
    with d1:
        start_date = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="search_start_date",
        )
    with d2:
        end_min = start_date if start_date else min_date
        end_max = min(end_min + timedelta(days=6), max_date)
        end_default = min(end_min + timedelta(days=2), end_max)
        end_date = st.date_input(
            "End date",
            value=end_default,
            min_value=end_min,
            max_value=end_max,
            key="search_end_date",
        )

    if start_date and end_date and start_date <= end_date:
        num_days = (end_date - start_date).days + 1
        travel_dates = [start_date + timedelta(days=i) for i in range(num_days)]
        st.session_state["travel_dates"] = [d.isoformat() for d in travel_dates]
        date_strs = [d.strftime("%b %d") for d in travel_dates]
        st.caption(f"Searching {num_days} date{'s' if num_days > 1 else ''}: {', '.join(date_strs)}")

    st.markdown("---")

    # Initialize chat history with opening bot message (tip merged in)
    if 'search_chat_messages' not in st.session_state:
        st.session_state.search_chat_messages = [{
            'role': 'bot',
            'text': (
                "Describe your flight — where are you flying from, where are you going, "
                "and when? Feel free to add any preferences (price, stops, airline, times, etc.). "
                "💡 **Tip:** Include all nearby airports explicitly (e.g. \"PIE\" or \"SRQ\") if needed."
            ),
        }]

    # If there's a pending search, run it first
    if st.session_state.get('pending_chat_search'):
        prompt = st.session_state.pop('pending_chat_search')
        with st.spinner("Searching for flights..."):
            _execute_chat_search(prompt, flight_client)
        st.rerun()

    # Render iMessage-style chat bubbles
    bubbles_html = '<div class="imsg-container">'
    for msg in st.session_state.search_chat_messages:
        if msg['role'] == 'bot':
            bubbles_html += (
                f'<div class="imsg-row imsg-bot-row">'
                f'<div class="imsg-bubble imsg-bot">{_md_to_html(msg["text"])}</div>'
                f'</div>'
            )
        else:
            bubbles_html += (
                f'<div class="imsg-row imsg-user-row">'
                f'<div class="imsg-bubble imsg-user">{_esc(msg["text"])}</div>'
                f'</div>'
            )
    bubbles_html += '</div>'
    st.markdown(bubbles_html, unsafe_allow_html=True)

    # Input form with textarea + [mic | Send →]
    with st.form("chat_input_form", clear_on_submit=True):
        user_input = st.text_area(
            "",
            placeholder="Describe your trip — origin, destination, date, and any preferences...",
            label_visibility="collapsed",
            height=90,
        )
        col_mic, col_send = st.columns([1, 6])
        with col_mic:
            components.html(MIC_BTN_HTML, height=42)
        with col_send:
            submitted = st.form_submit_button("Send →", use_container_width=True, type="primary")

    if submitted and user_input and user_input.strip():
        st.session_state.all_flights = []
        st.session_state.selected_flights = []
        st.session_state.outbound_submitted = False
        st.session_state.review_confirmed = False
        text = user_input.strip()
        st.session_state.search_chat_messages.append({'role': 'user', 'text': text})
        st.session_state.pending_chat_search = text
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
