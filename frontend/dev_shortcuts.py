"""
Dev-mode shortcuts — ONLY active when ?dev=true is in the URL.
Never activates in production since participants won't add this param.
"""
import json
import streamlit as st

# ── Mock data ──────────────────────────────────────────────────────────────
MOCK_FLIGHTS = [
    {"id": "AA100", "flight_number": "AA100", "airline": "AA",
     "origin": "JFK", "destination": "LAX",
     "departure_time": "2026-05-01T08:00:00", "arrival_time": "2026-05-01T11:30:00",
     "duration_min": 330, "stops": 0, "price": 250.0,
     "cabin": "ECONOMY", "checked_bags": 1, "layover_airports": []},
    {"id": "DL200", "flight_number": "DL200", "airline": "DL",
     "origin": "JFK", "destination": "LAX",
     "departure_time": "2026-05-01T10:00:00", "arrival_time": "2026-05-01T13:45:00",
     "duration_min": 345, "stops": 0, "price": 189.0,
     "cabin": "ECONOMY", "checked_bags": 0, "layover_airports": []},
    {"id": "UA300", "flight_number": "UA300", "airline": "UA",
     "origin": "EWR", "destination": "LAX",
     "departure_time": "2026-05-01T07:30:00", "arrival_time": "2026-05-01T14:00:00",
     "duration_min": 390, "stops": 1, "price": 142.0,
     "cabin": "ECONOMY", "checked_bags": 0, "layover_airports": ["ORD"]},
    {"id": "B6400", "flight_number": "B6400", "airline": "B6",
     "origin": "JFK", "destination": "LAX",
     "departure_time": "2026-05-01T14:00:00", "arrival_time": "2026-05-01T17:30:00",
     "duration_min": 330, "stops": 0, "price": 210.0,
     "cabin": "BUSINESS", "checked_bags": 2, "layover_airports": []},
    {"id": "WN500", "flight_number": "WN500", "airline": "WN",
     "origin": "LGA", "destination": "LAX",
     "departure_time": "2026-05-01T06:00:00", "arrival_time": "2026-05-01T13:30:00",
     "duration_min": 450, "stops": 2, "price": 110.0,
     "cabin": "ECONOMY", "checked_bags": 1, "layover_airports": ["DFW", "PHX"]},
]

PAGES = [
    "— no skip —",
    "prolific gate",
    "screening",
    "consent",
    "search",
    "ranking",
    "review / results",
    "cross-validation",
    "post-survey",
]

_DEV_PID = "DEV_TEST_USER"


def is_dev_mode() -> bool:
    return st.query_params.get("dev", "") == "true"


def inject_dev_base_state():
    """Bypass prolific gate, screening, and consent in dev mode."""
    if not st.session_state.get('prolific_id'):
        st.session_state.prolific_id = _DEV_PID
    if not st.session_state.get('token'):
        st.session_state.token = _DEV_PID
    st.session_state.token_valid = True
    st.session_state.token_group = "A"
    st.session_state.study_id = "DEV_STUDY"
    st.session_state.screening_completed = True
    st.session_state.consent_given = True


def render_dev_panel():
    """Render dev shortcuts in sidebar. No-op outside dev mode."""
    if not is_dev_mode():
        return
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🛠 Dev Mode")
        page = st.selectbox("Skip to page:", PAGES, key="dev_skip_select")
        if st.button("Go", key="dev_go", type="primary", use_container_width=True):
            if page != "— no skip —":
                _inject_state(page)
            st.rerun()
        if st.button("Reset session", key="dev_reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def _inject_state(page: str):
    """Set session state so the app renders the requested page directly."""
    # Always set the base identity
    st.session_state.prolific_id = _DEV_PID
    st.session_state.token = _DEV_PID
    st.session_state.token_valid = True
    st.session_state.token_group = "A"
    st.session_state.session_id = "dev_session"
    st.session_state.rerank_targets = []
    st.session_state.study_id = "DEV_STUDY"

    if page == "prolific gate":
        # Clear identity so the gate shows
        del st.session_state["prolific_id"]
        return

    st.session_state.screening_completed = True
    if page == "screening":
        return

    st.session_state.consent_given = True
    if page == "consent":
        return

    if page == "search":
        return

    # Pages that need flights loaded
    st.session_state.all_flights = MOCK_FLIGHTS
    st.session_state.original_prompt = "Cheapest direct flight from New York to LA on May 1st"
    st.session_state.selected_route = "JFK-LAX-2026-05-01"

    if page == "ranking":
        return

    # Submit rankings
    st.session_state.selected_flights = MOCK_FLIGHTS[:5]
    st.session_state.outbound_submitted = True
    st.session_state.csv_generated = True
    st.session_state.review_confirmed = True
    st.session_state.search_id = _DEV_PID

    if page == "review / results":
        return

    if page == "cross-validation":
        # Leave cross_validation_completed unset so the CV section renders
        st.session_state.cv_seed_prompt_id = 1
        st.session_state.cv_seed_prompt_data = {
            'id': 1,
            'slot_number': 1,
            'prompt_text': 'Cheapest direct flight from New York City to LA on May 1, morning departure preferred, no connections',
            'flights_json': json.dumps(MOCK_FLIGHTS),
        }
        return

    st.session_state.cross_validation_completed = True
    st.session_state.all_reranks_completed = True

    # post-survey: already past CV
