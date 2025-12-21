"""
Example of how to integrate the guided tour into app.py

Add this code to your app.py file:
"""

# At the top of app.py, add import:
# from components.guided_tour import show_guided_tour, get_flight_app_tour_steps, init_tour_trigger

# Early in the app (after st.set_page_config), add tour initialization:
def setup_tour():
    """
    Initialize and show guided tour on first visit.
    """
    # Check if this is first visit (use session state)
    if 'tour_shown' not in st.session_state:
        st.session_state.tour_shown = False

    # Manual trigger button in sidebar
    if st.sidebar.button('🎓 Show Tutorial', help='Restart the guided walkthrough'):
        st.session_state.tour_shown = False
        st.rerun()

    # Show tour if not shown yet
    if not st.session_state.tour_shown:
        tour_steps = get_flight_app_tour_steps()
        show_guided_tour(tour_steps, auto_start=True, height=0)
        st.session_state.tour_shown = True


# Example usage in app.py main function:
"""
def main():
    st.set_page_config(page_title="Flight Ranking", layout="wide")

    # Initialize tour (do this early)
    setup_tour()

    # Rest of your app code...
    st.title("Flight Ranking App")
    # ...
"""

# Alternative: Simpler approach with custom tour overlay
# This version is more reliable for Streamlit apps

def show_simple_tour():
    """
    Shows a simple tour using st.info boxes that appear step by step.
    More reliable than Driver.js for Streamlit.
    """
    if 'tour_step' not in st.session_state:
        st.session_state.tour_step = 0

    if 'skip_tour' not in st.session_state:
        st.session_state.skip_tour = False

    # Tour steps
    tour_steps = [
        ("Welcome! ✈️", "Welcome to the Flight Ranking App. This tutorial will show you how to use the app. Click 'Next' to continue or 'Skip' to close this tutorial."),
        ("Enter Your Query", "Start by typing your flight preferences in the text box above. For example: 'I want to fly from NYC to LA on Dec 25th, prefer cheap flights'."),
        ("Search Flights", "Click the search button to find flights matching your criteria. Our AI will parse your request automatically."),
        ("View Results", "Browse through the flight results. You can see all available options with prices, times, and airlines."),
        ("Select & Rank", "Select flights you like by checking boxes, then rank them by dragging. This helps us learn your preferences."),
        ("LILO Mode", "Try LILO mode for personalized recommendations! The system will ask questions to understand what you value most."),
    ]

    # Show current step
    if not st.session_state.skip_tour and st.session_state.tour_step < len(tour_steps):
        title, desc = tour_steps[st.session_state.tour_step]

        with st.container():
            st.info(f"**{title}** ({st.session_state.tour_step + 1}/{len(tour_steps)})\n\n{desc}")

            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("← Back", disabled=st.session_state.tour_step == 0):
                    st.session_state.tour_step -= 1
                    st.rerun()
            with col2:
                if st.session_state.tour_step < len(tour_steps) - 1:
                    if st.button("Next →"):
                        st.session_state.tour_step += 1
                        st.rerun()
                else:
                    if st.button("Finish!"):
                        st.session_state.skip_tour = True
                        st.rerun()
            with col3:
                if st.button("Skip Tour"):
                    st.session_state.skip_tour = True
                    st.rerun()

            st.markdown("---")
