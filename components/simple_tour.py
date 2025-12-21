"""
Simple, clean guided tour for the flight ranking app.
Uses native Streamlit components for reliability.
"""
import streamlit as st


def init_tour_state():
    """Initialize tour-related session state."""
    if 'tour_active' not in st.session_state:
        st.session_state.tour_active = False
    if 'tour_step' not in st.session_state:
        st.session_state.tour_step = 0
    if 'tour_completed' not in st.session_state:
        st.session_state.tour_completed = False


def get_tour_steps():
    """
    Define all tour steps with titles and descriptions.
    Returns list of (title, description) tuples.
    """
    return [
        (
            "Welcome to Flight Ranker! ✈️",
            """Welcome! This tool helps you find and rank flights based on your preferences.

**What you'll learn:**
- How to search for flights using natural language
- How to select and rank your favorite flights
- How to try advanced personalized recommendations with LILO mode

Click **Next** to continue, or **Skip** to close this tutorial."""
        ),
        (
            "Step 1: Describe Your Flight 📝",
            """Start by describing your ideal flight in the text box below.

**Example:**
*"I want to fly from NYC to Los Angeles on December 25th. I prefer cheap flights but if a flight is longer than 12 hours I'd prefer to pay a bit more money to take a shorter flight."*

**You can mention:**
- Origin and destination (cities or airport codes)
- Travel dates
- Preferences (price, time, comfort, layovers, airlines, etc.)
- Any special requirements

The AI will parse your request automatically!"""
        ),
        (
            "Step 2: Search Buttons 🔍",
            """You'll see two search options:

**🔍 Search Flights** (Regular)
- Quick search with standard results
- Best for straightforward queries

**✨ AI Search** (Advanced)
- Uses AI to refine results based on your preferences
- Takes slightly longer but provides smarter rankings
- Has a cooldown to prevent rate limit issues

Click whichever fits your needs!"""
        ),
        (
            "Step 3: Review Results 📊",
            """After searching, you'll see all available flights in a table.

**What you can do:**
- Sort by price, duration, or other columns
- Check boxes next to flights you like (select 5-10)
- View detailed flight information
- Use filters in the sidebar to narrow results

**Tip:** Select flights across different price/time ranges to help the system learn your preferences!"""
        ),
        (
            "Step 4: Rank Your Selections 🎯",
            """Once you've selected flights, you'll rank them by dragging.

**How it works:**
1. Selected flights appear in a ranking panel
2. Drag flights up/down to order them (1 = best)
3. Click submit when you're satisfied

**Why this matters:**
Your rankings help us understand what you value most in a flight!"""
        ),
        (
            "Step 5: LILO Mode (Optional) 🤖",
            """After your initial ranking, you can try **LILO Preference Learning**.

**What is LILO?**
- An advanced AI system that learns your preferences
- Asks clarifying questions about your choices
- Provides personalized flight recommendations

**How it works:**
1. System shows diverse flight options
2. You rank a few and explain why
3. System asks 3 follow-up questions
4. You get personalized top 10 recommendations

This is optional but helps improve the system!"""
        ),
        (
            "Step 6: Validation & Survey 📋",
            """At the end, you'll:

1. **Cross-Validation Rankings:** Rank a few more flights to validate the system's understanding
2. **Brief Survey:** Share feedback about your experience

**Your data helps:**
- Improve flight recommendation algorithms
- Make travel planning easier for everyone
- Advance AI research

All data is anonymized and used for research purposes only."""
        ),
        (
            "You're Ready! 🚀",
            """That's it! You now know how to use Flight Ranker.

**Quick Tips:**
- Be specific in your flight descriptions
- Select diverse options when ranking
- Use the sidebar for filters and navigation
- Feel free to search multiple times

**Need help?**
Click the **🎓 Show Tutorial** button in the sidebar anytime to see this guide again.

Click **Finish** to start searching for flights!"""
        )
    ]


def show_tour():
    """
    Display the current tour step with navigation.
    Call this function in your main app after page config.
    """
    init_tour_state()

    steps = get_tour_steps()

    if st.session_state.tour_active and st.session_state.tour_step < len(steps):
        title, description = steps[st.session_state.tour_step]

        # Create a prominent container for the tour
        with st.container():
            # Gradient header
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px 30px; border-radius: 12px 12px 0 0; color: white;">
                <h2 style="color: white; margin: 0; font-size: 24px;">📚 {title}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 14px;">
                    Step {st.session_state.tour_step + 1} of {len(steps)}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Content area
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 25px 30px;
                        border-radius: 0 0 12px 12px; border: 2px solid #667eea;
                        border-top: none; margin-bottom: 20px;">
                <div style="color: #2c3e50; line-height: 1.8; font-size: 15px;">
                    {description.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Navigation buttons
            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

            with col1:
                if st.session_state.tour_step > 0:
                    if st.button("← Back", use_container_width=True, key="tour_back"):
                        st.session_state.tour_step -= 1
                        st.rerun()

            with col2:
                if st.session_state.tour_step < len(steps) - 1:
                    if st.button("Next →", use_container_width=True, type="primary", key="tour_next"):
                        st.session_state.tour_step += 1
                        st.rerun()
                else:
                    if st.button("Finish! 🎉", use_container_width=True, type="primary", key="tour_finish"):
                        st.session_state.tour_active = False
                        st.session_state.tour_completed = True
                        st.session_state.tour_step = 0
                        st.success("✅ Tutorial complete! Happy searching!")
                        st.rerun()

            with col3:
                if st.button("Skip Tour", use_container_width=True, key="tour_skip"):
                    st.session_state.tour_active = False
                    st.session_state.tour_completed = True
                    st.session_state.tour_step = 0
                    st.rerun()

            # Divider
            st.markdown("---")


def start_tour():
    """Start or restart the tour."""
    st.session_state.tour_active = True
    st.session_state.tour_step = 0
    st.session_state.tour_completed = False


def check_auto_start(skip_for_demo=True):
    """
    Check if tour should auto-start.

    Args:
        skip_for_demo: If True, don't auto-start for DEMO token users
    """
    init_tour_state()

    # Don't auto-start if already completed
    if st.session_state.tour_completed:
        return

    # Don't auto-start for DEMO token if specified
    if skip_for_demo and st.session_state.get('token', '').upper() == 'DEMO':
        st.session_state.tour_completed = True  # Mark as completed to prevent future auto-start
        return

    # Auto-start if tour hasn't been completed yet
    if not st.session_state.tour_completed:
        start_tour()


def add_tour_button_to_sidebar():
    """Add a button to sidebar to manually trigger tour."""
    init_tour_state()

    st.sidebar.markdown("---")
    if st.sidebar.button("🎓 Show Tutorial", use_container_width=True, help="Restart the guided walkthrough"):
        start_tour()
        st.rerun()
