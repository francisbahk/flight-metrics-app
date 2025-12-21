"""
Interactive Demo/Tutorial Mode with Spotlight Effect
Shows users how to use the flight ranking app with simulated data and guided walkthrough.
"""
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta


def get_fake_flight_data():
    """
    Generate 10 fake flights for JFK→LAX demo.
    Based on prompt: "I want to fly from JFK to LAX on December 25th..."
    """
    fake_flights = [
        {
            'id': 'JFK_LAX1',
            'airline': 'B6',  # JetBlue
            'origin': 'JFK',
            'destination': 'LAX',
            'departure_time': '2024-12-25T08:00:00-05:00',  # 8 AM EST
            'arrival_time': '2024-12-25T11:30:00-08:00',    # 11:30 AM PST
            'duration': '6 hr 30 min',
            'duration_min': 390,
            'stops': 0,
            'price': 289,
            'segments': []
        },
        {
            'id': 'JFK_LAX2',
            'airline': 'DL',  # Delta
            'origin': 'JFK',
            'destination': 'LAX',
            'departure_time': '2024-12-25T10:15:00-05:00',
            'arrival_time': '2024-12-25T13:45:00-08:00',
            'duration': '6 hr 30 min',
            'duration_min': 390,
            'stops': 0,
            'price': 324,
            'segments': []
        },
        {
            'id': 'JFK_LAX3',
            'airline': 'AA',  # American
            'origin': 'JFK',
            'destination': 'LAX',
            'departure_time': '2024-12-25T12:30:00-05:00',
            'arrival_time': '2024-12-25T16:00:00-08:00',
            'duration': '6 hr 30 min',
            'duration_min': 390,
            'stops': 0,
            'price': 356,
            'segments': []
        },
        {
            'id': 'JFK_LAX4',
            'airline': 'UA',  # United
            'origin': 'JFK',
            'destination': 'LAX',
            'departure_time': '2024-12-25T14:45:00-05:00',
            'arrival_time': '2024-12-25T18:15:00-08:00',
            'duration': '6 hr 30 min',
            'duration_min': 390,
            'stops': 0,
            'price': 398,
            'segments': []
        },
        {
            'id': 'JFK_LAX5',
            'airline': 'B6',  # JetBlue
            'origin': 'JFK',
            'destination': 'LAX',
            'departure_time': '2024-12-25T06:00:00-05:00',
            'arrival_time': '2024-12-25T09:30:00-08:00',
            'duration': '6 hr 30 min',
            'duration_min': 390,
            'stops': 0,
            'price': 267,
            'segments': []
        },
        {
            'id': 'JFK_LAX6',
            'airline': 'NK',  # Spirit (budget)
            'origin': 'JFK',
            'destination': 'LAX',
            'departure_time': '2024-12-25T05:30:00-05:00',
            'arrival_time': '2024-12-25T09:00:00-08:00',
            'duration': '6 hr 30 min',
            'duration_min': 390,
            'stops': 0,
            'price': 189,
            'segments': []
        },
        {
            'id': 'JFK_LAX7',
            'airline': 'DL',  # Delta
            'origin': 'JFK',
            'destination': 'LAX',
            'departure_time': '2024-12-25T07:00:00-05:00',
            'arrival_time': '2024-12-25T13:45:00-08:00',
            'duration': '9 hr 45 min',
            'duration_min': 585,
            'stops': 1,
            'price': 245,
            'segments': []
        },
        {
            'id': 'JFK_LAX8',
            'airline': 'AA',  # American
            'origin': 'JFK',
            'destination': 'LAX',
            'departure_time': '2024-12-25T11:00:00-05:00',
            'arrival_time': '2024-12-25T14:30:00-08:00',
            'duration': '6 hr 30 min',
            'duration_min': 390,
            'stops': 0,
            'price': 412,
            'segments': []
        },
        {
            'id': 'JFK_LAX9',
            'airline': 'F9',  # Frontier (budget)
            'origin': 'JFK',
            'destination': 'LAX',
            'departure_time': '2024-12-25T09:15:00-05:00',
            'arrival_time': '2024-12-25T12:45:00-08:00',
            'duration': '6 hr 30 min',
            'duration_min': 390,
            'stops': 0,
            'price': 215,
            'segments': []
        },
        {
            'id': 'JFK_LAX10',
            'airline': 'UA',  # United
            'origin': 'JFK',
            'destination': 'LAX',
            'departure_time': '2024-12-25T16:30:00-05:00',
            'arrival_time': '2024-12-25T20:00:00-08:00',
            'duration': '6 hr 30 min',
            'duration_min': 390,
            'stops': 0,
            'price': 445,
            'segments': []
        },
    ]
    return fake_flights


def get_tutorial_steps():
    """
    Define all tutorial steps with element selectors and descriptions.
    Returns list of step configurations.
    """
    steps = [
        {
            'target': '.stTextArea',  # Prompt input
            'title': 'Step 1: Describe Your Flight',
            'description': 'Type your flight preferences here. Be specific about dates, origins, destinations, and what matters to you.',
            'highlight_area': 'prompt'
        },
        {
            'target': '[data-testid="stButton"] button',  # Search button
            'title': 'Step 2: Search Flights',
            'description': 'Hit search flights to find available options.',
            'highlight_area': 'search-button'
        },
        {
            'target': '[data-testid="stSidebar"]',  # Sidebar filters
            'title': 'Step 3: Filter Results',
            'description': 'Use filters to narrow down results by airline, price, time, or stops.',
            'highlight_area': 'filters'
        },
        {
            'target': '.stDataFrame',  # Flight results table
            'title': 'Step 4: View Results',
            'description': 'View your results. Each flight shows price, duration, airline, and times.',
            'highlight_area': 'results-table'
        },
        {
            'target': '[data-testid="stCheckbox"]',  # Checkboxes
            'title': 'Step 5: Select Flights',
            'description': 'Select flights you like. Choose 5-10 flights to rank.',
            'highlight_area': 'checkboxes'
        },
        {
            'target': '.sortable-container',  # Ranking area
            'title': 'Step 6: Rank Selections',
            'description': 'Drag to rank your selections from best to worst.',
            'highlight_area': 'ranking'
        },
        {
            'target': '[data-testid="stButton"]:last-of-type button',  # Submit button
            'title': 'Step 7: Submit Rankings',
            'description': 'Submit your rankings to continue.',
            'highlight_area': 'submit-button'
        },
    ]
    return steps


def create_spotlight_overlay(current_step):
    """
    Creates HTML/CSS/JS for spotlight overlay effect.
    Darkens entire page except highlighted element.

    Args:
        current_step: Current tutorial step configuration
    """
    target_selector = current_step.get('target', 'body')
    title = current_step.get('title', 'Tutorial')
    description = current_step.get('description', '')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            /* Overlay that darkens everything */
            #tutorial-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.75);
                z-index: 9998;
                pointer-events: none;
                transition: opacity 0.3s ease;
            }}

            /* Spotlight cutout (will be positioned dynamically) */
            #spotlight-cutout {{
                position: fixed;
                background: transparent;
                border: 3px solid #3b82f6;
                border-radius: 12px;
                box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.75),
                            0 0 30px rgba(59, 130, 246, 0.5),
                            inset 0 0 30px rgba(59, 130, 246, 0.2);
                z-index: 9999;
                pointer-events: none;
                transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            }}

            /* Floating instruction card */
            #tutorial-card {{
                position: fixed;
                background: white;
                border-radius: 16px;
                padding: 24px 28px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                z-index: 10000;
                max-width: 400px;
                transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            }}

            #tutorial-card h3 {{
                color: #1f2937;
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 12px;
            }}

            #tutorial-card p {{
                color: #4b5563;
                font-size: 15px;
                line-height: 1.6;
                margin-bottom: 20px;
            }}

            #tutorial-card .button-group {{
                display: flex;
                gap: 12px;
                justify-content: flex-end;
            }}

            #tutorial-card button {{
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
                border: none;
            }}

            #tutorial-card .btn-primary {{
                background: #3b82f6;
                color: white;
            }}

            #tutorial-card .btn-primary:hover {{
                background: #2563eb;
                transform: translateY(-1px);
            }}

            #tutorial-card .btn-secondary {{
                background: #f3f4f6;
                color: #4b5563;
            }}

            #tutorial-card .btn-secondary:hover {{
                background: #e5e7eb;
            }}

            #tutorial-card .btn-exit {{
                background: transparent;
                color: #ef4444;
                border: 1px solid #ef4444;
            }}

            #tutorial-card .btn-exit:hover {{
                background: #fef2f2;
            }}

            /* Animated cursor */
            #tutorial-cursor {{
                position: fixed;
                width: 24px;
                height: 24px;
                background: #3b82f6;
                border-radius: 50%;
                z-index: 10001;
                pointer-events: none;
                opacity: 0;
                transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 0 20px rgba(59, 130, 246, 0.6);
            }}

            #tutorial-cursor.clicking {{
                animation: clickPulse 0.6s ease;
            }}

            @keyframes clickPulse {{
                0%, 100% {{
                    transform: scale(1);
                    opacity: 1;
                }}
                50% {{
                    transform: scale(1.5);
                    opacity: 0.7;
                }}
            }}

            /* Progress indicator */
            .progress-dots {{
                display: flex;
                gap: 8px;
                justify-content: center;
                margin-top: 16px;
                padding-top: 16px;
                border-top: 1px solid #e5e7eb;
            }}

            .progress-dot {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #d1d5db;
                transition: all 0.3s;
            }}

            .progress-dot.active {{
                background: #3b82f6;
                transform: scale(1.3);
            }}
        </style>
    </head>
    <body>
        <!-- Spotlight cutout -->
        <div id="spotlight-cutout"></div>

        <!-- Instruction card -->
        <div id="tutorial-card">
            <h3>{title}</h3>
            <p>{description}</p>

            <div class="button-group">
                <button class="btn-exit" onclick="exitTutorial()">Exit Tutorial</button>
                <button class="btn-secondary" onclick="prevStep()" id="btn-prev">Back</button>
                <button class="btn-primary" onclick="nextStep()" id="btn-next">Next</button>
            </div>

            <div class="progress-dots" id="progress-dots"></div>
        </div>

        <!-- Animated cursor -->
        <div id="tutorial-cursor"></div>

        <script>
            // Get parent window's document (Streamlit iframe)
            const targetDoc = window.parent.document;
            const targetSelector = '{target_selector}';

            function updateSpotlight() {{
                const targetElement = targetDoc.querySelector(targetSelector);
                if (!targetElement) {{
                    console.warn('Target element not found:', targetSelector);
                    return;
                }}

                const rect = targetElement.getBoundingClientRect();
                const spotlight = document.getElementById('spotlight-cutout');
                const card = document.getElementById('tutorial-card');
                const cursor = document.getElementById('tutorial-cursor');

                // Position spotlight cutout
                spotlight.style.left = rect.left - 10 + 'px';
                spotlight.style.top = rect.top - 10 + 'px';
                spotlight.style.width = rect.width + 20 + 'px';
                spotlight.style.height = rect.height + 20 + 'px';

                // Position card dynamically (right side if space, otherwise left)
                const spaceOnRight = window.innerWidth - rect.right;
                if (spaceOnRight > 450) {{
                    card.style.left = rect.right + 30 + 'px';
                    card.style.top = rect.top + 'px';
                }} else {{
                    card.style.left = rect.left - 430 + 'px';
                    card.style.top = rect.top + 'px';
                }}

                // Ensure card is visible
                const cardRect = card.getBoundingClientRect();
                if (cardRect.bottom > window.innerHeight) {{
                    card.style.top = window.innerHeight - cardRect.height - 20 + 'px';
                }}
                if (cardRect.top < 0) {{
                    card.style.top = '20px';
                }}

                // Animate cursor to element center
                cursor.style.opacity = '1';
                cursor.style.left = rect.left + rect.width / 2 - 12 + 'px';
                cursor.style.top = rect.top + rect.height / 2 - 12 + 'px';

                // Trigger click animation
                setTimeout(() => {{
                    cursor.classList.add('clicking');
                    setTimeout(() => cursor.classList.remove('clicking'), 600);
                }}, 600);
            }}

            function nextStep() {{
                window.parent.postMessage({{type: 'tutorial-next'}}, '*');
            }}

            function prevStep() {{
                window.parent.postMessage({{type: 'tutorial-prev'}}, '*');
            }}

            function exitTutorial() {{
                window.parent.postMessage({{type: 'tutorial-exit'}}, '*');
            }}

            // Initialize on load
            window.addEventListener('load', () => {{
                setTimeout(updateSpotlight, 100);
            }});

            // Re-position on window resize
            window.addEventListener('resize', updateSpotlight);
        </script>
    </body>
    </html>
    """
    return html


def init_demo_mode():
    """Initialize demo/tutorial mode session state."""
    if 'demo_active' not in st.session_state:
        st.session_state.demo_active = False
    if 'demo_step' not in st.session_state:
        st.session_state.demo_step = 0


def start_demo():
    """Start interactive demo mode with fake data."""
    st.session_state.demo_active = True
    st.session_state.demo_step = 0

    # Load fake data
    st.session_state.all_flights = get_fake_flight_data()
    st.session_state.original_prompt = "I want to fly from JFK to LAX on December 25th. I prefer cheap flights but if a flight is longer than 12 hours I'd prefer to pay a bit more money to take a shorter flight. I also want to arrive in the afternoon since I want to have time to eat dinner"

    # Set demo flags
    st.session_state.parsed_params = {
        'origins': ['JFK'],
        'destinations': ['LAX'],
        'departure_date': '2024-12-25',
        'preferences': {'cheap': True, 'afternoon_arrival': True}
    }


def show_demo_overlay():
    """Display the spotlight overlay for current demo step."""
    if not st.session_state.demo_active:
        return

    steps = get_tutorial_steps()
    current_step_idx = st.session_state.demo_step

    if current_step_idx >= len(steps):
        # Tutorial complete
        st.session_state.demo_active = False
        st.success("🎉 Tutorial complete! You're ready to search for real flights.")
        return

    current_step = steps[current_step_idx]
    overlay_html = create_spotlight_overlay(current_step)

    # Display overlay using components.html
    components.html(overlay_html, height=0, scrolling=False)
