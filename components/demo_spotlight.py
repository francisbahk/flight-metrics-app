"""
Simple spotlight for tutorial - just highlights an element and shows a card.
"""
import streamlit as st


def show_spotlight_step(step_num):
    """Show spotlight on the current step's element."""

    steps = [
        {'id': 'demo-prompt', 'title': 'Enter Your Flight Preferences', 'desc': 'Describe your ideal flight here.'},
        {'id': 'demo-search-btn', 'title': 'Search for Flights', 'desc': 'Click to search for available flights.'},
        {'id': 'demo-filters', 'title': 'Use Filters', 'desc': 'Filter results by airline, price, or stops.'},
        {'id': 'demo-results', 'title': 'View Results', 'desc': 'See all available flights matching your criteria.'},
        {'id': 'demo-checkboxes', 'title': 'Select Flights', 'desc': 'Choose flights you want to compare.'},
        {'id': 'demo-ranking', 'title': 'Rank Flights', 'desc': 'Drag to rank your selected flights.'},
        {'id': 'demo-submit', 'title': 'Submit Rankings', 'desc': 'Click submit to save your rankings.'},
    ]

    if step_num >= len(steps):
        return

    step = steps[step_num]

    # CSS to highlight current element
    st.markdown(f"""
    <style>
        /* Highlight the current step */
        #{step['id']} {{
            position: relative !important;
            z-index: 9999 !important;
            box-shadow: 0 0 0 4px #3b82f6, 0 0 0 9999px rgba(0, 0, 0, 0.8) !important;
            border-radius: 8px !important;
        }}

        /* Instruction card */
        .tutorial-card {{
            position: fixed;
            top: 50%;
            right: 30px;
            transform: translateY(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
            z-index: 10001;
            max-width: 350px;
            color: white;
        }}

        .tutorial-card h3 {{
            margin: 0 0 12px 0;
            font-size: 18px;
        }}

        .tutorial-card p {{
            margin: 0 0 16px 0;
            opacity: 0.95;
        }}
    </style>

    <div class="tutorial-card">
        <h3>{step['title']}</h3>
        <p>{step['desc']}</p>
        <p style="font-size: 13px; opacity: 0.8; margin: 0;">Step {step_num + 1} of 7</p>
    </div>

    <script>
        setTimeout(() => {{
            const el = document.getElementById('{step['id']}');
            if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}, 100);
    </script>
    """, unsafe_allow_html=True)
