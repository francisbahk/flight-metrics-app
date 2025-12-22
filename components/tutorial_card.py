"""
Tutorial card with buttons built-in using Streamlit native components.
"""
import streamlit as st


def show_tutorial_card(step_num):
    """Show tutorial card with navigation buttons inside."""

    steps = [
        {'id': 'demo-prompt', 'title': 'Enter Flight Preferences', 'desc': 'Type your flight criteria here.'},
        {'id': 'demo-search-btn', 'title': 'Search', 'desc': 'Click to search for flights.'},
        {'id': 'demo-filters', 'title': 'Filter Results', 'desc': 'Use sidebar filters to narrow results.'},
        {'id': 'demo-results', 'title': 'View Results', 'desc': 'Browse available flights.'},
        {'id': 'demo-checkboxes', 'title': 'Select Flights', 'desc': 'Choose flights to compare.'},
        {'id': 'demo-ranking', 'title': 'Rank Flights', 'desc': 'Drag to rank your selections.'},
        {'id': 'demo-submit', 'title': 'Submit', 'desc': 'Click submit when done.'},
    ]

    if step_num >= len(steps):
        return None

    step = steps[step_num]

    # Add highlighting CSS for the current step
    st.markdown(f"""
    <style>
        /* Highlight current element with gray outline */
        #{step['id']} {{
            position: relative !important;
            z-index: 9999 !important;
            box-shadow: 0 0 0 3px #888888, 0 0 0 9999px rgba(0,0,0,0.75) !important;
            border-radius: 8px !important;
        }}

        /* Tutorial card styling */
        [data-testid="stVerticalBlock"] > div:has(> .tutorial-card-content) {{
            position: fixed !important;
            top: 20px !important;
            right: 20px !important;
            z-index: 10001 !important;
            max-width: 350px !important;
            pointer-events: auto !important;
        }}

        .tutorial-card-content {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
            color: white;
        }}

        .tutorial-card-content h3 {{
            color: white !important;
            margin: 0 0 12px 0;
            font-size: 20px;
        }}

        .tutorial-card-content p {{
            color: white !important;
            margin: 0 0 16px 0;
            font-size: 14px;
        }}

        .tutorial-card-content hr {{
            border-color: rgba(255,255,255,0.3);
            margin: 16px 0;
        }}

        /* Make buttons work despite demo-mode */
        .tutorial-card-content button {{
            pointer-events: auto !important;
        }}

        /* Scroll to element script */
        </style>
    <script>
        setTimeout(() => {{
            const el = document.getElementById('{step['id']}');
            if (el) {{
                el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}, 300);
    </script>
    """, unsafe_allow_html=True)

    # Create tutorial card using Streamlit containers
    st.markdown('<div class="tutorial-card-content">', unsafe_allow_html=True)

    st.markdown(f"### {step['title']}")
    st.markdown(f"{step['desc']}")
    st.markdown(f"*Step {step_num + 1} of {len(steps)}*")

    st.markdown("---")

    # Navigation buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Exit", key=f"exit_{step_num}", use_container_width=True):
            st.session_state.demo_active = False
            st.session_state.demo_step = 0
            st.rerun()

    with col2:
        if step_num > 0:
            if st.button("Back", key=f"back_{step_num}", use_container_width=True):
                st.session_state.demo_step -= 1
                st.rerun()
        else:
            st.button("Back", key=f"back_{step_num}", disabled=True, use_container_width=True)

    with col3:
        if step_num < len(steps) - 1:
            if st.button("Next", key=f"next_{step_num}", use_container_width=True, type="primary"):
                st.session_state.demo_step += 1
                st.rerun()
        else:
            if st.button("Finish", key=f"finish_{step_num}", use_container_width=True, type="primary"):
                st.session_state.demo_active = False
                st.session_state.demo_step = 0
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
