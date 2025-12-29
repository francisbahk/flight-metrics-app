"""
Interactive guided tutorial with sidebar card and element highlighting.
"""
import streamlit as st


def show_guided_tutorial(step_num):
    """Show guided tutorial with sidebar card and highlighted element."""

    steps = [
        {'id': 'demo-prompt', 'title': '1. Describe Your Flight', 'desc': 'Enter your travel details in natural language, as if you\'re telling a flight itinerary manager how to book your ideal trip. Be specific about what matters to you!'},
        {'id': 'demo-search-btn', 'title': '2. Search for Flights', 'desc': 'Choose between Standard Search or AI Personalization. After you submit your prompt, use the filter sidebar on the left to narrow down options.'},
        {'id': 'demo-filters', 'title': '3. Use Filters', 'desc': 'Narrow down your flight options using the sidebar filters. Filter by airline, connections, price range, duration, times, and airports.'},
        {'id': 'demo-results', 'title': '4. Browse Available Flights', 'desc': 'Review the search results showing all available flights. Each flight displays price, duration, times, airline, and stops.'},
        {'id': 'demo-checkboxes', 'title': '5. Select Your Top 5 Flights', 'desc': 'Check the boxes next to your 5 favorite flights. These are the flights you want to rank and compare.'},
        {'id': 'demo-ranking', 'title': '6. Drag to Rank', 'desc': 'Reorder your selected flights by dragging them. Put your most preferred flight at #1, second choice at #2, and so on.'},
        {'id': 'demo-submit', 'title': '7. Submit Rankings', 'desc': 'Click Submit to save your rankings. You can optionally download your selections as a CSV file.'},
    ]

    if step_num >= len(steps):
        return None

    step = steps[step_num]
    total_steps = len(steps)

    # Add overlay and highlighting CSS
    st.markdown(f"""
    <style>
        /* Dark overlay covering main content area only */
        [data-testid="stAppViewContainer"] > section:first-child::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.75);
            z-index: 999;
            pointer-events: none;
        }}

        /* Highlighted element */
        #{step['id']} {{
            position: relative !important;
            z-index: 1000 !important;
            box-shadow: 0 0 0 4px #4CAF50, 0 0 40px rgba(76, 175, 80, 0.6) !important;
            border-radius: 8px !important;
        }}
    </style>

    <script>
        // Auto-scroll to highlighted element
        setTimeout(() => {{
            const el = document.getElementById('{step['id']}');
            if (el) {{
                el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}, 500);
    </script>
    """, unsafe_allow_html=True)

    # Show tutorial card in SIDEBAR
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 24px;
            margin: 20px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        ">
        """, unsafe_allow_html=True)

        st.markdown(f"### {step['title']}")
        st.markdown(f"{step['desc']}")
        st.caption(f"Step {step_num + 1} of {total_steps}")
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
            if step_num < total_steps - 1:
                if st.button("Next", key=f"next_{step_num}", use_container_width=True, type="primary"):
                    st.session_state.demo_step += 1
                    st.rerun()
            else:
                if st.button("Finish", key=f"finish_{step_num}", use_container_width=True, type="primary"):
                    st.session_state.demo_active = False
                    st.session_state.demo_step = 0
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
