"""
Tutorial card with buttons built-in using Streamlit native components.
"""
import streamlit as st


def show_tutorial_card(step_num):
    """Show tutorial card with navigation buttons inside."""

    steps = [
        {
            'id': 'demo-prompt',
            'title': '1. Describe Your Flight',
            'desc': 'Enter your travel details in natural language, as if you\'re telling a flight itinerary manager how to book your ideal trip. Be specific about what matters to you!',
            'tips': '💡 Example: "I want to fly from JFK to LAX on December 26th. I prefer cheap flights but if a flight is longer than 12 hours I\'d prefer to pay a bit more money to take a shorter flight."'
        },
        {
            'id': 'demo-search-btn',
            'title': '2. Search for Flights',
            'desc': 'Choose between Standard Search or AI Personalization. After you submit your prompt, use the filter sidebar on the left to narrow down options by price, airlines, connections, flight duration, and airports.',
            'tips': '💡 AI Search results may be less accurate while we continue to improve the system.'
        },
        {
            'id': 'demo-filters',
            'title': '3. Use Filters',
            'desc': 'Narrow down your flight options using the sidebar filters. Filter by airline, number of connections, price range, flight duration, departure/arrival times, and specific airports.',
            'tips': '💡 Note: All search results may be less accurate while we continue to improve the system.'
        },
        {
            'id': 'demo-results',
            'title': '4. Browse Available Flights',
            'desc': 'Review the search results showing all available flights. Each flight displays price, duration, departure/arrival times, airline, and number of stops.',
            'tips': '💡 Scroll through to see all options before selecting your favorites.'
        },
        {
            'id': 'demo-checkboxes',
            'title': '5. Select Your Top 5 Flights',
            'desc': 'Check the boxes next to your 5 favorite flights (both outbound and return if applicable). These are the flights you want to rank and compare.',
            'tips': '💡 Select exactly 5 flights that best match your preferences.'
        },
        {
            'id': 'demo-ranking',
            'title': '6. Drag to Rank',
            'desc': 'Reorder your selected flights by dragging them in the ranking panel. Put your most preferred flight at #1, second choice at #2, and so on.',
            'tips': '💡 Your rankings help us understand your preferences and improve our flight search tools.'
        },
        {
            'id': 'demo-submit',
            'title': '7. Submit Rankings',
            'desc': 'Click Submit to save your rankings. You can optionally download your selections as a CSV file for your records.',
            'tips': '💡 Thank you for participating in our pilot data-collection study!'
        },
    ]

    if step_num >= len(steps):
        return None

    step = steps[step_num]

    # Add highlighting CSS for the current step - lighter overlay
    st.markdown(f"""
    <style>
        /* Highlight current element with gray outline and lighter overlay */
        #{step['id']} {{
            position: relative !important;
            z-index: 9999 !important;
            box-shadow: 0 0 0 3px #888888, 0 0 0 9999px rgba(0,0,0,0.5) !important;
            border-radius: 8px !important;
        }}
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

    # Render tutorial card in SIDEBAR (always works, no CSS hacks)
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

        # Show tips if available
        if 'tips' in step:
            st.markdown(f"<small style='color: #e0e0e0;'>{step['tips']}</small>", unsafe_allow_html=True)

        st.caption(f"Step {step_num + 1} of {len(steps)}")

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

        st.markdown("</div>", unsafe_allow_html=True)
