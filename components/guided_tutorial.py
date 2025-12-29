"""
State-driven tutorial with visible overlay and highlighting.
"""
import streamlit as st


def show_guided_tutorial(step_num):
    """Show tutorial with overlay and highlighted element."""

    steps = [
        {'id': 'demo-prompt', 'title': '1. Describe Your Flight', 'desc': 'Enter your travel details in natural language. Be specific about what matters to you!'},
        {'id': 'demo-search-btn', 'title': '2. Search for Flights', 'desc': 'Choose Standard Search or AI Personalization to find flights.'},
        {'id': 'demo-filters', 'title': '3. Use Filters', 'desc': 'Narrow down results by airline, price, connections, and more.'},
        {'id': 'demo-results', 'title': '4. Browse Flights', 'desc': 'Review available flights with price, duration, and airline info.'},
        {'id': 'demo-checkboxes', 'title': '5. Select Top 5', 'desc': 'Check boxes next to your 5 favorite flights to rank.'},
        {'id': 'demo-ranking', 'title': '6. Drag to Rank', 'desc': 'Reorder by dragging. #1 is most preferred.'},
        {'id': 'demo-submit', 'title': '7. Submit', 'desc': 'Click Submit to save your rankings.'},
    ]

    if step_num >= len(steps):
        return

    step = steps[step_num]

    # Inject overlay as actual HTML div + CSS for highlighting
    st.markdown(f"""
    <div id="tutorial-overlay" style="
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        z-index: 998;
        pointer-events: none;
    "></div>

    <style>
        /* Highlighted element above overlay */
        #{step['id']} {{
            position: relative !important;
            z-index: 999 !important;
            box-shadow: 0 0 0 4px #4CAF50 !important;
            border-radius: 8px !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Tutorial card in sidebar
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 24px;
            border-radius: 12px;
            margin: 20px 0;
        ">
        """, unsafe_allow_html=True)

        st.markdown(f"### {step['title']}")
        st.write(step['desc'])
        st.caption(f"Step {step_num + 1} of {len(steps)}")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Exit", key=f"exit_{step_num}"):
                st.session_state.demo_active = False
                st.session_state.demo_step = 0
                st.rerun()

        with col2:
            if step_num > 0:
                if st.button("Back", key=f"back_{step_num}"):
                    st.session_state.demo_step -= 1
                    st.rerun()

        with col3:
            if step_num < len(steps) - 1:
                if st.button("Next", key=f"next_{step_num}", type="primary"):
                    st.session_state.demo_step += 1
                    st.rerun()
            else:
                if st.button("Finish", key=f"finish_{step_num}", type="primary"):
                    st.session_state.demo_active = False
                    st.session_state.demo_step = 0
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
