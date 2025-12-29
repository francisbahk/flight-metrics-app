"""
Interactive guided tutorial with floating tooltips and element highlighting.
Shows the actual website in a pre-populated state with step-by-step guidance.
"""
import streamlit as st


def show_guided_tutorial(step_num):
    """
    Show guided tutorial with floating tooltip next to highlighted element.
    """

    steps = [
        {'id': 'demo-prompt', 'title': 'Step 1: Describe Your Flight', 'desc': 'Enter your travel details in natural language, as if you\'re telling a flight itinerary manager how to book your ideal trip. Be specific about what matters to you!'},
        {'id': 'demo-search-btn', 'title': 'Step 2: Search for Flights', 'desc': 'Choose between Standard Search or AI Personalization. After you submit your prompt, use the filter sidebar on the left to narrow down options.'},
        {'id': 'demo-filters', 'title': 'Step 3: Use Filters', 'desc': 'Narrow down your flight options using the sidebar filters. Filter by airline, connections, price range, duration, times, and airports.'},
        {'id': 'demo-results', 'title': 'Step 4: Browse Available Flights', 'desc': 'Review the search results showing all available flights. Each flight displays price, duration, times, airline, and stops.'},
        {'id': 'demo-checkboxes', 'title': 'Step 5: Select Your Top 5 Flights', 'desc': 'Check the boxes next to your 5 favorite flights. These are the flights you want to rank and compare.'},
        {'id': 'demo-ranking', 'title': 'Step 6: Drag to Rank', 'desc': 'Reorder your selected flights by dragging them. Put your most preferred flight at #1, second choice at #2, and so on.'},
        {'id': 'demo-submit', 'title': 'Step 7: Submit Rankings', 'desc': 'Click Submit to save your rankings. You can optionally download your selections as a CSV file.'},
    ]

    if step_num >= len(steps):
        return None

    step = steps[step_num]
    total_steps = len(steps)

    # Inject CSS for overlay and highlighting
    st.markdown(f"""
    <style>
        /* Full page dark overlay */
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.75);
            z-index: 999;
            pointer-events: none;
        }}

        /* Highlighted element appears ABOVE overlay */
        #{step['id']} {{
            position: relative !important;
            z-index: 1000 !important;
            box-shadow: 0 0 0 4px #4CAF50, 0 0 40px rgba(76, 175, 80, 0.6) !important;
            border-radius: 8px !important;
        }}

        /* Floating tooltip container */
        .tutorial-tooltip-container {{
            position: fixed !important;
            bottom: 20px;
            right: 20px;
            z-index: 1001 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            max-width: 420px;
            color: white;
        }}

        /* Hide default Streamlit button styling for tutorial buttons */
        .tutorial-tooltip-container button {{
            background: white !important;
            color: #667eea !important;
            border: none !important;
            padding: 10px 16px !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            width: 100% !important;
        }}

        .tutorial-tooltip-container button:hover {{
            background: #f0f0f0 !important;
        }}

        .tutorial-tooltip-container button[kind="secondary"] {{
            background: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
        }}

        .tutorial-tooltip-container button[kind="secondary"]:hover {{
            background: rgba(255, 255, 255, 0.25) !important;
        }}

        .tutorial-tooltip-container .stButton {{
            margin: 0 !important;
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

    # Create tooltip card with Streamlit container
    st.markdown('<div class="tutorial-tooltip-container">', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size: 12px; color: rgba(255,255,255,0.7); margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.2);">
        Step {step_num + 1} of {total_steps}
    </div>
    <h3 style="margin: 0 0 12px 0; font-size: 18px; font-weight: 600; color: white;">{step['title']}</h3>
    <p style="margin: 0 0 20px 0; font-size: 14px; line-height: 1.6; color: rgba(255,255,255,0.95);">{step['desc']}</p>
    """, unsafe_allow_html=True)

    # Navigation buttons inside tooltip
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("❌ Exit", key=f"exit_{step_num}", type="secondary"):
            st.session_state.demo_active = False
            st.session_state.demo_step = 0
            st.rerun()

    with col2:
        if step_num > 0:
            if st.button("⬅️ Back", key=f"back_{step_num}", type="secondary"):
                st.session_state.demo_step -= 1
                st.rerun()
        else:
            st.button("⬅️ Back", key=f"back_{step_num}", disabled=True, type="secondary")

    with col3:
        if step_num < total_steps - 1:
            if st.button("Next ➡️", key=f"next_{step_num}", type="primary"):
                st.session_state.demo_step += 1
                st.rerun()
        else:
            if st.button("✅ Finish", key=f"finish_{step_num}", type="primary"):
                st.session_state.demo_active = False
                st.session_state.demo_step = 0
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
