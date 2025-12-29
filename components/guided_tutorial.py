"""
Interactive guided tutorial with floating tooltips and element highlighting.
Shows the actual website in a pre-populated state with step-by-step guidance.
"""
import streamlit as st


def show_guided_tutorial(step_num):
    """
    Show guided tutorial with floating tooltip next to highlighted element.

    Args:
        step_num: Current step index (0-6)
    """

    steps = [
        {
            'id': 'demo-prompt',
            'title': 'Step 1: Describe Your Flight',
            'desc': 'Enter your travel details in natural language, as if you\'re telling a flight itinerary manager how to book your ideal trip. Be specific about what matters to you!',
            'position': 'bottom'  # tooltip position relative to element
        },
        {
            'id': 'demo-search-btn',
            'title': 'Step 2: Search for Flights',
            'desc': 'Choose between Standard Search or AI Personalization. After you submit your prompt, use the filter sidebar on the left to narrow down options.',
            'position': 'bottom'
        },
        {
            'id': 'demo-filters',
            'title': 'Step 3: Use Filters',
            'desc': 'Narrow down your flight options using the sidebar filters. Filter by airline, connections, price range, duration, times, and airports.',
            'position': 'right'
        },
        {
            'id': 'demo-results',
            'title': 'Step 4: Browse Available Flights',
            'desc': 'Review the search results showing all available flights. Each flight displays price, duration, times, airline, and stops.',
            'position': 'left'
        },
        {
            'id': 'demo-checkboxes',
            'title': 'Step 5: Select Your Top 5 Flights',
            'desc': 'Check the boxes next to your 5 favorite flights. These are the flights you want to rank and compare.',
            'position': 'left'
        },
        {
            'id': 'demo-ranking',
            'title': 'Step 6: Drag to Rank',
            'desc': 'Reorder your selected flights by dragging them. Put your most preferred flight at #1, second choice at #2, and so on.',
            'position': 'left'
        },
        {
            'id': 'demo-submit',
            'title': 'Step 7: Submit Rankings',
            'desc': 'Click Submit to save your rankings. You can optionally download your selections as a CSV file.',
            'position': 'bottom'
        },
    ]

    if step_num >= len(steps):
        return None

    step = steps[step_num]

    # Add CSS for highlighting current element and showing tooltip
    st.markdown(f"""
    <style>
        /* Dark overlay for everything except highlighted element */
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            z-index: 9998;
            pointer-events: none;
        }}

        /* Highlight current element */
        #{step['id']} {{
            position: relative !important;
            z-index: 9999 !important;
            box-shadow: 0 0 0 4px #4CAF50, 0 0 30px rgba(76, 175, 80, 0.5) !important;
            border-radius: 8px !important;
            background: white !important;
        }}

        /* Floating tooltip */
        .tutorial-tooltip {{
            position: fixed;
            z-index: 10000;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            max-width: 400px;
            animation: fadeIn 0.3s ease-in;
        }}

        .tutorial-tooltip h3 {{
            margin: 0 0 12px 0;
            font-size: 18px;
            color: white;
        }}

        .tutorial-tooltip p {{
            margin: 0 0 16px 0;
            font-size: 14px;
            line-height: 1.5;
            color: rgba(255, 255, 255, 0.95);
        }}

        .tutorial-tooltip-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .tutorial-step-indicator {{
            font-size: 12px;
            color: rgba(255, 255, 255, 0.8);
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>

    <script>
        // Auto-scroll to highlighted element
        setTimeout(() => {{
            const el = document.getElementById('{step['id']}');
            if (el) {{
                el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}, 300);
    </script>
    """, unsafe_allow_html=True)

    # Render tooltip as a container
    # Position it based on viewport - for now, use bottom-right corner
    tooltip_html = f"""
    <div class="tutorial-tooltip" style="bottom: 20px; right: 20px;">
        <h3>{step['title']}</h3>
        <p>{step['desc']}</p>
        <div class="tutorial-step-indicator">
            Step {step_num + 1} of {len(steps)}
        </div>
    </div>
    """

    st.markdown(tooltip_html, unsafe_allow_html=True)

    # Navigation buttons - render them using Streamlit components
    st.markdown("<div style='position: fixed; bottom: 80px; right: 20px; z-index: 10001;'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("❌ Exit", key=f"exit_tutorial_{step_num}", type="secondary"):
            st.session_state.demo_active = False
            st.session_state.demo_step = 0
            st.rerun()

    with col2:
        if step_num > 0:
            if st.button("⬅️ Back", key=f"back_tutorial_{step_num}"):
                st.session_state.demo_step -= 1
                st.rerun()
        else:
            st.button("⬅️ Back", key=f"back_tutorial_{step_num}", disabled=True)

    with col3:
        if step_num < len(steps) - 1:
            if st.button("Next ➡️", key=f"next_tutorial_{step_num}", type="primary"):
                st.session_state.demo_step += 1
                st.rerun()
        else:
            if st.button("✅ Finish", key=f"finish_tutorial_{step_num}", type="primary"):
                st.session_state.demo_active = False
                st.session_state.demo_step = 0
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
