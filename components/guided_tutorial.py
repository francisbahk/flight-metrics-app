"""
Interactive guided tutorial with floating tooltips and element highlighting.
Shows the actual website in a pre-populated state with step-by-step guidance.
"""
import streamlit as st
import streamlit.components.v1 as components


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
        },
        {
            'id': 'demo-search-btn',
            'title': 'Step 2: Search for Flights',
            'desc': 'Choose between Standard Search or AI Personalization. After you submit your prompt, use the filter sidebar on the left to narrow down options.',
        },
        {
            'id': 'demo-filters',
            'title': 'Step 3: Use Filters',
            'desc': 'Narrow down your flight options using the sidebar filters. Filter by airline, connections, price range, duration, times, and airports.',
        },
        {
            'id': 'demo-results',
            'title': 'Step 4: Browse Available Flights',
            'desc': 'Review the search results showing all available flights. Each flight displays price, duration, times, airline, and stops.',
        },
        {
            'id': 'demo-checkboxes',
            'title': 'Step 5: Select Your Top 5 Flights',
            'desc': 'Check the boxes next to your 5 favorite flights. These are the flights you want to rank and compare.',
        },
        {
            'id': 'demo-ranking',
            'title': 'Step 6: Drag to Rank',
            'desc': 'Reorder your selected flights by dragging them. Put your most preferred flight at #1, second choice at #2, and so on.',
        },
        {
            'id': 'demo-submit',
            'title': 'Step 7: Submit Rankings',
            'desc': 'Click Submit to save your rankings. You can optionally download your selections as a CSV file.',
        },
    ]

    if step_num >= len(steps):
        return None

    step = steps[step_num]
    total_steps = len(steps)

    # Create overlay and tooltip with inline buttons
    tutorial_html = f"""
    <style>
        /* Dark overlay - covers entire page */
        #tutorial-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 9998;
            pointer-events: none;
        }}

        /* Highlighted element should appear ABOVE overlay */
        #{step['id']} {{
            position: relative !important;
            z-index: 9999 !important;
            box-shadow: 0 0 0 4px #4CAF50 !important;
            border-radius: 8px !important;
        }}

        /* Floating tooltip card */
        .tutorial-tooltip {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 10000;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            max-width: 420px;
            animation: fadeIn 0.3s ease-in;
        }}

        .tutorial-tooltip h3 {{
            margin: 0 0 12px 0;
            font-size: 18px;
            font-weight: 600;
            color: white;
        }}

        .tutorial-tooltip p {{
            margin: 0 0 16px 0;
            font-size: 14px;
            line-height: 1.6;
            color: rgba(255, 255, 255, 0.95);
        }}

        .tutorial-step-indicator {{
            font-size: 12px;
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 16px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .tutorial-buttons {{
            display: flex;
            gap: 8px;
            margin-top: 16px;
        }}

        .tutorial-btn {{
            flex: 1;
            padding: 10px 16px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .tutorial-btn-exit {{
            background: rgba(255, 255, 255, 0.1);
            color: white;
        }}

        .tutorial-btn-exit:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}

        .tutorial-btn-back {{
            background: rgba(255, 255, 255, 0.15);
            color: white;
        }}

        .tutorial-btn-back:hover {{
            background: rgba(255, 255, 255, 0.25);
        }}

        .tutorial-btn-back:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}

        .tutorial-btn-next {{
            background: white;
            color: #667eea;
            font-weight: 600;
        }}

        .tutorial-btn-next:hover {{
            background: #f0f0f0;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>

    <!-- Dark overlay -->
    <div id="tutorial-overlay"></div>

    <!-- Floating tooltip with built-in buttons -->
    <div class="tutorial-tooltip">
        <div class="tutorial-step-indicator">
            Step {step_num + 1} of {total_steps}
        </div>
        <h3>{step['title']}</h3>
        <p>{step['desc']}</p>
        <div class="tutorial-buttons">
            <button class="tutorial-btn tutorial-btn-exit" onclick="exitTutorial()">❌ Exit</button>
            <button class="tutorial-btn tutorial-btn-back" onclick="prevStep()" {'disabled' if step_num == 0 else ''}>⬅️ Back</button>
            <button class="tutorial-btn tutorial-btn-next" onclick="nextStep()">
                {'✅ Finish' if step_num == total_steps - 1 else 'Next ➡️'}
            </button>
        </div>
    </div>

    <script>
        // Auto-scroll to highlighted element
        setTimeout(() => {{
            const el = document.getElementById('{step['id']}');
            if (el) {{
                el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}, 300);

        // Navigation functions that communicate with Streamlit
        function exitTutorial() {{
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                data: {{ action: 'exit' }}
            }}, '*');
        }}

        function prevStep() {{
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                data: {{ action: 'back' }}
            }}, '*');
        }}

        function nextStep() {{
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                data: {{ action: 'next' }}
            }}, '*');
        }}
    </script>
    """

    # Render the HTML
    action = components.html(tutorial_html, height=0)

    # Handle button actions via session state instead
    # Use regular Streamlit buttons HIDDEN off-screen for actual navigation
    st.markdown('<div style="position: fixed; left: -9999px; visibility: hidden;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Exit", key=f"exit_{step_num}", type="secondary"):
            st.session_state.demo_active = False
            st.session_state.demo_step = 0
            st.rerun()
    with col2:
        if step_num > 0 and st.button("Back", key=f"back_{step_num}"):
            st.session_state.demo_step -= 1
            st.rerun()
    with col3:
        if step_num < total_steps - 1:
            if st.button("Next", key=f"next_{step_num}", type="primary"):
                st.session_state.demo_step += 1
                st.rerun()
        else:
            if st.button("Finish", key=f"finish_{step_num}", type="primary"):
                st.session_state.demo_active = False
                st.session_state.demo_step = 0
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)  # Close hidden buttons container
