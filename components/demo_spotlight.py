"""
Real spotlight overlay that works with Streamlit.
Injects CSS directly into the page to create darkening effect with cutouts.
"""
import streamlit as st


def show_spotlight_step(step_num, total_steps):
    """
    Show spotlight overlay for a specific tutorial step.
    Uses CSS to darken everything except the highlighted area.
    """

    # Define spotlight steps with element IDs and instructions
    steps = [
        {
            'target_id': 'demo-prompt',
            'title': 'Step 1: Describe Your Flight',
            'description': 'Type your flight preferences here. Be specific about dates, origins, destinations, and what matters to you.'
        },
        {
            'target_id': 'demo-search-btn',
            'title': 'Step 2: Search Flights',
            'description': 'Hit search flights to find available options.'
        },
        {
            'target_id': 'demo-filters',
            'title': 'Step 3: Filter Results',
            'description': 'Use filters to narrow down results by airline, price, time, or stops.'
        },
        {
            'target_id': 'demo-results',
            'title': 'Step 4: View Results',
            'description': 'View your results. Each flight shows price, duration, airline, and times.'
        },
        {
            'target_id': 'demo-checkboxes',
            'title': 'Step 5: Select Flights',
            'description': 'Select flights you like. Choose 5-10 flights to rank.'
        },
        {
            'target_id': 'demo-ranking',
            'title': 'Step 6: Rank Selections',
            'description': 'Drag to rank your selections from best to worst.'
        },
        {
            'target_id': 'demo-submit',
            'title': 'Step 7: Submit Rankings',
            'description': 'Submit your rankings to continue.'
        }
    ]

    if step_num >= len(steps):
        return

    current_step = steps[step_num]

    # Inject spotlight CSS and instruction card
    st.markdown(f"""
    <style>
        /* Dark overlay covering entire page */
        #spotlight-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.75);
            z-index: 9998;
            pointer-events: none;
        }}

        /* Highlighted element gets bright */
        #{current_step['target_id']} {{
            position: relative;
            z-index: 9999 !important;
            box-shadow: 0 0 0 4px #3b82f6, 0 0 0 9999px rgba(0, 0, 0, 0.75) !important;
            border-radius: 8px;
            background: white !important;
        }}

        /* Instruction card */
        #tutorial-card {{
            position: fixed;
            top: 50%;
            right: 40px;
            transform: translateY(-50%);
            background: white;
            border-radius: 16px;
            padding: 28px 32px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
            z-index: 10000;
            max-width: 400px;
            animation: slideIn 0.4s ease;
        }}

        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(-50%) translateX(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(-50%) translateX(0);
            }}
        }}

        #tutorial-card h2 {{
            color: #1f2937;
            font-size: 22px;
            font-weight: 600;
            margin: 0 0 12px 0;
        }}

        #tutorial-card p {{
            color: #4b5563;
            font-size: 16px;
            line-height: 1.6;
            margin: 0 0 20px 0;
        }}

        #tutorial-card .progress {{
            color: #9ca3af;
            font-size: 14px;
            margin: 0;
            padding-top: 16px;
            border-top: 1px solid #e5e7eb;
            text-align: center;
        }}

        /* Animated cursor pointer */
        #demo-cursor {{
            position: fixed;
            width: 24px;
            height: 24px;
            background: #3b82f6;
            border-radius: 50%;
            z-index: 10001;
            pointer-events: none;
            animation: pulse 2s infinite, moveToCursor 0.8s ease;
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.8);
        }}

        @keyframes pulse {{
            0%, 100% {{
                transform: scale(1);
                opacity: 0.9;
            }}
            50% {{
                transform: scale(1.3);
                opacity: 0.6;
            }}
        }}

        /* Disable all interactions except tutorial controls */
        .demo-mode input,
        .demo-mode button:not(.tutorial-btn),
        .demo-mode select,
        .demo-mode textarea {{
            pointer-events: none !important;
            opacity: 0.7;
        }}
    </style>

    <div id="spotlight-overlay"></div>

    <div id="tutorial-card">
        <h2>{current_step['title']}</h2>
        <p>{current_step['description']}</p>
        <p class="progress">Step {step_num + 1} of {total_steps}</p>
    </div>

    <div id="demo-cursor"></div>

    <script>
        // Position cursor near target element
        setTimeout(() => {{
            const target = document.getElementById('{current_step['target_id']}');
            if (target) {{
                const rect = target.getBoundingClientRect();
                const cursor = document.getElementById('demo-cursor');
                if (cursor) {{
                    cursor.style.left = (rect.left + rect.width / 2 - 12) + 'px';
                    cursor.style.top = (rect.top + rect.height / 2 - 12) + 'px';
                }}

                // Scroll element into view
                target.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}, 100);
    </script>
    """, unsafe_allow_html=True)
