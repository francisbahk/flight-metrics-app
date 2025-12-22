"""
Tutorial card with buttons built-in using HTML component.
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

    # Inject highlighting CSS and tutorial card directly into the page
    st.markdown(f"""
    <style>
        /* Highlight current element with gray outline */
        #{step['id']} {{
            position: relative !important;
            z-index: 9999 !important;
            box-shadow: 0 0 0 3px #888888, 0 0 0 9999px rgba(0,0,0,0.75) !important;
            border-radius: 8px !important;
        }}

        /* Tutorial card - fixed position on right side */
        .tutorial-card-wrapper {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10001;
            max-width: 350px;
            animation: slideIn 0.3s ease-out;
        }}

        @keyframes slideIn {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}

        .tutorial-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}

        .tutorial-card h3 {{
            margin: 0 0 12px 0;
            font-size: 20px;
            font-weight: 600;
        }}

        .tutorial-card p {{
            margin: 0 0 16px 0;
            font-size: 14px;
            line-height: 1.5;
            opacity: 0.95;
        }}

        .tutorial-step-counter {{
            font-size: 12px;
            opacity: 0.8;
            margin: 0 0 20px 0;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255,255,255,0.3);
        }}

        .tutorial-btn-container {{
            display: flex;
            gap: 8px;
            margin-top: 20px;
        }}

        .tutorial-btn {{
            flex: 1;
            padding: 10px 16px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            background: white;
            color: #667eea;
            transition: all 0.2s;
            text-decoration: none;
            text-align: center;
            display: inline-block;
        }}

        .tutorial-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}

        .tutorial-btn:active {{
            transform: translateY(0);
        }}

        .tutorial-btn.disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            pointer-events: none;
        }}
    </style>

    <div class="tutorial-card-wrapper">
        <div class="tutorial-card">
            <h3>{step['title']}</h3>
            <p>{step['desc']}</p>
            <p class="tutorial-step-counter">Step {step_num + 1} of {len(steps)}</p>

            <div class="tutorial-btn-container">
                <a href="?tutorial_action=exit" class="tutorial-btn">Exit</a>
                <a href="?tutorial_action=back" class="tutorial-btn {'disabled' if step_num == 0 else ''}">Back</a>
                <a href="?tutorial_action=next" class="tutorial-btn">{'Finish' if step_num == len(steps)-1 else 'Next'}</a>
            </div>
        </div>
    </div>

    <script>
        // Scroll to highlighted element
        setTimeout(() => {{
            const el = document.getElementById('{step['id']}');
            if (el) {{
                el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}, 200);
    </script>
    """, unsafe_allow_html=True)
