"""
Tutorial card with buttons built-in using HTML component.
"""
import streamlit as st
import streamlit.components.v1 as components


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

    # HTML component with card and buttons
    card_html = f"""
    <style>
        /* Highlight current element */
        #{step['id']} {{
            position: relative !important;
            z-index: 9999 !important;
            box-shadow: 0 0 0 4px #3b82f6, 0 0 0 9999px rgba(0,0,0,0.8) !important;
            border-radius: 8px !important;
        }}

        /* Tutorial card */
        #tutorial-card {{
            position: fixed;
            top: 50%;
            right: 30px;
            transform: translateY(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
            z-index: 10001;
            max-width: 350px;
            color: white;
            font-family: sans-serif;
        }}

        #tutorial-card h3 {{
            margin: 0 0 12px 0;
            font-size: 18px;
        }}

        #tutorial-card p {{
            margin: 0 0 16px 0;
            font-size: 14px;
            opacity: 0.95;
        }}

        .btn-container {{
            display: flex;
            gap: 8px;
            margin-top: 20px;
            padding-top: 16px;
            border-top: 1px solid rgba(255,255,255,0.3);
        }}

        .btn {{
            flex: 1;
            padding: 10px 16px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            background: white;
            color: #667eea;
        }}

        .btn:hover {{
            opacity: 0.9;
        }}

        .btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
    </style>

    <div id="tutorial-card">
        <h3>{step['title']}</h3>
        <p>{step['desc']}</p>
        <p style="font-size: 13px; opacity: 0.8; margin: 0;">Step {step_num + 1} of 7</p>

        <div class="btn-container">
            <button class="btn" onclick="window.parent.location.href = window.parent.location.pathname + '?tutorial_action=exit'">Exit</button>
            <button class="btn" onclick="window.parent.location.href = window.parent.location.pathname + '?tutorial_action=back'" {'disabled' if step_num == 0 else ''}>Back</button>
            <button class="btn" onclick="window.parent.location.href = window.parent.location.pathname + '?tutorial_action=next'">
                {('Finish' if step_num == 6 else 'Next')}
            </button>
        </div>
    </div>

    <script>
        setTimeout(() => {{
            const el = document.getElementById('{step['id']}');
            if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}, 100);
    </script>
    """

    # Render HTML component and capture button clicks
    result = components.html(card_html, height=0)

    return result
