"""
Spotlight effect for tutorial demo - highlights specific elements.
Card positioning is dynamic based on highlighted element location.
"""
import streamlit as st


def show_spotlight_step(step_num, total_steps):
    """
    Show spotlight overlay for a specific tutorial step.

    Args:
        step_num: Current step number (0-indexed)
        total_steps: Total number of steps
    """

    # Define spotlight steps with element IDs and instructions
    steps = [
        {
            'target_id': 'demo-prompt',
            'title': 'Describe Your Flight',
            'description': 'This is where users enter their flight preferences in natural language.',
            'position': 'bottom'  # Card appears below element
        },
        {
            'target_id': 'demo-search-btn',
            'title': 'Search for Flights',
            'description': 'Click the search button to find available flights based on your criteria.',
            'position': 'bottom'
        },
        {
            'target_id': 'demo-filters',
            'title': 'Filter Results',
            'description': 'Use these filters to narrow down results by airline, price, or number of stops.',
            'position': 'right'
        },
        {
            'target_id': 'demo-results',
            'title': 'View Flight Results',
            'description': 'All matching flights appear here with price, duration, and airline info.',
            'position': 'bottom'
        },
        {
            'target_id': 'demo-checkboxes',
            'title': 'Select Flights to Rank',
            'description': 'Check the boxes next to flights you want to compare and rank.',
            'position': 'bottom'
        },
        {
            'target_id': 'demo-ranking',
            'title': 'Rank Your Selections',
            'description': 'Drag flights to rank them from best to worst based on your preferences.',
            'position': 'bottom'
        },
        {
            'target_id': 'demo-submit',
            'title': 'Submit Your Rankings',
            'description': 'Click submit to save your rankings and continue.',
            'position': 'bottom'
        }
    ]

    if step_num >= len(steps):
        return

    current_step = steps[step_num]

    # Inject spotlight CSS with dynamic card positioning
    st.markdown(f"""
    <style>
        /* Darken everything except highlighted element */
        #{current_step['target_id']} {{
            position: relative !important;
            z-index: 9999 !important;
            box-shadow: 0 0 0 4px #3b82f6, 0 0 0 9999px rgba(0, 0, 0, 0.80) !important;
            border-radius: 8px !important;
        }}

        /* Instruction card - dynamically positioned by JavaScript */
        #tutorial-card-{step_num} {{
            position: fixed;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
            z-index: 10001 !important;
            max-width: 350px;
            color: white;
            transition: all 0.3s ease;
        }}

        #tutorial-card-{step_num} h3 {{
            color: white;
            font-size: 18px;
            font-weight: 700;
            margin: 0 0 10px 0;
        }}

        #tutorial-card-{step_num} p {{
            color: rgba(255, 255, 255, 0.95);
            font-size: 14px;
            line-height: 1.6;
            margin: 0 0 16px 0;
        }}

    </style>

    <div id="tutorial-card-{step_num}">
        <h3>{current_step['title']}</h3>
        <p>{current_step['description']}</p>
        <p style="margin: 8px 0 0 0; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.3); font-size: 13px; opacity: 0.8;">Step {step_num + 1} of {total_steps}</p>
    </div>

    <script>
        // Position card dynamically near highlighted element
        function positionCard() {{
            const target = document.getElementById('{current_step['target_id']}');
            const card = document.getElementById('tutorial-card-{step_num}');

            if (target && card) {{
                const rect = target.getBoundingClientRect();
                const cardRect = card.getBoundingClientRect();

                // Position based on step configuration
                if ('{current_step['position']}' === 'bottom') {{
                    card.style.left = Math.max(20, rect.left) + 'px';
                    card.style.top = (rect.bottom + 20) + 'px';
                }} else if ('{current_step['position']}' === 'right') {{
                    card.style.left = (rect.right + 20) + 'px';
                    card.style.top = rect.top + 'px';
                }} else {{
                    // Default: top-right of screen
                    card.style.right = '20px';
                    card.style.top = '20px';
                }}

                // Scroll element into view
                target.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        // Position on load and resize
        setTimeout(positionCard, 100);
        window.addEventListener('resize', positionCard);
    </script>
    """, unsafe_allow_html=True)
