"""
Guided walkthrough component for the flight ranking app.
Uses Driver.js for smooth, professional onboarding experience.
"""
import streamlit as st
import streamlit.components.v1 as components


def create_tour_html(tour_steps, auto_start=False):
    """
    Creates HTML/JS for guided tour using Driver.js.

    Args:
        tour_steps: List of dicts with 'element', 'title', 'description'
        auto_start: Whether to start tour automatically
    """
    # Convert tour steps to JavaScript
    steps_js = "[\n"
    for step in tour_steps:
        element = step.get('element', '')
        title = step.get('title', '').replace("'", "\\'")
        description = step.get('description', '').replace("'", "\\'")
        popover_class = step.get('popoverClass', 'tour-popover')

        steps_js += f"""    {{
        element: '{element}',
        popover: {{
            title: '{title}',
            description: '{description}',
            popoverClass: '{popover_class}',
            side: 'bottom',
            align: 'start'
        }}
    }},\n"""
    steps_js += "]"

    auto_start_js = "true" if auto_start else "false"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.css"/>
        <style>
            .driver-popover {{
                max-width: 400px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }}

            .driver-popover-title {{
                font-size: 18px;
                font-weight: 600;
                color: #1f2937;
                margin-bottom: 8px;
            }}

            .driver-popover-description {{
                font-size: 14px;
                line-height: 1.6;
                color: #4b5563;
            }}

            .driver-popover-footer {{
                margin-top: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .driver-popover-prev-btn,
            .driver-popover-next-btn,
            .driver-popover-close-btn {{
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
            }}

            .driver-popover-next-btn {{
                background: #3b82f6;
                color: white;
                border: none;
            }}

            .driver-popover-next-btn:hover {{
                background: #2563eb;
            }}

            .driver-popover-prev-btn {{
                background: #f3f4f6;
                color: #4b5563;
                border: none;
            }}

            .driver-popover-prev-btn:hover {{
                background: #e5e7eb;
            }}

            .driver-popover-close-btn {{
                background: transparent;
                color: #9ca3af;
                border: 1px solid #e5e7eb;
            }}

            .driver-popover-close-btn:hover {{
                background: #fee;
                color: #ef4444;
                border-color: #ef4444;
            }}

            /* Overlay styling */
            .driver-overlay {{
                background: rgba(0, 0, 0, 0.7);
            }}

            /* Highlighted element styling */
            .driver-highlighted-element {{
                box-shadow: 0 0 0 4px #3b82f6, 0 0 0 8px rgba(59, 130, 246, 0.3);
                border-radius: 8px;
            }}

            /* Progress indicator */
            .driver-popover-progress-text {{
                font-size: 12px;
                color: #9ca3af;
                margin-right: auto;
            }}
        </style>
    </head>
    <body>
        <script src="https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.js.iife.js"></script>
        <script>
            // Wait for page to fully load
            window.addEventListener('load', function() {{
                // Check if tour has been completed
                const tourCompleted = localStorage.getItem('flightAppTourCompleted');
                const autoStart = {auto_start_js};

                // Initialize Driver.js
                const driver = window.driver({{
                    showProgress: true,
                    steps: {steps_js},
                    onDestroyStarted: () => {{
                        // Mark tour as completed when destroyed (closed or finished)
                        localStorage.setItem('flightAppTourCompleted', 'true');
                        driver.destroy();
                    }},
                    onDestroyed: () => {{
                        // Notify parent window that tour is complete
                        if (window.parent) {{
                            window.parent.postMessage({{type: 'tourCompleted'}}, '*');
                        }}
                    }},
                    nextBtnText: 'Next →',
                    prevBtnText: '← Back',
                    doneBtnText: 'Finish!',
                    closeBtnText: 'Skip Tour',
                    progressText: '{{currentStep}} of {{totalSteps}}',
                }});

                // Expose driver globally so it can be triggered from outside
                window.flightTourDriver = driver;

                // Auto-start tour if enabled and not completed
                if (autoStart && !tourCompleted) {{
                    setTimeout(() => {{
                        driver.drive();
                    }}, 500); // Small delay to ensure elements are rendered
                }}

                // Listen for manual trigger from parent
                window.addEventListener('message', function(event) {{
                    if (event.data.type === 'startTour') {{
                        driver.drive();
                    }} else if (event.data.type === 'resetTour') {{
                        localStorage.removeItem('flightAppTourCompleted');
                        driver.drive();
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    return html


def show_guided_tour(tour_steps, auto_start=False, height=0):
    """
    Display guided tour in Streamlit app.

    Args:
        tour_steps: List of tour step configurations
        auto_start: Whether to auto-start the tour
        height: Component height (0 for invisible)
    """
    tour_html = create_tour_html(tour_steps, auto_start)
    components.html(tour_html, height=height, scrolling=False)


def get_flight_app_tour_steps():
    """
    Defines the tour steps for the flight ranking app.
    Uses CSS selectors to target Streamlit elements.
    """
    steps = [
        {
            'element': '[data-testid="stTextInput"]',
            'title': 'Welcome to Flight Ranking! ✈️',
            'description': 'This is where you describe your ideal flight. Tell us where you want to go, when, and what matters to you (price, time, comfort, etc.).'
        },
        {
            'element': '[data-testid="stButton"]',
            'title': 'Search for Flights',
            'description': 'Click this button to search for flights based on your preferences. Our AI will parse your request and find the best options.'
        },
        {
            'element': '.stDataFrame',
            'title': 'Flight Results',
            'description': 'All available flights will appear here. You can see prices, times, airlines, and more. Select the flights you like by checking the boxes.'
        },
        {
            'element': '[data-testid="stCheckbox"]',
            'title': 'Select Flights',
            'description': 'Check the boxes next to flights you\'re interested in. You can select multiple flights to compare and rank them.'
        },
        {
            'element': '.stMarkdown',
            'title': 'Rank Your Selections',
            'description': 'After selecting flights, you\'ll be able to drag and drop them to rank your preferences. This helps us learn what matters most to you.'
        },
        {
            'element': '[data-testid="stSidebar"]',
            'title': 'Navigation & Options',
            'description': 'Use the sidebar to access different features like LILO preference learning, validation mode, and settings.'
        }
    ]
    return steps


def init_tour_trigger():
    """
    Creates a button to manually trigger the tour.
    Returns True if button was clicked.
    """
    if st.sidebar.button('🎓 Show Tutorial', help='Restart the guided walkthrough'):
        return True
    return False


def reset_tour():
    """Reset tour completion status."""
    # This will be handled by JavaScript localStorage
    # We use st.rerun() to refresh and trigger auto-start
    st.rerun()
