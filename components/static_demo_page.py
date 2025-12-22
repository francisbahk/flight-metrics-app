"""
Static demo page - displays the actual PDF with highlighting overlays.
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path


def render_static_demo_page(step_num):
    """
    Shows the actual PDF screenshot and highlights specific sections for each tutorial step.
    """

    # Define highlight coordinates for each step (percentages of PDF dimensions)
    highlight_steps = [
        {
            "title": "1. Your flight prompt",
            "coords": {"left": "18%", "top": "52%", "width": "64%", "height": "12%"}
        },
        {
            "title": "2. Search buttons",
            "coords": {"left": "18%", "top": "66%", "width": "64%", "height": "6%"}
        },
        {
            "title": "3. Filters sidebar",
            "coords": {"left": "2%", "top": "15%", "width": "13%", "height": "52%"}
        },
        {
            "title": "4. All Available Flights",
            "coords": {"left": "18%", "top": "76%", "width": "42%", "height": "20%"}
        },
        {
            "title": "5. Select flights (checkboxes)",
            "coords": {"left": "18%", "top": "80%", "width": "42%", "height": "16%"}
        },
        {
            "title": "6. Your Top 5 rankings",
            "coords": {"left": "62%", "top": "76%", "width": "36%", "height": "19%"}
        },
        {
            "title": "7. Submit button",
            "coords": {"left": "62%", "top": "88%", "width": "36%", "height": "4%"}
        },
    ]

    current_step = highlight_steps[min(step_num, len(highlight_steps) - 1)]
    coords = current_step["coords"]

    # Use local PDF file
    pdf_path = Path(__file__).parent.parent / "screencapture-listen-cornell3-streamlit-app-2025-12-21-23_27_16.pdf"

    # Display PDF with st.components
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    # Embed PDF with highlighting overlay using base64
    import base64
    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

    html_content = f"""
    <style>
        .pdf-container {{
            position: relative;
            width: 100%;
            height: 100vh;
            overflow: hidden;
        }}

        .pdf-frame {{
            width: 100%;
            height: 100%;
            border: none;
        }}

        .highlight-overlay {{
            position: absolute;
            border: 4px solid #667eea;
            border-radius: 12px;
            box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.7),
                       0 0 30px rgba(102, 126, 234, 0.9);
            z-index: 1000;
            pointer-events: none;
            left: {coords['left']};
            top: {coords['top']};
            width: {coords['width']};
            height: {coords['height']};
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{
                box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.7),
                           0 0 30px rgba(102, 126, 234, 0.9);
            }}
            50% {{
                box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.7),
                           0 0 40px rgba(102, 126, 234, 1);
            }}
        }}
    </style>

    <div class="pdf-container">
        <iframe src="data:application/pdf;base64,{pdf_base64}#page=1&view=FitH" class="pdf-frame"></iframe>
        <div class="highlight-overlay"></div>
    </div>
    """

    components.html(html_content, height=800, scrolling=False)
