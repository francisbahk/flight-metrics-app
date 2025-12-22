"""
Static demo page - displays the actual PDF screenshot with tutorial highlighting.
"""
import streamlit as st
import os
from pathlib import Path
from PIL import Image, ImageDraw


def render_static_demo_page(step_num):
    """
    Shows the actual PDF/screenshot image and highlights specific sections for each tutorial step.
    """

    # Path to tutorial images
    static_dir = Path(__file__).parent.parent / "static" / "tutorial"
    page_1_path = static_dir / "page_1.png"

    # Define highlight coordinates for each step (based on the PDF screenshot)
    # Coordinates are in pixels for the 834x1250 image
    highlight_steps = [
        {
            "title": "1. Your flight prompt",
            "page": 1,
            "coords": (150, 650, 684, 800)  # x1, y1, x2, y2
        },
        {
            "title": "2. Search buttons",
            "page": 1,
            "coords": (150, 830, 684, 880)
        },
        {
            "title": "3. Filters sidebar",
            "page": 1,
            "coords": (17, 190, 130, 650)
        },
        {
            "title": "4. All Available Flights",
            "page": 1,
            "coords": (150, 950, 500, 1200)
        },
        {
            "title": "5. Select flights (checkboxes)",
            "page": 1,
            "coords": (150, 1000, 500, 1180)
        },
        {
            "title": "6. Your Top 5 rankings",
            "page": 1,
            "coords": (517, 950, 817, 1180)
        },
        {
            "title": "7. Submit button",
            "page": 1,
            "coords": (517, 1120, 817, 1160)
        },
    ]

    current_step = highlight_steps[min(step_num, len(highlight_steps) - 1)]
    coords = current_step["coords"]

    # Check if tutorial image exists
    if not page_1_path.exists():
        st.error("📸 Tutorial screenshot not found!")
        st.info(f"""
        **To set up the tutorial:**

        Run this command to download and convert the PDF:
        ```bash
        curl -L -o static/tutorial/screenshot.pdf "https://github.com/francisbahk/flight-metrics-app/raw/main/screencapture-listen-cornell3-streamlit-app-2025-12-21-23_27_16.pdf"
        python3 -c "import fitz; pdf=fitz.open('static/tutorial/screenshot.pdf'); [pdf[i].get_pixmap(dpi=150).save(f'static/tutorial/page_{i+1}.png') for i in range(len(pdf))]"
        ```
        """)
        st.write(f"**Current step ({step_num + 1}/7):** {current_step['title']}")
        return

    # Load and annotate the image
    img = Image.open(page_1_path)
    img_with_highlight = img.copy()
    draw = ImageDraw.Draw(img_with_highlight, "RGBA")

    # Draw semi-transparent dark overlay everywhere
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 180))
    img_with_highlight.paste(overlay, (0, 0), overlay)

    # Cut out the highlighted area (make it bright again)
    x1, y1, x2, y2 = coords
    highlight_region = img.crop((x1, y1, x2, y2))
    img_with_highlight.paste(highlight_region, (x1, y1))

    # Draw a border around the highlighted area
    border_color = (102, 126, 234, 255)  # Purple border
    border_width = 4
    for i in range(border_width):
        draw.rectangle(
            [(x1 - i, y1 - i), (x2 + i, y2 + i)],
            outline=border_color,
            width=2
        )

    # Display the annotated image
    st.image(img_with_highlight, use_container_width=True, caption=current_step["title"])
