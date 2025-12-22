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

    # Use screenshot image (converted from PDF)
    from PIL import Image, ImageDraw

    img_path = Path(__file__).parent.parent / "tutorial_screenshot.png"

    # Load and annotate the image
    img = Image.open(img_path)
    img_with_highlight = img.copy().convert("RGBA")

    # Create overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 180))

    # Calculate pixel coordinates from percentages
    x1 = int(float(coords['left'].strip('%')) / 100 * img.width)
    y1 = int(float(coords['top'].strip('%')) / 100 * img.height)
    w = int(float(coords['width'].strip('%')) / 100 * img.width)
    h = int(float(coords['height'].strip('%')) / 100 * img.height)
    x2 = x1 + w
    y2 = y1 + h

    # Cut out the highlighted area from overlay
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 0, 0, 0))

    # Composite overlay onto image
    img_with_highlight = Image.alpha_composite(img_with_highlight, overlay)

    # Draw border around highlighted area
    draw = ImageDraw.Draw(img_with_highlight)
    border_color = (102, 126, 234, 255)
    for i in range(4):
        draw.rectangle(
            [(x1 - i, y1 - i), (x2 + i, y2 + i)],
            outline=border_color,
            width=2
        )

    # Display the annotated image
    st.image(img_with_highlight, use_container_width=True)
