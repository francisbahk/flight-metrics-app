"""
Static demo page - displays pre-rendered tutorial images.
"""
import streamlit as st
from pathlib import Path


def render_static_demo_page(step_num):
    """
    Shows pre-rendered tutorial screenshots with highlights already baked in.
    No dynamic rendering - just simple image display.
    """

    # Path to pre-rendered tutorial images
    img_path = Path(__file__).parent.parent / "tutorial_images" / f"step_{step_num + 1}.png"

    # Display the pre-rendered image
    if img_path.exists():
        st.image(str(img_path), width='stretch')
    else:
        st.error(f"Tutorial image not found: {img_path}")
