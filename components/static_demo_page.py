"""
Static demo page - shows a frozen version of the app for tutorial purposes.
Completely separate from the real app, with spotlight highlighting.
"""
import streamlit as st
from components.demo_spotlight import show_spotlight_step


def render_static_demo_page(step_num):
    """
    Renders a completely static, frozen version of the flight app.
    Shows what the app looks like after searching, with spotlight on current step.
    """

    # Add demo-mode class to disable interactions
    st.markdown("""
    <style>
        /* Disable all clicks except tutorial buttons */
        .demo-mode * {{
            pointer-events: none !important;
        }}

        .tutorial-btn {{
            pointer-events: auto !important;
        }}
    </style>
    <div class="demo-mode">
    """, unsafe_allow_html=True)

    # Spotlight rendering removed - tutorial_card.py handles both highlighting and card

    st.title("Flight Ranker ✈️")

    # Step 1: Prompt box (pre-filled, disabled)
    st.markdown('<div id="demo-prompt" style="padding: 10px 0;">', unsafe_allow_html=True)
    st.text_area(
        "Describe your ideal flight",
        value="I want to fly from JFK to LAX on December 25th. I prefer cheap flights but if a flight is longer than 12 hours I'd prefer to pay a bit more money to take a shorter flight. I also want to arrive in the afternoon since I want to have time to eat dinner.",
        height=120,
        disabled=True,
        key="demo_prompt_input"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 2: Search button
    st.markdown('<div id="demo-search-btn" style="padding: 10px 0;">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.button("🔍 Search Flights", disabled=True, use_container_width=True)
    with col2:
        st.button("✨ AI Search", disabled=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Step 3: Sidebar filters
    with st.sidebar:
        st.markdown('<div id="demo-filters" style="padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 10px 0;">', unsafe_allow_html=True)
        st.markdown("### 🔍 Filters")
        st.multiselect("Airlines", ["JetBlue", "Delta", "American"], disabled=True)
        st.multiselect("Stops", ["Direct", "1 stop"], disabled=True)
        st.slider("Price Range", 0, 1000, (0, 500), disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 4: Results table
    st.markdown('<div id="demo-results" style="padding: 10px 0;">', unsafe_allow_html=True)
    st.markdown("### ✈️ Found 10 flights")

    # Show fake results in a table-like format
    flights_html = """
    <div style="margin: 20px 0;">
        <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: #f9f9f9;">
            <strong>JetBlue 123</strong> | <strong>$289</strong><br>
            JFK → LAX | 8:00 AM - 11:30 AM | 6 hr 30 min | Direct
        </div>
        <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: #f9f9f9;">
            <strong>Delta 456</strong> | <strong>$324</strong><br>
            JFK → LAX | 10:15 AM - 1:45 PM | 6 hr 30 min | Direct
        </div>
        <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: #f9f9f9;">
            <strong>American 789</strong> | <strong>$356</strong><br>
            JFK → LAX | 12:30 PM - 4:00 PM | 6 hr 30 min | Direct
        </div>
    </div>
    """
    st.markdown(flights_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 5: Checkboxes
    st.markdown('<div id="demo-checkboxes" style="padding: 10px 0; background: #f8f9fa; border-radius: 8px; margin: 10px 0;">', unsafe_allow_html=True)
    st.markdown("**Select flights to rank:**")
    st.checkbox("✓ JetBlue 123 - $289", value=True, disabled=True)
    st.checkbox("✓ Delta 456 - $324", value=True, disabled=True)
    st.checkbox("✓ American 789 - $356", value=True, disabled=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Step 6: Ranking area
    st.markdown('<div id="demo-ranking" style="padding: 10px 0;">', unsafe_allow_html=True)
    st.markdown("### 🎯 Rank Your Selected Flights")
    ranking_html = """
    <div style="border: 2px dashed #3b82f6; border-radius: 8px; padding: 20px; margin: 16px 0;">
        <div style="background: white; border: 1px solid #ccc; border-radius: 6px; padding: 12px; margin-bottom: 8px;">
            #1: JetBlue 123 - $289
        </div>
        <div style="background: white; border: 1px solid #ccc; border-radius: 6px; padding: 12px; margin-bottom: 8px;">
            #2: Delta 456 - $324
        </div>
        <div style="background: white; border: 1px solid #ccc; border-radius: 6px; padding: 12px;">
            #3: American 789 - $356
        </div>
    </div>
    """
    st.markdown(ranking_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 7: Submit button
    st.markdown('<div id="demo-submit" style="padding: 10px 0;">', unsafe_allow_html=True)
    st.button("✅ Submit Rankings", disabled=True, use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # Close demo-mode div
    st.markdown('</div>', unsafe_allow_html=True)
