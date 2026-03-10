"""
Prolific ID gate — shown before anything else when visiting the study URL.

The entire page is replaced by a simple form asking for the participant's
Prolific ID.  Once submitted the ID is stored in st.session_state.prolific_id
and the normal app renders on the next rerun.
"""
import streamlit as st


_GATE_CSS = """
<style>
    /* Hide every element inside the app */
    .stApp * { visibility: hidden !important; }

    /* Re-show only the form card and everything inside it.
       visibility:visible on a child overrides inherited visibility:hidden — CSS spec. */
    .block-container,
    .block-container * { visibility: visible !important; }

    /* Center the card in the viewport */
    .block-container {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        max-width: 420px !important;
        width: 90% !important;
        padding: 2.5rem 2rem !important;
        background: #f8f9fa !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.09) !important;
    }
</style>
"""


def render_prolific_id_gate():
    """Render a full-page Prolific ID entry form. Call before st.stop()."""
    st.markdown(_GATE_CSS, unsafe_allow_html=True)

    st.markdown("## Welcome to the Flight Study")
    st.markdown(
        "Please enter your **Prolific ID** to continue. "
        "This is used only to verify your participation — it is never shared "
        "and will not be linked to your responses."
    )
    st.markdown("---")

    with st.form("prolific_id_form", clear_on_submit=False):
        prolific_id = st.text_input(
            "Prolific ID",
            placeholder="e.g. 64a1b2c3d4e5f6a7b8c9d0e1",
            max_chars=64,
        )
        submitted = st.form_submit_button("Continue to Study", type="primary", use_container_width=True)

        if submitted:
            pid = prolific_id.strip()
            if not pid:
                st.error("Please enter your Prolific ID before continuing.")
            elif len(pid) < 6:
                st.error("That doesn't look like a valid Prolific ID. Please check and try again.")
            else:
                st.session_state.prolific_id = pid
                st.rerun()
