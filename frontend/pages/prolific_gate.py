"""
Prolific ID gate — shown before anything else when visiting a phase URL.

The entire page is replaced by a simple form asking for the participant's
Prolific ID.  Once submitted the ID is stored in st.session_state.prolific_id
and the normal app renders on the next rerun.
"""
import streamlit as st
from phases import get_phase_label


_GATE_CSS = """
<style>
    /* Hide all Streamlit chrome so only the gate form is visible */
    header[data-testid="stHeader"]          { display: none !important; }
    [data-testid="stSidebar"]               { display: none !important; }
    [data-testid="stSidebarCollapsedControl"]{ display: none !important; }
    #MainMenu                               { display: none !important; }
    footer                                  { display: none !important; }
    .stDeployButton                         { display: none !important; }
    .stToolbar                              { display: none !important; }

    /* Remove default page padding so we control centering ourselves */
    .main .block-container {
        max-width: 520px !important;
        margin: 8vh auto 0 auto !important;
        padding: 2.5rem 2rem !important;
        background: #f8f9fa;
        border-radius: 14px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.09);
    }
</style>
"""


def render_prolific_id_gate(phase_url: str):
    """
    Render a full-page Prolific ID entry form.

    Call this before render_header() and follow with st.stop() so nothing
    else renders until the participant submits their ID.

    Args:
        phase_url: The phase identifier from the URL (e.g. 'PHASEONE').
    """
    st.markdown(_GATE_CSS, unsafe_allow_html=True)

    label = get_phase_label(phase_url)

    st.markdown("## Welcome to the Flight Study")
    st.markdown(
        f"You are accessing **{label}** of our flight-preference research study.  \n"
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
