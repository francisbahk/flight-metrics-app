"""
Screening questions page — shown after Prolific ID gate, before the main study.

4 mandatory questions. If the participant answers "0" to question 4 (flights
shopped in the past 3 years) they are screen-out redirected to Prolific.
"""
import streamlit as st
import streamlit.components.v1 as components


_SCREEN_CSS = """
<style>
    .stApp * { visibility: hidden !important; }
    .block-container,
    .block-container * { visibility: visible !important; }
    .block-container {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        max-width: 560px !important;
        width: 92% !important;
        padding: 2.5rem 2rem !important;
        background: #f8f9fa !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.09) !important;
        max-height: 90vh !important;
        overflow-y: auto !important;
    }
</style>
"""

_SCREEN_OUT_URL = "https://app.prolific.com/submissions/complete?cc=CW9AC4DG"

_OPTIONS = ["0", "1–4", "5+"]

_QUESTIONS = [
    "How many books have you read in the past 12 months?",
    "How many times have you ordered takeout or food delivery in the past month?",
    "How many times have you attended a fitness class (e.g., yoga, pilates, spin) in the past 3 months?",
    "How many flights have you shopped for in the past 3 years?",
]


def render_screening_page():
    """Render the screening questions page. Call before st.stop()."""
    st.markdown(_SCREEN_CSS, unsafe_allow_html=True)

    # If already screened out, just show the redirect
    if st.session_state.get('screened_out'):
        _render_screen_out()
        return

    st.markdown("## Eligibility Questions")
    st.markdown(
        "Please answer the following questions to confirm your eligibility for this study. "
        "All questions are required."
    )
    st.markdown("---")

    with st.form("screening_form", clear_on_submit=False):
        answers = []
        for i, question in enumerate(_QUESTIONS):
            st.markdown(f"**{i + 1}. {question}**")
            ans = st.radio(
                label=question,
                options=_OPTIONS,
                index=None,
                key=f"screen_q{i + 1}",
                label_visibility="collapsed",
                horizontal=True,
            )
            answers.append(ans)
            if i < len(_QUESTIONS) - 1:
                st.markdown("")

        st.markdown("---")
        submitted = st.form_submit_button(
            "Continue", type="primary", use_container_width=True
        )

        if submitted:
            # Validate all answered
            unanswered = [i + 1 for i, a in enumerate(answers) if a is None]
            if unanswered:
                qs = ", ".join(f"Question {n}" for n in unanswered)
                st.error(f"Please answer all questions before continuing. Missing: {qs}")
            else:
                q4_answer = answers[3]
                if q4_answer == "0":
                    # Save screening data then screen out
                    _save_screening(answers, screened_out=True)
                    st.session_state.screened_out = True
                    st.rerun()
                else:
                    _save_screening(answers, screened_out=False)
                    st.session_state.screening_completed = True
                    st.session_state.screening_answers = {
                        f"q{i + 1}": a for i, a in enumerate(answers)
                    }
                    st.rerun()


def _render_screen_out():
    """Show screen-out message and redirect."""
    st.markdown("## Thank You")
    st.markdown(
        "Based on your responses, you do not meet the eligibility criteria for this study. "
        "Please click the button below to return to Prolific."
    )
    st.markdown("---")
    st.link_button(
        "Return to Prolific", _SCREEN_OUT_URL, type="primary", use_container_width=True
    )
    # Auto-redirect via JS
    components.html(
        f'<script>window.top.location.href = "{_SCREEN_OUT_URL}";</script>',
        height=0,
    )


def _save_screening(answers: list, screened_out: bool):
    """Persist screening answers to the database."""
    try:
        from backend.db import save_screening_data
        prolific_id = st.session_state.get('prolific_id', '')
        save_screening_data(
            prolific_id=prolific_id,
            answers={f"q{i + 1}": a for i, a in enumerate(answers)},
            screened_out=screened_out,
        )
    except Exception as e:
        print(f"[SCREENING] Failed to save screening data: {e}")
