import streamlit as st


def render_informed_consent():
    st.title("Informed Consent")

    st.markdown("""
<div style="
    background:#f8f9fa;
    border:1px solid #dee2e6;
    border-radius:8px;
    padding:2rem;
    font-family:monospace;
    font-size:0.88rem;
    line-height:1.7;
    white-space:pre-wrap;
">INFORMED CONSENT — LISTEN Project (Pilot Study)
Cornell University, School of Operations Research and Information Engineering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOTE: This is a pilot study. Data will not be published in any academic papers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STUDY OVERVIEW
You are invited to participate in the LISTEN Project, a pilot study
conducted by Matthew T. Ford, Ph.D. Student, School of Operations
Research and Information Engineering, Cornell University.
Faculty Advisor: Peter I. Frazier, Cornell University.

WHAT YOU WILL DO
You will review options or scenarios and express preferences or
rankings among them. This takes approximately 15 minutes.

RISKS AND BENEFITS
There are no anticipated risks. There are no direct benefits to you.
Data collected in this pilot study will not be published in any
academic papers, journals, or conference proceedings.

COMPENSATION
Compensation is as listed on Prolific.

CONFIDENTIALITY
No personal information is collected. Responses are anonymous and
stored securely, accessible only to the research team.

VOLUNTARY PARTICIPATION
Participation is voluntary. You may stop at any time without penalty.

QUESTIONS
Contact Matthew T. Ford at mtf62@cornell.edu.
For rights as a participant: Cornell IRB, 607-255-6182
  or https://researchservices.cornell.edu/offices/IRB

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

By continuing in this study, you acknowledge that you have read and
understood the above information and accept the terms of participation.</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    agreed = st.checkbox(
        "I have read and understood the above information and agree to participate in this study."
    )

    st.markdown("")  # spacing

    if st.button("Continue", type="primary", disabled=not agreed):
        st.session_state.consent_given = True
        st.rerun()
