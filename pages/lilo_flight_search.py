"""
LILO-Based Flight Search
Interactive flight search that learns your preferences through natural language feedback
"""
import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lilo_integration import StreamlitLILOBridge

# Page configuration
st.set_page_config(
    page_title="LILO Flight Search",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'lilo_bridge' not in st.session_state:
    st.session_state.lilo_bridge = None
if 'lilo_session_id' not in st.session_state:
    st.session_state.lilo_session_id = None
if 'current_questions' not in st.session_state:
    st.session_state.current_questions = []
if 'current_flights' not in st.session_state:
    st.session_state.current_flights = pd.DataFrame()
if 'lilo_phase' not in st.session_state:
    st.session_state.lilo_phase = 'setup'  # setup, initial_questions, optimization, complete


# ============================================================================
# HEADER
# ============================================================================
st.title("🤖 AI-Powered Flight Search with LILO")
st.markdown("""
This experimental flight search learns your preferences through conversation!
Instead of complex filters, just answer a few questions and the AI will learn what you value.
""")

# ============================================================================
# API KEY SETUP
# ============================================================================
if st.session_state.lilo_phase == 'setup':
    st.header("1️⃣ Setup")

    with st.expander("ℹ️ How LILO Works", expanded=False):
        st.markdown("""
        **LILO (Learned Inference via Learned Optimization)** is an AI system that:

        1. **Asks you natural language questions** about flight preferences
        2. **Learns from your answers** using LLMs and Bayesian optimization
        3. **Suggests better flight options** based on what it learned
        4. **Improves iteratively** as you provide more feedback

        This is **research technology** - it's experimental but can find flights that match your preferences better than traditional filters!
        """)

    st.subheader("Enter your Gemini API Key")
    st.markdown("Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey)")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Your API key is only used for this session and not stored"
    )

    if st.button("Initialize LILO", type="primary", disabled=not api_key):
        with st.spinner("Initializing LILO optimizer..."):
            try:
                # Initialize bridge
                st.session_state.lilo_bridge = StreamlitLILOBridge(api_key=api_key)

                # Create session
                session_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.lilo_session_id = session_id

                # Create empty flight dataframe (will be populated during optimization)
                flights_df = pd.DataFrame()

                session = st.session_state.lilo_bridge.create_session(
                    session_id=session_id,
                    flights_df=flights_df
                )

                # Get initial questions
                questions = st.session_state.lilo_bridge.get_initial_questions(session_id)
                st.session_state.current_questions = questions
                st.session_state.lilo_phase = 'initial_questions'

                st.success("✅ LILO initialized! Let's learn your preferences.")
                st.rerun()

            except Exception as e:
                st.error(f"Error initializing LILO: {str(e)}")

# ============================================================================
# INITIAL PREFERENCE QUESTIONS
# ============================================================================
elif st.session_state.lilo_phase == 'initial_questions':
    st.header("2️⃣ Tell us your flight preferences")

    st.markdown("""
    The AI will ask you a few questions to understand what you value in flights.
    Be as specific or general as you like - natural language works!
    """)

    # Display questions and collect answers
    answers = []
    with st.form("initial_preferences"):
        for i, question in enumerate(st.session_state.current_questions):
            st.subheader(f"Question {i+1}")
            st.markdown(f"**{question}**")
            answer = st.text_area(
                f"Your answer",
                key=f"q_init_{i}",
                height=100,
                placeholder="E.g., 'I prefer morning flights because...'"
            )
            answers.append(answer)

        submitted = st.form_submit_button("Submit Preferences", type="primary")

        if submitted:
            # Check all questions are answered
            if all(answers):
                with st.spinner("Processing your preferences..."):
                    try:
                        # Submit answers to LILO
                        result = st.session_state.lilo_bridge.submit_user_answers(
                            st.session_state.lilo_session_id,
                            answers
                        )

                        st.session_state.lilo_phase = 'optimization'
                        st.success("✅ Preferences recorded! Generating flight options...")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error processing preferences: {str(e)}")
            else:
                st.warning("Please answer all questions before continuing.")

# ============================================================================
# OPTIMIZATION LOOP
# ============================================================================
elif st.session_state.lilo_phase == 'optimization':
    st.header("3️⃣ Exploring Flight Options")

    # Get status
    status = st.session_state.lilo_bridge.get_session_status(
        st.session_state.lilo_session_id
    )

    # Progress indicator
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Round", status['current_trial'])
    with col2:
        st.metric("Total Options Explored", status['total_experiments'])
    with col3:
        st.metric("Pending Questions", len(st.session_state.current_questions))

    st.markdown("---")

    # If no current flights, generate them
    if st.session_state.current_flights.empty or st.button("Get Next Flight Options"):
        with st.spinner("Finding flights based on your preferences..."):
            try:
                flights_df, questions = st.session_state.lilo_bridge.get_next_flight_options(
                    st.session_state.lilo_session_id
                )

                st.session_state.current_flights = flights_df
                st.session_state.current_questions = questions

            except Exception as e:
                st.error(f"Error generating flight options: {str(e)}")

    # Display current flight options
    if not st.session_state.current_flights.empty:
        st.subheader("Generated Flight Options")

        st.info("""
        💡 These are **candidate configurations** representing different preference weights.
        In a real implementation, these would map to actual flights from your search.
        """)

        st.dataframe(
            st.session_state.current_flights,
            use_container_width=True
        )

    # Ask feedback questions
    if st.session_state.current_questions:
        st.subheader("📋 Help us understand your preferences better")

        answers = []
        with st.form(f"feedback_round_{status['current_trial']}"):
            for i, question in enumerate(st.session_state.current_questions):
                st.markdown(f"**Q{i+1}: {question}**")
                answer = st.text_area(
                    "Your answer",
                    key=f"q_opt_{status['current_trial']}_{i}",
                    height=80,
                    placeholder="Share your thoughts..."
                )
                answers.append(answer)

            col_a, col_b = st.columns([1, 3])
            with col_a:
                submit_feedback = st.form_submit_button("Submit Feedback", type="primary")
            with col_b:
                finish_optimization = st.form_submit_button("Finish & See Best Options")

            if submit_feedback:
                if all(answers):
                    with st.spinner("Learning from your feedback..."):
                        try:
                            result = st.session_state.lilo_bridge.submit_user_answers(
                                st.session_state.lilo_session_id,
                                answers
                            )

                            # Clear current data to trigger next iteration
                            st.session_state.current_flights = pd.DataFrame()
                            st.session_state.current_questions = []
                            st.success("✅ Feedback recorded!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error submitting feedback: {str(e)}")
                else:
                    st.warning("Please answer all questions.")

            if finish_optimization:
                st.session_state.lilo_phase = 'complete'
                st.rerun()

# ============================================================================
# COMPLETION
# ============================================================================
elif st.session_state.lilo_phase == 'complete':
    st.header("✅ Optimization Complete!")

    st.success("""
    🎉 LILO has learned your preferences!
    In a real implementation, we would now show you the best matching flights.
    """)

    # Get final status
    status = st.session_state.lilo_bridge.get_session_status(
        st.session_state.lilo_session_id
    )

    st.metric("Total Rounds Completed", status['current_trial'])
    st.metric("Total Options Explored", status['total_experiments'])

    # Show session data
    session = st.session_state.lilo_bridge.sessions[st.session_state.lilo_session_id]
    optimizer = session.optimizer

    if not optimizer.exp_df.empty:
        st.subheader("All Explored Options")
        st.dataframe(optimizer.exp_df, use_container_width=True)

    if not optimizer.context_df.empty:
        st.subheader("Your Feedback History")
        st.dataframe(optimizer.context_df, use_container_width=True)

    if st.button("Start New Search"):
        # Reset session
        st.session_state.lilo_phase = 'setup'
        st.session_state.lilo_bridge = None
        st.session_state.lilo_session_id = None
        st.session_state.current_questions = []
        st.session_state.current_flights = pd.DataFrame()
        st.rerun()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("About LILO")

    st.markdown("""
    **LILO** is a research system for interactive optimization using natural language.

    **Paper**: [arXiv:2510.17671](https://arxiv.org/abs/2510.17671)

    **How it works**:
    - Uses LLMs to interpret your preferences
    - Bayesian optimization to suggest options
    - Learns iteratively from your feedback
    """)

    if st.session_state.lilo_phase != 'setup':
        st.markdown("---")
        st.header("Session Info")

        if st.session_state.lilo_session_id:
            status = st.session_state.lilo_bridge.get_session_status(
                st.session_state.lilo_session_id
            )
            st.code(f"""
Session: {st.session_state.lilo_session_id}
Phase: {st.session_state.lilo_phase}
Round: {status['current_trial']}
Options Explored: {status['total_experiments']}
            """)

    st.markdown("---")
    st.caption("⚠️ Experimental Research System")
