"""
Admin page for downloading completed session data.
Password-protected interface showing only fully completed sessions.
"""

import streamlit as st
from admin_utils import get_all_sessions_summary
from export_session_data import export_session_to_csv
import os

# Page configuration
st.set_page_config(
    page_title="Admin - Session Data",
    page_icon="🔐",
    layout="wide"
)

# Initialize session state for authentication
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

# Authentication
if not st.session_state.admin_authenticated:
    st.title("🔐 Admin Access")
    st.markdown("---")

    password = st.text_input("Enter password:", type="password")

    if st.button("Login", type="primary"):
        if password == "password":
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password")

    st.stop()

# Main admin interface (only shown if authenticated)
st.title("📊 Completed Session Data")
st.markdown("---")

# Get all sessions
sessions = get_all_sessions_summary()

# Filter to only completed sessions
completed_sessions = [s for s in sessions if s['is_completed']]

if not completed_sessions:
    st.info("No completed sessions found.")
    st.stop()

st.markdown(f"**Total completed sessions:** {len(completed_sessions)}")
st.markdown("---")

# Create a table view of completed sessions
for session in completed_sessions:
    token = session['completion_token']
    completed_at = session['completed_at'].strftime('%Y-%m-%d %H:%M:%S')
    prompt = session.get('search_prompt', 'N/A')

    # Create expander for each session
    with st.expander(f"**{token}** - Completed: {completed_at}"):
        # Session details
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Prompt:**")
            st.markdown(f"_{prompt[:100]}..._" if len(prompt) > 100 else f"_{prompt}_")

        with col2:
            st.markdown("**Data Available:**")
            if session.get('has_survey'):
                st.markdown("✅ Survey")
            if session.get('has_cv'):
                st.markdown(f"✅ Cross-validation ({session['cv_count']})")
            if session.get('lilo_completed'):
                st.markdown(f"✅ LILO ({session['lilo_rankings']} rankings)")

        with col3:
            st.markdown("**Quality Metrics:**")
            if session.get('survey_satisfaction'):
                st.markdown(f"Satisfaction: {session['survey_satisfaction']}/5")

        st.markdown("---")

        # Download button
        try:
            # Generate CSV
            csv_file = export_session_to_csv(token, output_file=None)

            if csv_file:
                # Read the CSV
                with open(csv_file, 'r', encoding='utf-8') as f:
                    csv_data = f.read()

                # Provide download
                st.download_button(
                    label="📥 Download Complete Session Data (CSV)",
                    data=csv_data,
                    file_name=f"session_data_{token}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"download_{token}"
                )

                # Clean up temp file
                if os.path.exists(csv_file):
                    os.remove(csv_file)
            else:
                st.warning("⚠️ Could not generate CSV for this session")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

st.markdown("---")

# Logout button
if st.button("🚪 Logout"):
    st.session_state.admin_authenticated = False
    st.rerun()
