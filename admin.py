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

# Simple list of sessions with download links
for session in completed_sessions:
    token = session['completion_token']
    completed_at = session['completed_at'].strftime('%Y-%m-%d %H:%M:%S')

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**{token}** — {completed_at}")

    with col2:
        try:
            csv_file = export_session_to_csv(token, output_file=None)
            if csv_file:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    csv_data = f.read()
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"session_data_{token}.csv",
                    mime="text/csv",
                    key=f"download_{token}"
                )
                if os.path.exists(csv_file):
                    os.remove(csv_file)
            else:
                st.markdown("_No data_")
        except Exception as e:
            st.markdown(f"_Error_")

st.markdown("---")

# Logout button
if st.button("🚪 Logout"):
    st.session_state.admin_authenticated = False
    st.rerun()
