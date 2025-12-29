"""
Admin Dashboard for Flight Research Study
Researcher login and real-time data viewing/export
"""
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
import io

load_dotenv()

# Page config
st.set_page_config(
    page_title="Admin Dashboard - Flight Study",
    page_icon="🔐",
    layout="wide"
)

# Authentication
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (st.session_state["username"] == os.getenv("ADMIN_USERNAME", "admin") and
            st.session_state["password"] == os.getenv("ADMIN_PASSWORD", "research123")):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password
        st.markdown("### 🔐 Researcher Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.markdown("### 🔐 Researcher Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct
        return True


if not check_password():
    st.stop()

# Logged in - show dashboard
st.title("📊 Flight Study Admin Dashboard")
st.markdown(f"**Logged in as:** {st.session_state.get('username', 'admin')} | **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Logout button in sidebar
with st.sidebar:
    st.markdown("### 🔐 Session")
    if st.button("🚪 Logout"):
        st.session_state["password_correct"] = False
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔄 Refresh Data")
    if st.button("↻ Reload Dashboard"):
        st.rerun()

# Import database functions
try:
    from backend.db import SessionLocal, Search, UserRanking, FlightShown, LILOSession, LILORound, CompletionToken, AccessToken
    db = SessionLocal()
except Exception as e:
    st.error(f"❌ Database connection error: {e}")
    st.stop()

# Tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview",
    "🔍 All Searches",
    "🎯 LILO Sessions",
    "🏆 Rankings",
    "🎟️ Tokens",
    "📥 Export Data"
])

# TAB 1: Overview
with tab1:
    st.markdown("## 📈 Study Overview")

    col1, col2, col3, col4 = st.columns(4)

    # Metrics
    with col1:
        total_searches = db.query(Search).count()
        st.metric("Total Searches", total_searches)

    with col2:
        total_lilo = db.query(LILOSession).count()
        completed_lilo = db.query(LILOSession).filter(LILOSession.completed_at.isnot(None)).count()
        st.metric("LILO Sessions", f"{completed_lilo}/{total_lilo}",
                 delta=f"{(completed_lilo/total_lilo*100) if total_lilo > 0 else 0:.1f}% complete")

    with col3:
        total_tokens = db.query(AccessToken).count()
        used_tokens = db.query(AccessToken).filter(AccessToken.is_used == 1).count()
        st.metric("Access Tokens", f"{used_tokens}/{total_tokens} used",
                 delta=f"{(used_tokens/total_tokens*100) if total_tokens > 0 else 0:.1f}%")

    with col4:
        completion_tokens = db.query(CompletionToken).count()
        st.metric("Completed Studies", completion_tokens)

    st.markdown("---")

    # Recent activity
    st.markdown("### 🕐 Recent Activity (Last 10 Searches)")
    recent_searches = db.query(Search).order_by(Search.created_at.desc()).limit(10).all()

    if recent_searches:
        recent_data = []
        for s in recent_searches:
            recent_data.append({
                "Search ID": s.search_id,
                "Session ID": s.session_id[:16] + "...",
                "Prompt": s.user_prompt[:50] + "..." if len(s.user_prompt) > 50 else s.user_prompt,
                "Origin": s.parsed_origins[0] if s.parsed_origins else "N/A",
                "Destination": s.parsed_destinations[0] if s.parsed_destinations else "N/A",
                "Date": s.departure_date,
                "Created": s.created_at.strftime("%Y-%m-%d %H:%M")
            })
        st.dataframe(pd.DataFrame(recent_data), use_container_width=True)
    else:
        st.info("No searches yet")

# TAB 2: All Searches
with tab2:
    st.markdown("## 🔍 All Searches")

    all_searches = db.query(Search).order_by(Search.created_at.desc()).all()

    if all_searches:
        search_data = []
        for s in all_searches:
            # Get rankings count
            rankings_count = db.query(UserRanking).filter(UserRanking.search_id == s.search_id).count()

            # Get LILO info
            lilo = db.query(LILOSession).filter(LILOSession.search_id == s.search_id).first()
            lilo_status = "N/A"
            if lilo:
                if lilo.completed_at:
                    lilo_status = f"✅ {lilo.num_rounds} rounds"
                else:
                    lilo_status = f"⏳ {lilo.num_rounds} rounds"

            search_data.append({
                "ID": s.search_id,
                "Session": s.session_id[:12] + "...",
                "Completion Token": s.completion_token if s.completion_token else "N/A",
                "Prompt": s.user_prompt[:60] + "..." if len(s.user_prompt) > 60 else s.user_prompt,
                "Route": f"{s.parsed_origins[0] if s.parsed_origins else '?'} → {s.parsed_destinations[0] if s.parsed_destinations else '?'}",
                "Date": s.departure_date,
                "Rankings": rankings_count,
                "LILO": lilo_status,
                "Created": s.created_at.strftime("%Y-%m-%d %H:%M")
            })

        df = pd.DataFrame(search_data)

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            filter_lilo = st.multiselect("Filter by LILO Status",
                                        options=df["LILO"].unique().tolist(),
                                        default=None)
        with col2:
            search_filter = st.text_input("Search by Prompt/Route", "")

        # Apply filters
        if filter_lilo:
            df = df[df["LILO"].isin(filter_lilo)]
        if search_filter:
            df = df[df["Prompt"].str.contains(search_filter, case=False, na=False) |
                   df["Route"].str.contains(search_filter, case=False, na=False)]

        st.dataframe(df, use_container_width=True, height=500)
        st.caption(f"Showing {len(df)} searches")
    else:
        st.info("No searches yet")

# TAB 3: LILO Sessions
with tab3:
    st.markdown("## 🎯 LILO Preference Learning Sessions")

    lilo_sessions = db.query(LILOSession).order_by(LILOSession.created_at.desc()).all()

    if lilo_sessions:
        for lilo in lilo_sessions:
            with st.expander(f"Session {lilo.session_id[:16]}... | Rounds: {lilo.num_rounds} | {'✅ Completed' if lilo.completed_at else '⏳ In Progress'}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Session ID:** `{lilo.session_id}`")
                    st.markdown(f"**Search ID:** {lilo.search_id}")
                    st.markdown(f"**Rounds Completed:** {lilo.num_rounds}")
                    st.markdown(f"**Created:** {lilo.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    if lilo.completed_at:
                        st.markdown(f"**Completed:** {lilo.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")

                with col2:
                    if lilo.feedback_summary:
                        st.markdown("**Feedback Summary:**")
                        st.info(lilo.feedback_summary)

                # Show rounds
                rounds = db.query(LILORound).filter(LILORound.session_id == lilo.session_id).order_by(LILORound.round_number).all()

                if rounds:
                    st.markdown("### 🔄 Rounds")
                    for r in rounds:
                        st.markdown(f"#### Round {r.round_number}")
                        st.markdown(f"**Rankings:** {r.user_rankings}")
                        st.markdown(f"**Feedback:** {r.user_feedback}")
                        if r.generated_questions:
                            st.markdown(f"**Questions Generated:** {r.generated_questions}")
                        if r.extracted_preferences:
                            st.markdown(f"**User Answers:** {r.extracted_preferences}")
                        st.markdown("---")

                # Show utility scores if available
                if lilo.final_utility_scores:
                    st.markdown("### 📊 Final Utility Scores")
                    st.markdown(f"Computed {len(lilo.final_utility_scores)} utility scores")
                    st.line_chart(lilo.final_utility_scores[:20])  # Show top 20
    else:
        st.info("No LILO sessions yet")

# TAB 4: Rankings
with tab4:
    st.markdown("## 🏆 User Rankings")

    all_rankings = db.query(UserRanking).join(Search).order_by(UserRanking.submitted_at.desc()).limit(100).all()

    if all_rankings:
        ranking_data = []
        for r in all_rankings:
            flight = db.query(FlightShown).filter(FlightShown.id == r.flight_id).first()
            search = db.query(Search).filter(Search.search_id == r.search_id).first()

            if flight and search:
                flight_info = flight.flight_data
                ranking_data.append({
                    "Search ID": r.search_id,
                    "Session": search.session_id[:12] + "...",
                    "User Rank": r.user_rank,
                    "Algorithm": flight.algorithm,
                    "Algo Rank": flight.algorithm_rank,
                    "Airline": flight_info.get('airline', 'N/A'),
                    "Route": f"{flight_info.get('origin', '?')} → {flight_info.get('destination', '?')}",
                    "Price": f"${flight_info.get('price', 0):.0f}",
                    "Duration": f"{flight_info.get('duration_min', 0)//60}h{flight_info.get('duration_min', 0)%60}m",
                    "Submitted": r.submitted_at.strftime("%Y-%m-%d %H:%M")
                })

        df = pd.DataFrame(ranking_data)
        st.dataframe(df, use_container_width=True, height=500)

        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_rank_by_algo = df.groupby("Algorithm")["User Rank"].mean().round(2)
            st.markdown("**Avg User Rank by Algorithm:**")
            st.dataframe(avg_rank_by_algo)
        with col2:
            rank_distribution = df["User Rank"].value_counts().sort_index()
            st.markdown("**Rank Distribution:**")
            st.bar_chart(rank_distribution)
        with col3:
            algo_counts = df["Algorithm"].value_counts()
            st.markdown("**Rankings by Algorithm:**")
            st.bar_chart(algo_counts)
    else:
        st.info("No rankings yet")

# TAB 5: Tokens
with tab5:
    st.markdown("## 🎟️ Access & Completion Tokens")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔑 Access Tokens")
        access_tokens = db.query(AccessToken).order_by(AccessToken.created_at.desc()).all()

        if access_tokens:
            access_data = []
            for t in access_tokens:
                access_data.append({
                    "Token": t.token,
                    "Status": "✅ Used" if t.is_used else "⏳ Available",
                    "Created": t.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Used At": t.used_at.strftime("%Y-%m-%d %H:%M") if t.used_at else "N/A",
                    "Completion Token": t.completion_token if t.completion_token else "N/A"
                })
            st.dataframe(pd.DataFrame(access_data), use_container_width=True, height=400)
        else:
            st.info("No access tokens generated yet")

    with col2:
        st.markdown("### 🎁 Completion Tokens")
        completion_tokens = db.query(CompletionToken).order_by(CompletionToken.created_at.desc()).all()

        if completion_tokens:
            completion_data = []
            for t in completion_tokens:
                completion_data.append({
                    "Token": t.token,
                    "Session ID": t.session_id[:16] + "...",
                    "Created": t.created_at.strftime("%Y-%m-%d %H:%M")
                })
            st.dataframe(pd.DataFrame(completion_data), use_container_width=True, height=400)
        else:
            st.info("No studies completed yet")

# TAB 6: Export Data
with tab6:
    st.markdown("## 📥 Export Research Data")
    st.info("Export anonymized research data for analysis. No personally identifiable information is included.")

    export_type = st.selectbox("Select data to export:", [
        "All Searches",
        "LILO Sessions & Rounds",
        "User Rankings",
        "Flight Data Shown",
        "Token Usage Stats",
        "Complete Dataset (All Tables)"
    ])

    if st.button("📊 Generate Export", type="primary"):
        with st.spinner("Generating export..."):
            try:
                if export_type == "All Searches":
                    searches = db.query(Search).all()
                    data = [{
                        "search_id": s.search_id,
                        "session_id": s.session_id,
                        "completion_token": s.completion_token,
                        "user_prompt": s.user_prompt,
                        "origins": s.parsed_origins,
                        "destinations": s.parsed_destinations,
                        "preferences": s.parsed_preferences,
                        "departure_date": s.departure_date,
                        "created_at": s.created_at
                    } for s in searches]
                    df = pd.DataFrame(data)

                elif export_type == "LILO Sessions & Rounds":
                    sessions = db.query(LILOSession).all()
                    data = []
                    for s in sessions:
                        rounds = db.query(LILORound).filter(LILORound.session_id == s.session_id).all()
                        for r in rounds:
                            data.append({
                                "session_id": s.session_id,
                                "search_id": s.search_id,
                                "round_number": r.round_number,
                                "flights_shown": r.flights_shown,
                                "user_rankings": r.user_rankings,
                                "user_feedback": r.user_feedback,
                                "generated_questions": r.generated_questions,
                                "extracted_preferences": r.extracted_preferences,
                                "feedback_summary": s.feedback_summary,
                                "final_utilities": s.final_utility_scores,
                                "completed": s.completed_at is not None
                            })
                    df = pd.DataFrame(data)

                elif export_type == "User Rankings":
                    rankings = db.query(UserRanking).all()
                    data = []
                    for r in rankings:
                        flight = db.query(FlightShown).filter(FlightShown.id == r.flight_id).first()
                        if flight:
                            data.append({
                                "search_id": r.search_id,
                                "user_rank": r.user_rank,
                                "algorithm": flight.algorithm,
                                "algorithm_rank": flight.algorithm_rank,
                                "flight_data": flight.flight_data,
                                "submitted_at": r.submitted_at
                            })
                    df = pd.DataFrame(data)

                elif export_type == "Flight Data Shown":
                    flights = db.query(FlightShown).all()
                    data = [{
                        "search_id": f.search_id,
                        "algorithm": f.algorithm,
                        "algorithm_rank": f.algorithm_rank,
                        "display_position": f.display_position,
                        "flight_data": f.flight_data
                    } for f in flights]
                    df = pd.DataFrame(data)

                elif export_type == "Token Usage Stats":
                    access = db.query(AccessToken).all()
                    data = [{
                        "token": t.token,
                        "created_at": t.created_at,
                        "used_at": t.used_at,
                        "is_used": t.is_used,
                        "completion_token": t.completion_token
                    } for t in access]
                    df = pd.DataFrame(data)

                else:  # Complete Dataset
                    st.warning("Complete dataset export will create multiple files...")
                    # Create a ZIP with multiple CSVs
                    import zipfile

                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Searches
                        searches = db.query(Search).all()
                        searches_df = pd.DataFrame([{
                            "search_id": s.search_id,
                            "session_id": s.session_id,
                            "completion_token": s.completion_token,
                            "user_prompt": s.user_prompt,
                            "origins": s.parsed_origins,
                            "destinations": s.parsed_destinations,
                            "created_at": s.created_at
                        } for s in searches])
                        zip_file.writestr("searches.csv", searches_df.to_csv(index=False))

                        # LILO
                        lilo_sessions = db.query(LILOSession).all()
                        lilo_df = pd.DataFrame([{
                            "session_id": s.session_id,
                            "search_id": s.search_id,
                            "num_rounds": s.num_rounds,
                            "completed_at": s.completed_at
                        } for s in lilo_sessions])
                        zip_file.writestr("lilo_sessions.csv", lilo_df.to_csv(index=False))

                        # Rankings
                        rankings = db.query(UserRanking).all()
                        rankings_df = pd.DataFrame([{
                            "search_id": r.search_id,
                            "flight_id": r.flight_id,
                            "user_rank": r.user_rank,
                            "submitted_at": r.submitted_at
                        } for r in rankings])
                        zip_file.writestr("rankings.csv", rankings_df.to_csv(index=False))

                    buffer.seek(0)
                    st.download_button(
                        label="📦 Download Complete Dataset (ZIP)",
                        data=buffer,
                        file_name=f"flight_study_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
                    df = None

                # Download button for single file exports
                if df is not None:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{export_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                    st.success(f"✅ Export ready! {len(df)} records")
                    st.dataframe(df.head(10), use_container_width=True)
                    st.caption("Showing first 10 rows")

            except Exception as e:
                st.error(f"Export error: {e}")
                import traceback
                st.code(traceback.format_exc())

# Close database connection
db.close()
