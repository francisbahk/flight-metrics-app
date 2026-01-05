#!/usr/bin/env python3
"""
View saved LILO results from the database.
Usage: python view_lilo_results.py [session_id or completion_token]
"""

import sys
from backend.db import SessionLocal, LILOSession, LILOChatMessage, LILOIteration, LILOFinalRanking
from sqlalchemy import desc
import json
from tabulate import tabulate


def list_all_lilo_sessions():
    """List all LILO sessions in the database with comprehensive summaries."""
    db = SessionLocal()
    try:
        sessions = db.query(LILOSession).order_by(desc(LILOSession.created_at)).all()

        if not sessions:
            print("No LILO sessions found in database.")
            return

        print("\n" + "="*80)
        print("ALL LILO SESSIONS - COMPREHENSIVE SUMMARY")
        print("="*80 + "\n")

        table_data = []
        for session in sessions:
            # Count data points for this session
            message_count = db.query(LILOChatMessage).filter(
                LILOChatMessage.lilo_session_id == session.id
            ).count()

            ranking_count = db.query(LILOFinalRanking).filter(
                LILOFinalRanking.lilo_session_id == session.id
            ).count()

            # Get top flight info
            top_flight = db.query(LILOFinalRanking).filter(
                LILOFinalRanking.lilo_session_id == session.id
            ).order_by(LILOFinalRanking.rank).first()

            top_flight_info = "N/A"
            if top_flight:
                flight = top_flight.flight_data
                top_flight_info = f"${flight.get('price', '?')} | {flight.get('duration_min', '?')}min | {flight.get('stops', '?')} stops"

            table_data.append([
                session.id,
                session.completion_token or "N/A",
                session.num_iterations,
                message_count,
                ranking_count,
                session.started_at.strftime("%m/%d %H:%M") if session.started_at else "N/A",
                "✓" if session.completed_at else "✗",
                top_flight_info[:30]
            ])

        headers = ["ID", "Token", "Iters", "Msgs", "Rankings", "Started", "Done", "Top Flight"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"\nTotal sessions: {len(sessions)}")
        print("\nUse 'python view_lilo_results.py <ID>' to view detailed session summary")

    finally:
        db.close()


def view_lilo_session(identifier: str):
    """View detailed LILO session data by session_id or completion_token."""
    db = SessionLocal()
    try:
        # Try to find session by session_id or completion_token or DB ID
        try:
            db_id = int(identifier)
            session = db.query(LILOSession).filter(LILOSession.id == db_id).first()
        except ValueError:
            session = db.query(LILOSession).filter(
                (LILOSession.session_id == identifier) |
                (LILOSession.completion_token == identifier)
            ).first()

        if not session:
            print(f"❌ No LILO session found for identifier: {identifier}")
            return

        print("\n" + "="*80)
        print("LILO SESSION COMPREHENSIVE SUMMARY")
        print("="*80 + "\n")

        # Count data points
        message_count = db.query(LILOChatMessage).filter(
            LILOChatMessage.lilo_session_id == session.id
        ).count()

        iteration_count = db.query(LILOIteration).filter(
            LILOIteration.lilo_session_id == session.id
        ).count()

        ranking_count = db.query(LILOFinalRanking).filter(
            LILOFinalRanking.lilo_session_id == session.id
        ).count()

        # Calculate duration
        duration_str = "N/A"
        if session.started_at and session.completed_at:
            duration_minutes = (session.completed_at - session.started_at).total_seconds() / 60
            duration_str = f"{duration_minutes:.1f} minutes"

        # Session metadata
        print("📋 Session Metadata:")
        print(f"  Database ID: {session.id}")
        print(f"  Session ID: {session.session_id}")
        print(f"  Completion Token: {session.completion_token or 'N/A'}")
        print(f"  Search ID: {session.search_id}")
        print(f"  Status: {'✅ Completed' if session.completed_at else '⏳ In Progress'}")
        print(f"  Duration: {duration_str}")
        print(f"  Started: {session.started_at.strftime('%Y-%m-%d %H:%M:%S') if session.started_at else 'N/A'}")
        print(f"  Completed: {session.completed_at.strftime('%Y-%m-%d %H:%M:%S') if session.completed_at else 'Not completed'}")
        print()

        # Data summary
        print("📊 Data Summary:")
        print(f"  Iterations Configured: {session.num_iterations}")
        print(f"  Questions per Round: {session.questions_per_round}")
        print(f"  Chat Messages: {message_count}")
        print(f"  Iterations Saved: {iteration_count}")
        print(f"  Flights Ranked: {ranking_count}")
        print()

        # Chat transcript
        messages = db.query(LILOChatMessage).filter(
            LILOChatMessage.lilo_session_id == session.id
        ).order_by(LILOChatMessage.round_number, LILOChatMessage.message_index).all()

        if messages:
            print("💬 Chat Transcript:")
            print("-" * 80)
            current_round = -1
            for msg in messages:
                if msg.round_number != current_round:
                    current_round = msg.round_number
                    print(f"\n[Round {msg.round_number}]")

                speaker = "🤖 Bot" if msg.is_bot else "👤 User"
                print(f"{speaker}: {msg.message_text}")

                if msg.flight_a_data or msg.flight_b_data:
                    print("  (Flight comparison data attached)")
            print()

        # Iterations
        iterations = db.query(LILOIteration).filter(
            LILOIteration.lilo_session_id == session.id
        ).order_by(LILOIteration.iteration_number).all()

        if iterations:
            print("🔄 Iterations:")
            print("-" * 80)
            for iteration in iterations:
                print(f"\nIteration {iteration.iteration_number}:")
                print(f"  User Responses: {json.dumps(iteration.user_responses, indent=2)}")
                if iteration.utility_function_params:
                    print(f"  Utility Function Params: {json.dumps(iteration.utility_function_params, indent=2)}")
                if iteration.acquisition_value:
                    print(f"  Acquisition Values: {json.dumps(iteration.acquisition_value, indent=2)}")
            print()

        # Final rankings
        rankings = db.query(LILOFinalRanking).filter(
            LILOFinalRanking.lilo_session_id == session.id
        ).order_by(LILOFinalRanking.rank).all()

        if rankings:
            # Calculate utility statistics
            utility_scores = [r.utility_score for r in rankings]
            print("📈 Learned Utility Function Statistics:")
            print(f"  Max Utility:  {max(utility_scores):.6f}")
            print(f"  Min Utility:  {min(utility_scores):.6f}")
            print(f"  Average:      {sum(utility_scores)/len(utility_scores):.6f}")
            print(f"  Range:        {max(utility_scores) - min(utility_scores):.6f}")
            print()

            print("🏆 Top 20 Flight Rankings (by Learned Utility):")
            print("-" * 80)

            table_data = []
            for ranking in rankings[:20]:  # Show top 20
                flight = ranking.flight_data
                table_data.append([
                    ranking.rank,
                    f"{ranking.utility_score:.6f}",
                    flight.get('price', 'N/A'),
                    flight.get('duration_min', 'N/A'),
                    flight.get('stops', 'N/A'),
                    flight.get('airline', 'N/A')[:20] if flight.get('airline') else 'N/A'
                ])

            headers = ["Rank", "Utility Score", "Price", "Duration (min)", "Stops", "Airline"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print(f"\nShowing top 20 of {len(rankings)} total flights")
            print()

            # Show bottom 5 as well
            if len(rankings) > 5:
                print("⬇️  Bottom 5 Flights (Lowest Utility):")
                print("-" * 80)
                bottom_data = []
                for ranking in rankings[-5:]:
                    flight = ranking.flight_data
                    bottom_data.append([
                        ranking.rank,
                        f"{ranking.utility_score:.6f}",
                        flight.get('price', 'N/A'),
                        flight.get('duration_min', 'N/A'),
                        flight.get('stops', 'N/A'),
                        flight.get('airline', 'N/A')[:20] if flight.get('airline') else 'N/A'
                    ])
                print(tabulate(bottom_data, headers=headers, tablefmt="grid"))
                print()

        print("="*80)

    finally:
        db.close()


def export_lilo_csv(identifier: str, output_file: str = None):
    """Export LILO final rankings to CSV file."""
    db = SessionLocal()
    try:
        session = db.query(LILOSession).filter(
            (LILOSession.session_id == identifier) |
            (LILOSession.completion_token == identifier)
        ).first()

        if not session:
            print(f"❌ No LILO session found for identifier: {identifier}")
            return

        rankings = db.query(LILOFinalRanking).filter(
            LILOFinalRanking.lilo_session_id == session.id
        ).order_by(LILOFinalRanking.rank).all()

        if not rankings:
            print("❌ No final rankings found for this session.")
            return

        # Generate filename if not provided
        if not output_file:
            output_file = f"lilo_rankings_{session.completion_token or session.id}.csv"

        # Write CSV
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Rank', 'Utility Score', 'Price', 'Duration (min)', 'Stops',
                'Departure Time', 'Arrival Time', 'Airline', 'Flight ID'
            ])

            # Data
            for ranking in rankings:
                flight = ranking.flight_data
                writer.writerow([
                    ranking.rank,
                    f"{ranking.utility_score:.6f}",
                    flight.get('price', 'N/A'),
                    flight.get('duration_min', 'N/A'),
                    flight.get('stops', 'N/A'),
                    flight.get('departure_time', 'N/A'),
                    flight.get('arrival_time', 'N/A'),
                    flight.get('airline', 'N/A'),
                    flight.get('id', 'N/A')
                ])

        print(f"✅ Exported {len(rankings)} flights to: {output_file}")

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # No arguments - list all sessions
        list_all_lilo_sessions()
        print("\nUsage:")
        print("  python view_lilo_results.py                    # List all sessions")
        print("  python view_lilo_results.py <identifier>       # View specific session details")
        print("  python view_lilo_results.py <identifier> csv   # Export to CSV")
        print("\nIdentifier can be: session_id or completion_token")
    elif len(sys.argv) == 2:
        # View specific session
        view_lilo_session(sys.argv[1])
    elif len(sys.argv) >= 3 and sys.argv[2].lower() == 'csv':
        # Export to CSV
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        export_lilo_csv(sys.argv[1], output_file)
    else:
        print("Invalid arguments. Usage:")
        print("  python view_lilo_results.py                    # List all sessions")
        print("  python view_lilo_results.py <identifier>       # View specific session")
        print("  python view_lilo_results.py <identifier> csv [output_file.csv]  # Export to CSV")
