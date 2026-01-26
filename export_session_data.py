"""
Export comprehensive session data to CSV.

Each row represents data spread across multiple sections.
Total rows = max(Amadeus flights, LILO questions, CV flights, survey questions)
"""

import csv
import json
from typing import Dict, List, Optional
from backend.db import (
    SessionLocal, AccessToken, Search, LILOSession, LILOIteration,
    LILOFinalRanking, SurveyResponse, CrossValidation, FlightShown, UserRanking,
    LILOChatMessage
)


def export_session_to_csv(token: str, output_file: str = None) -> str:
    """
    Export all data for a participant token to CSV.

    Args:
        token: Access token or completion token
        output_file: Output CSV filename (default: {token}_data.csv)

    Returns:
        Path to generated CSV file
    """
    if not output_file:
        output_file = f"{token}_data.csv"

    db = SessionLocal()

    try:
        # Get all searches for this token
        searches = db.query(Search).filter(
            (Search.session_id == token) | (Search.completion_token == token)
        ).all()

        if not searches:
            print(f"No data found for token {token}")
            return None

        # Process each search (only completed LILO sessions)
        all_rows = []

        for search in searches:
            search_id = search.search_id
            prompt = search.user_prompt

            # ONLY include completed LILO sessions
            lilo_session = db.query(LILOSession).filter_by(search_id=search_id).first()
            if not lilo_session or lilo_session.completed_at is None:
                print(f"Skipping incomplete session: search_id={search_id}")
                continue

            # Get ALL flights from Amadeus
            all_flights = search.listen_ranked_flights_json or search.amadeus_flights_json or []

            if not all_flights:
                print(f"No flight data found for search_id={search_id}")
                continue

            # Get user rankings (top 5)
            user_rankings = {}  # flight_id -> rank
            rankings = db.query(UserRanking).filter_by(search_id=search_id).all()
            for r in rankings:
                flight_shown = db.query(FlightShown).filter_by(id=r.flight_id).first()
                if flight_shown:
                    flight_id = flight_shown.flight_data.get('id')
                    user_rankings[flight_id] = r.user_rank

            # Get LILO questions and responses (actual question text)
            lilo_question_texts = []
            lilo_responses = []

            # Get ALL chat messages ordered
            all_chat_messages = db.query(LILOChatMessage).filter_by(
                lilo_session_id=lilo_session.id
            ).order_by(LILOChatMessage.round_number, LILOChatMessage.message_index).all()

            # Match questions with user responses by looking at adjacent messages
            # A question is a bot message that contains '?' (anywhere) and is followed by a user response
            for i, msg in enumerate(all_chat_messages):
                # If this is a bot message that contains a question mark
                if msg.is_bot == 1 and '?' in msg.message_text:
                    # Look for the next user message as the answer
                    for j in range(i + 1, len(all_chat_messages)):
                        next_msg = all_chat_messages[j]
                        if next_msg.is_bot == 0:  # User response
                            lilo_question_texts.append(msg.message_text)
                            lilo_responses.append(next_msg.message_text)
                            break

            # Get iterations for utility function
            iterations = db.query(LILOIteration).filter_by(
                lilo_session_id=lilo_session.id
            ).order_by(LILOIteration.iteration_number).all()

            # Get utility function (from last iteration's params)
            utility_function = None
            if iterations:
                last_iteration = iterations[-1]
                if last_iteration.utility_function_params:
                    utility_function = json.dumps(last_iteration.utility_function_params)

            # Get LILO final rankings
            lilo_rankings = {}  # flight_id -> rank
            final_ranks = db.query(LILOFinalRanking).filter_by(
                lilo_session_id=lilo_session.id
            ).order_by(LILOFinalRanking.rank).all()

            for fr in final_ranks:
                flight_id = fr.flight_data.get('id')
                if fr.rank <= 5:
                    lilo_rankings[flight_id] = fr.rank

            # Get cross validation data - what THIS user ranked (someone else's prompt)
            # Find what prompt THIS user reviewed
            cross_val = db.query(CrossValidation).filter_by(
                reviewer_session_id=search.session_id
            ).first()

            cv_prompt = None
            cv_token = None
            cv_flights = []
            cv_rankings = {}  # flight_id -> rank

            if cross_val:
                cv_prompt = cross_val.reviewed_prompt  # The OTHER person's prompt
                cv_token = cross_val.reviewed_session_id  # The OTHER person's token

                # Get rankings THIS user gave (selected_flights_data has the top 5 in ranked order)
                if cross_val.selected_flights_data:
                    for rank, flight in enumerate(cross_val.selected_flights_data, 1):
                        flight_id = flight.get('id')
                        cv_rankings[flight_id] = rank

                # Get ALL flights from the reviewed session
                reviewed_search = db.query(Search).filter(
                    (Search.session_id == cv_token) | (Search.completion_token == cv_token)
                ).first()

                if reviewed_search:
                    # Get all flights from that OTHER session
                    cv_flights = reviewed_search.listen_ranked_flights_json or reviewed_search.amadeus_flights_json or []

            # Get survey responses
            survey = db.query(SurveyResponse).filter_by(
                session_id=search.session_id
            ).first()

            survey_questions = []
            survey_answers = []
            if survey:
                survey_data = {
                    'Satisfaction (1-5)': survey.satisfaction,
                    'Ease of Use (1-5)': survey.ease_of_use,
                    'Encountered Issues': survey.encountered_issues,
                    'Issues Description': survey.issues_description or '',
                    'Search Method': survey.search_method,
                    'Understood Ranking (1-5)': survey.understood_ranking,
                    'Helpful Features': str(survey.helpful_features) if survey.helpful_features else '',
                    'Flights Matched (1-5)': survey.flights_matched,
                    'Confusing/Frustrating': survey.confusing_frustrating or '',
                    'Missing Features': survey.missing_features or '',
                    'Would Use Again': survey.would_use_again,
                    'Would Use Again Reason': survey.would_use_again_reason or '',
                    'Compared to Others (1-5)': survey.compared_to_others,
                    'Additional Comments': survey.additional_comments or ''
                }
                for q, a in survey_data.items():
                    survey_questions.append(q)
                    survey_answers.append(str(a) if a is not None else '')

            # Determine total number of rows
            num_rows = max(
                len(all_flights),
                len(lilo_question_texts),
                len(cv_flights),
                len(survey_questions)
            )

            # Build rows
            for idx in range(num_rows):
                row = {}

                # Prompt and ID (only first row)
                row['prompt'] = prompt if idx == 0 else ''
                row['id'] = token if idx == 0 else ''

                # Flight data
                if idx < len(all_flights):
                    flight = all_flights[idx]
                    flight_id = flight.get('id')

                    row.update({
                        'unique_id': flight.get('id', ''),
                        'rank': user_rankings.get(flight_id, ''),
                        'name': flight.get('airline', ''),
                        'origin': flight.get('origin', ''),
                        'destination': flight.get('destination', ''),
                        'departure_time': flight.get('departure_time', ''),
                        'arrival_time': flight.get('arrival_time', ''),
                        'stops': flight.get('stops', ''),
                        'price': flight.get('price', ''),
                        'duration_min': flight.get('duration_min', ''),
                    })
                else:
                    # Empty flight data
                    row.update({
                        'unique_id': '', 'rank': '', 'name': '', 'origin': '',
                        'destination': '', 'departure_time': '', 'arrival_time': '',
                        'stops': '', 'price': '', 'duration_min': '',
                    })

                # LILO section
                row['questions'] = lilo_question_texts[idx] if idx < len(lilo_question_texts) else ''
                row['responses'] = lilo_responses[idx] if idx < len(lilo_responses) else ''
                row['utility_function'] = utility_function if idx == 0 else ''

                # Cross validation section
                row['prompt_cross'] = cv_prompt if idx == 0 else ''
                row['id_cross'] = cv_token if idx == 0 else ''

                if idx < len(cv_flights):
                    cv_flight = cv_flights[idx]
                    cv_flight_id = cv_flight.get('id')
                    row.update({
                        'unique_id_cross': cv_flight_id,
                        'rank_cross': cv_rankings.get(cv_flight_id, ''),
                        'name_cross': cv_flight.get('airline', ''),
                        'origin_cross': cv_flight.get('origin', ''),
                        'destination_cross': cv_flight.get('destination', ''),
                        'departure_time_cross': cv_flight.get('departure_time', ''),
                        'arrival_time_cross': cv_flight.get('arrival_time', ''),
                        'stops_cross': cv_flight.get('stops', ''),
                        'price_cross': cv_flight.get('price', ''),
                        'duration_min_cross': cv_flight.get('duration_min', ''),
                    })
                else:
                    # Empty cross validation data
                    row.update({
                        'unique_id_cross': '', 'rank_cross': '', 'name_cross': '',
                        'origin_cross': '', 'destination_cross': '',
                        'departure_time_cross': '', 'arrival_time_cross': '',
                        'stops_cross': '', 'price_cross': '', 'duration_min_cross': '',
                    })

                # Survey section
                row['survey_questions'] = survey_questions[idx] if idx < len(survey_questions) else ''
                row['survey_answers'] = survey_answers[idx] if idx < len(survey_answers) else ''

                all_rows.append(row)

        # Write CSV
        if all_rows:
            fieldnames = [
                # Prompt and ID
                'prompt', 'id',

                # Flight data section
                'unique_id', 'rank', 'name', 'origin', 'destination',
                'departure_time', 'arrival_time', 'stops', 'price', 'duration_min',

                # LILO section
                'questions', 'responses', 'utility_function',

                # Cross validation section
                'prompt_cross', 'id_cross',
                'unique_id_cross', 'rank_cross', 'name_cross', 'origin_cross',
                'destination_cross', 'departure_time_cross', 'arrival_time_cross',
                'stops_cross', 'price_cross', 'duration_min_cross',

                # Survey section
                'survey_questions', 'survey_answers'
            ]

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

            print(f"✓ Exported {len(all_rows)} rows to {output_file}")
            return output_file
        else:
            print(f"No data to export for token {token}")
            return None

    finally:
        db.close()


def export_simple_rankings_csv(token: str, output_file: str = None) -> str:
    """
    Export simplified rankings data (prompt + ranked flights only).

    Columns: prompt, id, unique_id, rank, name, origin, destination,
             departure_time, arrival_time, stops, price, duration_min

    Args:
        token: Access token or completion token
        output_file: Output CSV filename (default: {token}_rankings.csv)

    Returns:
        Path to generated CSV file
    """
    if not output_file:
        output_file = f"{token}_rankings.csv"

    db = SessionLocal()

    try:
        # Get all searches for this token
        searches = db.query(Search).filter(
            (Search.session_id == token) | (Search.completion_token == token)
        ).all()

        if not searches:
            print(f"No data found for token {token}")
            return None

        all_rows = []

        for search in searches:
            search_id = search.search_id
            prompt = search.user_prompt

            # Get ALL flights
            all_flights = search.listen_ranked_flights_json or search.amadeus_flights_json or []

            if not all_flights:
                print(f"No flight data found for search_id={search_id}")
                continue

            # Get user rankings (top 5)
            user_rankings = {}  # flight_id -> rank
            rankings = db.query(UserRanking).filter_by(search_id=search_id).all()
            for r in rankings:
                flight_shown = db.query(FlightShown).filter_by(id=r.flight_id).first()
                if flight_shown:
                    flight_id = flight_shown.flight_data.get('id')
                    user_rankings[flight_id] = r.user_rank

            # Create one row per flight
            # Only include prompt and id in the first row for this search
            is_first_row_for_search = (len(all_rows) == 0 or all_rows[-1].get('prompt') != '')

            for idx, flight in enumerate(all_flights):
                flight_id = flight.get('id')
                rank = user_rankings.get(flight_id, '')

                row = {
                    'prompt': prompt if idx == 0 else '',
                    'id': search_id if idx == 0 else '',
                    'unique_id': flight_id,
                    'rank': rank,
                    'name': flight.get('airline', ''),
                    'origin': flight.get('origin', ''),
                    'destination': flight.get('destination', ''),
                    'departure_time': flight.get('departure_time', ''),
                    'arrival_time': flight.get('arrival_time', ''),
                    'stops': flight.get('stops', ''),
                    'price': flight.get('price', ''),
                    'duration_min': flight.get('duration_min', ''),
                }

                all_rows.append(row)

        # Write CSV
        if all_rows:
            fieldnames = [
                'prompt', 'id', 'unique_id', 'rank', 'name', 'origin', 'destination',
                'departure_time', 'arrival_time', 'stops', 'price', 'duration_min'
            ]

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

            print(f"✓ Exported {len(all_rows)} rows to {output_file}")
            return output_file
        else:
            print(f"No data to export for token {token}")
            return None

    finally:
        db.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python export_session_data.py <token> [output_file.csv]")
        sys.exit(1)

    token = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    export_session_to_csv(token, output_file)
