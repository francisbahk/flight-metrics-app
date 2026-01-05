"""
Admin utilities for viewing complete session data and research results.
"""

from backend.db import (
    SessionLocal, Search, UserRanking, SurveyResponse, CrossValidation,
    LILOSession, LILOChatMessage, LILOIteration, LILOFinalRanking,
    CompletionToken, FlightShown, SequentialEvaluation
)
from sqlalchemy import desc, or_
from datetime import datetime
from typing import List, Dict, Optional
import json


def get_all_sessions_summary() -> List[Dict]:
    """
    Get summary of all research sessions (both completed and in-progress).
    Filters out data before January 1, 2026 and DEMO sessions.

    Returns:
        List of session summary dicts with key metrics from all stages
    """
    db = SessionLocal()
    try:
        # Filter cutoff date: January 1, 2026
        cutoff_date = datetime(2026, 1, 1)

        # Get all searches (this includes both completed and in-progress sessions)
        try:
            # Try ordering by created_at DESC with date filter
            searches = db.query(Search).filter(
                Search.created_at >= cutoff_date
            ).order_by(desc(Search.created_at)).all()
        except Exception as e:
            print(f"Error querying searches with ORDER BY: {e}")
            # Fallback 1: Try without ordering
            try:
                searches = db.query(Search).filter(
                    Search.created_at >= cutoff_date
                ).all()
                # Sort in Python instead
                searches = sorted(searches, key=lambda s: s.created_at, reverse=True)
            except Exception as e2:
                print(f"Error querying searches without ORDER BY: {e2}")
                # Fallback 2: Get completion tokens first
                completion_tokens = db.query(CompletionToken).filter(
                    CompletionToken.created_at >= cutoff_date
                ).order_by(desc(CompletionToken.created_at)).all()
                searches = []
                for comp_token in completion_tokens:
                    search = db.query(Search).filter(
                        Search.session_id == comp_token.session_id,
                        Search.created_at >= cutoff_date
                    ).first()
                    if search:
                        searches.append(search)

        summaries = []
        seen_sessions = set()  # Track unique sessions

        for search in searches:
            session_id = search.session_id
            token = search.completion_token

            # Skip DEMO sessions
            if token == "DEMO":
                continue

            # Skip if we've already processed this session
            session_key = token if token else session_id
            if session_key in seen_sessions:
                continue
            seen_sessions.add(session_key)

            # Get survey
            if token:
                survey = db.query(SurveyResponse).filter(
                    or_(SurveyResponse.completion_token == token, SurveyResponse.session_id == session_id)
                ).first()
            else:
                survey = db.query(SurveyResponse).filter(
                    SurveyResponse.session_id == session_id
                ).first()

            # Get cross-validation count
            if token:
                cv_count = db.query(CrossValidation).filter(
                    or_(CrossValidation.reviewer_token == token, CrossValidation.reviewer_session_id == session_id)
                ).count()
            else:
                cv_count = db.query(CrossValidation).filter(
                    CrossValidation.reviewer_session_id == session_id
                ).count()

            # Get LILO session
            if token:
                lilo = db.query(LILOSession).filter(
                    or_(LILOSession.completion_token == token, LILOSession.session_id == session_id)
                ).first()
            else:
                lilo = db.query(LILOSession).filter(
                    LILOSession.session_id == session_id
                ).first()

            # Count LILO data if exists
            lilo_messages = 0
            lilo_rankings = 0
            if lilo:
                lilo_messages = db.query(LILOChatMessage).filter(
                    LILOChatMessage.lilo_session_id == lilo.id
                ).count()
                lilo_rankings = db.query(LILOFinalRanking).filter(
                    LILOFinalRanking.lilo_session_id == lilo.id
                ).count()

            # Get completion info
            comp_token = db.query(CompletionToken).filter(
                CompletionToken.token == token
            ).first() if token else None

            summaries.append({
                'completion_token': token or 'In Progress',
                'session_id': session_id,
                'completed_at': comp_token.created_at if comp_token else search.created_at,
                'is_completed': comp_token is not None,
                'has_search': True,  # Always true since we're querying from searches
                'has_survey': survey is not None,
                'has_cv': cv_count > 0,
                'has_lilo': lilo is not None,
                'search_prompt': search.user_prompt,
                'origin': search.parsed_origins,
                'destination': search.parsed_destinations,
                'survey_satisfaction': survey.satisfaction if survey else None,
                'cv_count': cv_count,
                'lilo_messages': lilo_messages,
                'lilo_rankings': lilo_rankings,
                'lilo_completed': lilo.completed_at is not None if lilo else False
            })

        return summaries

    except Exception as e:
        print(f"Error in get_all_sessions_summary: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        db.close()


def get_complete_session_detail(identifier: str) -> Optional[Dict]:
    """
    Get complete detailed information about a research session.
    Includes search, rankings, survey, cross-validation, and LILO data.
    Filters out DEMO sessions and data before January 1, 2026.

    Args:
        identifier: Completion token or session_id

    Returns:
        Detailed session dict with all data, or None if not found
    """
    db = SessionLocal()
    try:
        # Filter out DEMO sessions
        if identifier == "DEMO":
            return None

        # Filter cutoff date: January 1, 2026
        cutoff_date = datetime(2026, 1, 1)

        # Try to find by completion token first
        comp_token = db.query(CompletionToken).filter(
            CompletionToken.token == identifier,
            CompletionToken.created_at >= cutoff_date
        ).first()

        if comp_token:
            session_id = comp_token.session_id
            completion_token = identifier
            completed_at = comp_token.created_at
        else:
            # Try to find by session_id
            search = db.query(Search).filter(
                Search.session_id == identifier,
                Search.created_at >= cutoff_date
            ).first()
            if not search:
                return None
            session_id = identifier
            completion_token = search.completion_token or 'In Progress'
            completed_at = search.created_at

            # Additional check: skip DEMO completion tokens
            if completion_token == "DEMO":
                return None

        result = {
            'completion_token': completion_token,
            'session_id': session_id,
            'completed_at': completed_at,
            'is_completed': comp_token is not None
        }

        # 1. Get flight search data
        search = db.query(Search).filter(
            or_(Search.completion_token == completion_token, Search.session_id == session_id)
        ).first()

        if search:
            result['search'] = {
                'search_id': search.search_id,
                'origin': search.parsed_origins,
                'destination': search.parsed_destinations,
                'departure_date': search.departure_date,
                'prompt': search.user_prompt,
                'created_at': search.created_at
            }
        else:
            result['search'] = None

        # 2. Get user rankings
        # UserRanking doesn't have completion_token or session_id directly, need to join through search
        rankings = db.query(UserRanking).filter(
            UserRanking.search_id == search.search_id
        ).order_by(UserRanking.user_rank).all() if search else []

        result['user_rankings'] = [{
            'rank': r.user_rank,
            'flight_id': r.flight_id,
            'created_at': r.submitted_at
        } for r in rankings]

        # 3. Get survey response
        survey = db.query(SurveyResponse).filter(
            or_(SurveyResponse.completion_token == completion_token, SurveyResponse.session_id == session_id)
        ).first()

        if survey:
            result['survey'] = {
                'satisfaction': survey.satisfaction,
                'ease_of_use': survey.ease_of_use,
                'encountered_issues': survey.encountered_issues,
                'issues_description': survey.issues_description,
                'search_method': survey.search_method,
                'understood_ranking': survey.understood_ranking,
                'helpful_features': survey.helpful_features,
                'flights_matched': survey.flights_matched,
                'confusing_frustrating': survey.confusing_frustrating,
                'missing_features': survey.missing_features,
                'would_use_again': survey.would_use_again,
                'would_use_again_reason': survey.would_use_again_reason,
                'compared_to_others': survey.compared_to_others,
                'additional_comments': survey.additional_comments,
                'created_at': survey.created_at
            }
        else:
            result['survey'] = None

        # 4. Get cross-validation results
        cv_results = db.query(CrossValidation).filter(
            or_(CrossValidation.reviewer_token == completion_token, CrossValidation.reviewer_session_id == session_id)
        ).all()

        result['cross_validation'] = [{
            'reviewed_prompt': cv.reviewed_prompt,
            'reviewed_session_id': cv.reviewed_session_id,
            'selected_flight_ids': cv.selected_flight_ids,
            'selected_flights_data': cv.selected_flights_data,
            'created_at': cv.created_at
        } for cv in cv_results]

        # 5. Get LILO data
        lilo = db.query(LILOSession).filter(
            or_(LILOSession.completion_token == completion_token, LILOSession.session_id == session_id)
        ).first()

        if lilo:
            # Get LILO chat messages
            messages = db.query(LILOChatMessage).filter(
                LILOChatMessage.lilo_session_id == lilo.id
            ).order_by(LILOChatMessage.round_number, LILOChatMessage.message_index).all()

            chat_by_round = {}
            for msg in messages:
                if msg.round_number not in chat_by_round:
                    chat_by_round[msg.round_number] = []
                chat_by_round[msg.round_number].append({
                    'is_bot': msg.is_bot,
                    'text': msg.message_text,
                    'has_flights': msg.flight_a_data is not None or msg.flight_b_data is not None
                })

            # Get LILO iterations
            iterations = db.query(LILOIteration).filter(
                LILOIteration.lilo_session_id == lilo.id
            ).order_by(LILOIteration.iteration_number).all()

            iteration_data = [{
                'number': it.iteration_number,
                'user_responses': it.user_responses,
                'utility_params': it.utility_function_params,
                'acquisition_values': it.acquisition_value
            } for it in iterations]

            # Get LILO final rankings
            lilo_rankings = db.query(LILOFinalRanking).filter(
                LILOFinalRanking.lilo_session_id == lilo.id
            ).order_by(LILOFinalRanking.rank).all()

            utility_stats = None
            if lilo_rankings:
                utility_scores = [r.utility_score for r in lilo_rankings]
                utility_stats = {
                    'max': max(utility_scores),
                    'min': min(utility_scores),
                    'avg': sum(utility_scores) / len(utility_scores),
                    'range': max(utility_scores) - min(utility_scores)
                }

            result['lilo'] = {
                'db_id': lilo.id,
                'num_iterations': lilo.num_iterations,
                'questions_per_round': lilo.questions_per_round,
                'started_at': lilo.started_at,
                'completed_at': lilo.completed_at,
                'chat_transcript': chat_by_round,
                'iterations': iteration_data,
                'rankings': {
                    'total': len(lilo_rankings),
                    'top_10': [{
                        'rank': r.rank,
                        'utility_score': r.utility_score,
                        'flight_data': r.flight_data
                    } for r in lilo_rankings[:10]],
                    'bottom_5': [{
                        'rank': r.rank,
                        'utility_score': r.utility_score,
                        'flight_data': r.flight_data
                    } for r in lilo_rankings[-5:]] if len(lilo_rankings) > 5 else [],
                    'utility_stats': utility_stats
                }
            }
        else:
            result['lilo'] = None

        return result

    finally:
        db.close()


def export_manual_rankings_csv(identifier: str) -> Optional[str]:
    """
    Export manual rankings to CSV.

    Args:
        identifier: Completion token or session_id

    Returns:
        CSV string or None if not found
    """
    db = SessionLocal()
    try:
        # Get session detail
        detail = get_complete_session_detail(identifier)
        if not detail or not detail.get('user_rankings'):
            return None

        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        writer.writerow(['Rank', 'Flight ID', 'Ranked At'])

        for ranking in detail['user_rankings']:
            writer.writerow([
                ranking['rank'],
                ranking['flight_id'],
                ranking['created_at'].strftime('%Y-%m-%d %H:%M:%S') if ranking['created_at'] else 'N/A'
            ])

        return output.getvalue()

    finally:
        db.close()


def export_cross_validation_csv(identifier: str) -> Optional[str]:
    """
    Export cross-validation data to CSV.

    Args:
        identifier: Completion token or session_id

    Returns:
        CSV string or None if not found
    """
    db = SessionLocal()
    try:
        # Get session detail
        detail = get_complete_session_detail(identifier)
        if not detail or not detail.get('cross_validation'):
            return None

        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        writer.writerow(['Review #', 'Reviewed Session ID', 'Reviewed Prompt', 'Selected Flights Count', 'Reviewed At'])

        for i, cv in enumerate(detail['cross_validation'], 1):
            writer.writerow([
                i,
                cv['reviewed_session_id'],
                cv['reviewed_prompt'][:100],  # Truncate long prompts
                len(cv.get('selected_flight_ids', [])),
                cv['created_at'].strftime('%Y-%m-%d %H:%M:%S') if cv['created_at'] else 'N/A'
            ])

        return output.getvalue()

    finally:
        db.close()


def export_survey_csv(identifier: str) -> Optional[str]:
    """
    Export survey responses to CSV.

    Args:
        identifier: Completion token or session_id

    Returns:
        CSV string or None if not found
    """
    db = SessionLocal()
    try:
        # Get session detail
        detail = get_complete_session_detail(identifier)
        if not detail or not detail.get('survey'):
            return None

        import csv
        from io import StringIO

        survey = detail['survey']

        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(['Question', 'Response'])

        # Write all survey responses
        writer.writerow(['Satisfaction (1-5)', survey.get('satisfaction', 'N/A')])
        writer.writerow(['Ease of Use (1-5)', survey.get('ease_of_use', 'N/A')])
        writer.writerow(['Encountered Issues', survey.get('encountered_issues', 'N/A')])
        writer.writerow(['Issues Description', survey.get('issues_description', 'N/A')])
        writer.writerow(['Search Method', survey.get('search_method', 'N/A')])
        writer.writerow(['Understood Ranking (1-5)', survey.get('understood_ranking', 'N/A')])
        writer.writerow(['Helpful Features', ', '.join(survey.get('helpful_features', [])) if survey.get('helpful_features') else 'N/A'])
        writer.writerow(['Flights Matched Expectations (1-5)', survey.get('flights_matched', 'N/A')])
        writer.writerow(['Confusing/Frustrating', survey.get('confusing_frustrating', 'N/A')])
        writer.writerow(['Missing Features', survey.get('missing_features', 'N/A')])
        writer.writerow(['Would Use Again', survey.get('would_use_again', 'N/A')])
        writer.writerow(['Would Use Again Reason', survey.get('would_use_again_reason', 'N/A')])
        writer.writerow(['Compared to Others (1-5)', survey.get('compared_to_others', 'N/A')])
        writer.writerow(['Additional Comments', survey.get('additional_comments', 'N/A')])
        writer.writerow(['Completed At', survey['created_at'].strftime('%Y-%m-%d %H:%M:%S') if survey['created_at'] else 'N/A'])

        return output.getvalue()

    finally:
        db.close()


def export_lilo_full_csv(identifier: str) -> Optional[str]:
    """
    Export complete LILO conversational flow to CSV with rounds, questions, answers, and flights.

    Args:
        identifier: Completion token or session_id

    Returns:
        CSV string or None if not found
    """
    db = SessionLocal()
    try:
        # Get session detail
        detail = get_complete_session_detail(identifier)
        if not detail or not detail.get('lilo'):
            return None

        lilo = detail['lilo']

        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Write session metadata
        writer.writerow(['LILO Session Data'])
        writer.writerow(['Session ID', detail['session_id']])
        writer.writerow(['Completion Token', detail['completion_token']])
        writer.writerow(['Iterations', lilo['num_iterations']])
        writer.writerow(['Questions per Round', lilo['questions_per_round']])
        writer.writerow([])

        # Write conversational flow by round
        chat_transcript = lilo.get('chat_transcript', {})
        for round_num in sorted(chat_transcript.keys()):
            writer.writerow([f'--- ROUND {round_num} ---'])
            writer.writerow([])

            messages = chat_transcript[round_num]
            for msg in messages:
                speaker = 'Bot' if msg['is_bot'] else 'User'
                writer.writerow([speaker, msg['text']])

                if msg.get('has_flights'):
                    writer.writerow(['', '(Flight comparison data included)'])

            writer.writerow([])

        # Write final rankings
        writer.writerow(['--- FINAL RANKINGS ---'])
        writer.writerow([])
        writer.writerow([
            'Rank', 'Utility Score', 'Price', 'Duration (min)', 'Stops',
            'Departure Time', 'Arrival Time', 'Airline', 'Flight ID'
        ])

        rankings = lilo['rankings']['top_10']
        for r in rankings:
            flight = r['flight_data']
            writer.writerow([
                r['rank'],
                f"{r['utility_score']:.6f}",
                flight.get('price', 'N/A'),
                flight.get('duration_min', 'N/A'),
                flight.get('stops', 'N/A'),
                flight.get('departure_time', 'N/A'),
                flight.get('arrival_time', 'N/A'),
                flight.get('airline', 'N/A'),
                flight.get('id', 'N/A')
            ])

        # Add utility stats
        writer.writerow([])
        writer.writerow(['--- UTILITY STATISTICS ---'])
        if lilo['rankings'].get('utility_stats'):
            stats = lilo['rankings']['utility_stats']
            writer.writerow(['Max Utility', f"{stats['max']:.6f}"])
            writer.writerow(['Min Utility', f"{stats['min']:.6f}"])
            writer.writerow(['Average Utility', f"{stats['avg']:.6f}"])
            writer.writerow(['Range', f"{stats['range']:.6f}"])

        return output.getvalue()

    finally:
        db.close()


# Legacy function - kept for backward compatibility
def export_session_csv(completion_token: str) -> Optional[str]:
    """
    Export complete session data to CSV string (LILO rankings if available).
    This is a legacy function that now calls export_lilo_full_csv.

    Args:
        completion_token: Completion token for the session

    Returns:
        CSV string or None if session not found
    """
    return export_lilo_full_csv(completion_token)
