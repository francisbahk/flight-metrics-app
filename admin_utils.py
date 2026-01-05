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

    Returns:
        List of session summary dicts with key metrics from all stages
    """
    db = SessionLocal()
    try:
        # Get all searches (this includes both completed and in-progress sessions)
        try:
            searches = db.query(Search).order_by(desc(Search.created_at)).all()
        except Exception as e:
            print(f"Error querying searches: {e}")
            # Fall back to completion tokens only
            completion_tokens = db.query(CompletionToken).order_by(desc(CompletionToken.created_at)).all()
            searches = []
            for comp_token in completion_tokens:
                search = db.query(Search).filter(
                    Search.session_id == comp_token.session_id
                ).first()
                if search:
                    searches.append(search)

        summaries = []
        seen_sessions = set()  # Track unique sessions

        for search in searches:
            session_id = search.session_id
            token = search.completion_token

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
                    or_(CrossValidation.completion_token == token, CrossValidation.session_id == session_id)
                ).count()
            else:
                cv_count = db.query(CrossValidation).filter(
                    CrossValidation.session_id == session_id
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
                'search_prompt': search.prompt,
                'origin': search.origin,
                'destination': search.destination,
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

    Args:
        identifier: Completion token or session_id

    Returns:
        Detailed session dict with all data, or None if not found
    """
    db = SessionLocal()
    try:
        # Try to find by completion token first
        comp_token = db.query(CompletionToken).filter(CompletionToken.token == identifier).first()

        if comp_token:
            session_id = comp_token.session_id
            completion_token = identifier
            completed_at = comp_token.created_at
        else:
            # Try to find by session_id
            search = db.query(Search).filter(Search.session_id == identifier).first()
            if not search:
                return None
            session_id = identifier
            completion_token = search.completion_token or 'In Progress'
            completed_at = search.created_at

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
                'origin': search.origin,
                'destination': search.destination,
                'departure_date': search.departure_date,
                'return_date': search.return_date,
                'prompt': search.prompt,
                'search_method': search.search_method,
                'flights_count': search.flights_count,
                'created_at': search.created_at
            }
        else:
            result['search'] = None

        # 2. Get user rankings
        rankings = db.query(UserRanking).filter(
            or_(UserRanking.completion_token == completion_token, UserRanking.session_id == session_id)
        ).order_by(UserRanking.rank_position).all()

        result['user_rankings'] = [{
            'rank': r.rank_position,
            'flight_data': r.flight_data,
            'created_at': r.created_at
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
            or_(CrossValidation.completion_token == completion_token, CrossValidation.session_id == session_id)
        ).all()

        result['cross_validation'] = [{
            'flight_id': cv.flight_id,
            'flight_data': cv.flight_data,
            'user_selected': cv.user_selected,
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


def export_session_csv(completion_token: str) -> Optional[str]:
    """
    Export complete session data to CSV string (LILO rankings if available).

    Args:
        completion_token: Completion token for the session

    Returns:
        CSV string or None if session not found
    """
    db = SessionLocal()
    try:
        comp_token = db.query(CompletionToken).filter(CompletionToken.token == completion_token).first()
        if not comp_token:
            return None

        session_id = comp_token.session_id

        # Try to get LILO rankings first
        lilo = db.query(LILOSession).filter(
            or_(LILOSession.completion_token == completion_token, LILOSession.session_id == session_id)
        ).first()

        if lilo:
            rankings = db.query(LILOFinalRanking).filter(
                LILOFinalRanking.lilo_session_id == lilo.id
            ).order_by(LILOFinalRanking.rank).all()

            if rankings:
                import csv
                from io import StringIO

                output = StringIO()
                writer = csv.writer(output)

                writer.writerow([
                    'Rank', 'Utility Score', 'Price', 'Duration (min)', 'Stops',
                    'Departure Time', 'Arrival Time', 'Airline', 'Flight ID'
                ])

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

                return output.getvalue()

        return None

    finally:
        db.close()
