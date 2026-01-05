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
    Get summary of all complete research sessions (grouped by completion token).

    Returns:
        List of session summary dicts with key metrics from all stages
    """
    db = SessionLocal()
    try:
        # Get all completion tokens
        completion_tokens = db.query(CompletionToken).order_by(desc(CompletionToken.created_at)).all()

        summaries = []
        for comp_token in completion_tokens:
            token = comp_token.token
            session_id = comp_token.session_id

            # Get search data
            search = db.query(Search).filter(
                or_(Search.completion_token == token, Search.session_id == session_id)
            ).first()

            # Get survey
            survey = db.query(SurveyResponse).filter(
                or_(SurveyResponse.completion_token == token, SurveyResponse.session_id == session_id)
            ).first()

            # Get cross-validation count
            cv_count = db.query(CrossValidation).filter(
                or_(CrossValidation.completion_token == token, CrossValidation.session_id == session_id)
            ).count()

            # Get LILO session
            lilo = db.query(LILOSession).filter(
                or_(LILOSession.completion_token == token, LILOSession.session_id == session_id)
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

            summaries.append({
                'completion_token': token,
                'session_id': session_id,
                'completed_at': comp_token.created_at,
                'has_search': search is not None,
                'has_survey': survey is not None,
                'has_cv': cv_count > 0,
                'has_lilo': lilo is not None,
                'search_prompt': search.prompt if search else None,
                'origin': search.origin if search else None,
                'destination': search.destination if search else None,
                'survey_satisfaction': survey.satisfaction if survey else None,
                'cv_count': cv_count,
                'lilo_messages': lilo_messages,
                'lilo_rankings': lilo_rankings,
                'lilo_completed': lilo.completed_at is not None if lilo else False
            })

        return summaries

    finally:
        db.close()


def get_complete_session_detail(completion_token: str) -> Optional[Dict]:
    """
    Get complete detailed information about a research session.
    Includes search, rankings, survey, cross-validation, and LILO data.

    Args:
        completion_token: Completion token for the session

    Returns:
        Detailed session dict with all data, or None if not found
    """
    db = SessionLocal()
    try:
        # Get completion token record
        comp_token = db.query(CompletionToken).filter(CompletionToken.token == completion_token).first()

        if not comp_token:
            return None

        session_id = comp_token.session_id
        result = {
            'completion_token': completion_token,
            'session_id': session_id,
            'completed_at': comp_token.created_at
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
