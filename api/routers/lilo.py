"""
LILO router - Interactive preference learning with 3 iterations using Gemini
"""
from fastapi import APIRouter, HTTPException
from ..models.requests import LILOInitRequest, LILORoundRequest, LILOFinalRequest
from ..models.responses import LILOInitResponse, LILORoundResponse, LILOFinalResponse
from pydantic import BaseModel, Field
from typing import Dict, List
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lilo_integration import StreamlitLILOBridge
from backend.db import SessionLocal, LILOSession, LILOIteration

router = APIRouter()

# Global LILO bridge (manages all sessions internally)
lilo_bridge = StreamlitLILOBridge()


# New request models for Gemini-powered LILO
class GenerateQuestionsRequest(BaseModel):
    """Generate initial preference questions"""
    user_prompt: str = Field(..., description="User's natural language query")


class RankWithFeedbackRequest(BaseModel):
    """Rank flights based on feedback"""
    session_id: str
    flights: List[Dict]
    feedback: str = Field(..., description="User feedback")
    initial_answers: Dict = Field(default={}, description="Initial preference answers")


@router.post("/init", response_model=LILOInitResponse)
async def initialize_lilo(request: LILOInitRequest):
    """
    Initialize LILO session and get initial preference questions.
    """
    try:
        # Create LILO session using language_bo_code
        session = lilo_bridge.create_session(
            session_id=request.session_id,
            flights_data=request.flights
        )

        # Get initial questions
        questions = lilo_bridge.get_initial_questions(request.session_id)

        # Save to database
        db = SessionLocal()
        try:
            lilo_session = LILOSession(
                session_id=request.session_id,
                user_id=None,
                num_rounds=0
            )
            db.add(lilo_session)
            db.commit()
        finally:
            db.close()

        return LILOInitResponse(
            session_id=request.session_id,
            round_number=0,
            flights_shown=[],  # No flights shown yet, just questions
            questions=questions,
            message="Please answer these questions to help us understand your preferences"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LILO init failed: {str(e)}")


@router.post("/round", response_model=LILORoundResponse)
async def submit_lilo_round(request: LILORoundRequest):
    """
    Submit LILO round feedback and get next round.

    Expects question_answers (Dict[str, str]) mapping questions to user's answers.
    """
    try:
        # Check if session exists
        if request.session_id not in lilo_bridge.sessions:
            raise HTTPException(status_code=404, detail="LILO session not found")

        # Run iteration with user's question answers
        user_feedback = request.question_answers or {}
        flights_to_show, next_questions = lilo_bridge.run_iteration(
            session_id=request.session_id,
            user_feedback=user_feedback
        )

        # Save round to database
        db = SessionLocal()
        try:
            lilo_iteration = LILOIteration(
                session_id=request.session_id,
                iteration_number=request.round_number,
                flights_shown=flights_to_show,
                user_feedback=str(user_feedback),
                generated_questions=next_questions
            )
            db.add(lilo_iteration)

            # Update session
            session = db.query(LILOSession).filter_by(session_id=request.session_id).first()
            if session:
                session.num_rounds = request.round_number

            db.commit()
        finally:
            db.close()

        # LILO has 2 iterations - if we just completed iteration 2, it's final
        is_final = request.round_number >= 2

        return LILORoundResponse(
            session_id=request.session_id,
            round_number=request.round_number,
            flights_shown=flights_to_show,
            questions=next_questions if not is_final else [],
            feedback_summary="",
            is_final_round=is_final
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LILO round failed: {str(e)}")


@router.post("/final", response_model=LILOFinalResponse)
async def get_lilo_final_ranking(request: LILOFinalRequest):
    """
    Get final LILO ranking after 2 rounds of feedback.

    Returns top 10 flights ranked by learned utility function.
    """
    try:
        # Check if session exists
        if request.session_id not in lilo_bridge.sessions:
            raise HTTPException(status_code=404, detail="LILO session not found")

        # Get session to access all flights
        session_obj = lilo_bridge.sessions[request.session_id]
        all_flights = session_obj.flights_data

        # Compute final rankings
        ranked_results = lilo_bridge.compute_final_rankings(
            session_id=request.session_id,
            all_flights=all_flights
        )

        # Extract top 10 flights and their scores
        final_rankings = [r['flight'] for r in ranked_results[:10]]
        utility_scores = [r['utility_score'] for r in ranked_results]

        # Save final results to database
        db = SessionLocal()
        try:
            session = db.query(LILOSession).filter_by(session_id=request.session_id).first()
            if session:
                session.final_utility_scores = utility_scores
                session.completed_at = db.func.now()

                # Save utility function parameters to last iteration
                last_iteration = db.query(LILOIteration).filter_by(
                    lilo_session_id=session.id
                ).order_by(LILOIteration.iteration_number.desc()).first()

                if last_iteration:
                    # Extract utility model info from LILO session
                    lilo_session_obj = lilo_bridge.sessions.get(request.session_id)
                    if lilo_session_obj and lilo_session_obj.optimizer:
                        optimizer = lilo_session_obj.optimizer

                        # Save utility scores as utility function representation
                        utility_func_data = {
                            'utility_scores_sample': utility_scores[:10],  # Top 10 utilities
                            'num_trials': optimizer.trial_index if hasattr(optimizer, 'trial_index') else None,
                            'model_type': 'pairwise_preference',
                        }

                        last_iteration.utility_function_params = utility_func_data

            db.commit()
        finally:
            db.close()

        # Clean up session from memory
        if request.session_id in lilo_bridge.sessions:
            del lilo_bridge.sessions[request.session_id]

        return LILOFinalResponse(
            session_id=request.session_id,
            final_rankings=final_rankings,
            utility_scores=utility_scores,
            feedback_summary=""
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LILO final failed: {str(e)}")


# New Gemini-powered endpoints for 3-iteration workflow

@router.post("/generate-questions")
async def generate_preference_questions(request: GenerateQuestionsRequest):
    """
    Generate initial preference questions using Gemini.

    Returns 5 questions tailored to the user's search query.
    """
    try:
        optimizer = LILOOptimizer()
        questions = optimizer.generate_initial_questions(request.user_prompt)

        return {
            "questions": questions,
            "message": "Answer these questions to help us understand your preferences"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


@router.post("/rank-with-feedback")
async def rank_flights_with_feedback(request: RankWithFeedbackRequest):
    """
    Rank flights using Gemini based on user feedback.

    Uses feedback and initial answers to estimate utility scores.
    """
    try:
        # Get or create optimizer
        optimizer = active_sessions.get(request.session_id)
        if not optimizer:
            optimizer = LILOOptimizer()
            active_sessions[request.session_id] = optimizer

        # Store initial answers if provided
        if request.initial_answers:
            optimizer.initial_answers = request.initial_answers

        # Add feedback to history
        optimizer.feedback_history.append(request.feedback)

        # Estimate utilities using Gemini
        utilities = optimizer.estimate_utilities(request.flights)

        # Sort flights by utility (descending)
        sorted_flights = sorted(
            zip(request.flights, utilities),
            key=lambda x: x[1],
            reverse=True
        )

        # Return all ranked flights
        ranked_flights = [f for f, _ in sorted_flights]

        return {
            "ranked_flights": ranked_flights,
            "feedback_summary": f"Processed {len(optimizer.feedback_history)} rounds of feedback"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking with feedback failed: {str(e)}")