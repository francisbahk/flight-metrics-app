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

from backend.lilo import LILOOptimizer
from backend.db import SessionLocal, LILOSession, LILORound

router = APIRouter()

# In-memory storage for active LILO sessions (could use Redis in production)
active_sessions: Dict[str, LILOOptimizer] = {}


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
    Initialize LILO session.

    Returns 15 diverse flights for Round 1.
    """
    try:
        # Create LILO optimizer
        optimizer = LILOOptimizer()
        optimizer.all_flights = request.flights

        # Select Round 1 candidates (15 diverse flights)
        flights_shown = optimizer.select_candidates(
            all_flights=request.flights,
            round_num=1,
            n_candidates=15
        )

        # Store optimizer in memory
        active_sessions[request.session_id] = optimizer

        # Save to database
        db = SessionLocal()
        try:
            lilo_session = LILOSession(
                session_id=request.session_id,
                user_id=None,  # Set from frontend if needed
                num_rounds=0
            )
            db.add(lilo_session)
            db.commit()
        finally:
            db.close()

        return LILOInitResponse(
            session_id=request.session_id,
            round_number=1,
            flights_shown=flights_shown,
            message="Round 1: Select and rank your top 5 flights, then provide feedback"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LILO init failed: {str(e)}")


@router.post("/round", response_model=LILORoundResponse)
async def submit_lilo_round(request: LILORoundRequest):
    """
    Submit LILO round feedback and get next round (or finish).

    Round 1: User ranks + feedback → Return Round 2 flights + questions
    Round 2: User ranks + feedback → Return final message
    """
    try:
        # Get optimizer from memory
        optimizer = active_sessions.get(request.session_id)
        if not optimizer:
            raise HTTPException(status_code=404, detail="LILO session not found")

        # Get all flights
        all_flights = optimizer.all_flights

        # Run this round
        round_result = optimizer.run_round(
            all_flights=all_flights,
            round_num=request.round_number,
            user_rankings=request.user_rankings,
            user_feedback=request.user_feedback,
            n_candidates=15
        )

        # Save round to database
        db = SessionLocal()
        try:
            lilo_round = LILORound(
                session_id=request.session_id,
                round_number=request.round_number,
                flights_shown=round_result['flights_shown'],
                user_rankings=request.user_rankings,
                user_feedback=request.user_feedback,
                generated_questions=round_result.get('questions'),
                extracted_preferences=None  # Could extract from optimizer
            )
            db.add(lilo_round)

            # Update session
            session = db.query(LILOSession).filter_by(session_id=request.session_id).first()
            if session:
                session.num_rounds = request.round_number
                session.feedback_summary = round_result.get('feedback_summary')

            db.commit()
        finally:
            db.close()

        # Determine if this is the final round
        is_final = request.round_number >= 2

        if is_final:
            # Round 2 complete - ready for final ranking
            return LILORoundResponse(
                session_id=request.session_id,
                round_number=request.round_number,
                flights_shown=round_result['flights_shown'],
                questions=[],
                feedback_summary=round_result.get('feedback_summary'),
                is_final_round=True
            )
        else:
            # Round 1 complete - show Round 2 flights
            return LILORoundResponse(
                session_id=request.session_id,
                round_number=request.round_number + 1,
                flights_shown=round_result['flights_shown'],
                questions=round_result.get('questions', []),
                feedback_summary=round_result.get('feedback_summary'),
                is_final_round=False
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
        # Get optimizer from memory
        optimizer = active_sessions.get(request.session_id)
        if not optimizer:
            raise HTTPException(status_code=404, detail="LILO session not found")

        # Get final ranking
        final_rankings = optimizer.get_final_ranking(
            all_flights=optimizer.all_flights,
            top_k=10
        )

        # Get utility scores
        utility_scores = optimizer.estimate_utilities(optimizer.all_flights)

        # Save final results to database
        db = SessionLocal()
        try:
            session = db.query(LILOSession).filter_by(session_id=request.session_id).first()
            if session:
                session.final_utility_scores = utility_scores
                session.completed_at = db.func.now()
            db.commit()
        finally:
            db.close()

        # Clean up session from memory
        if request.session_id in active_sessions:
            del active_sessions[request.session_id]

        return LILOFinalResponse(
            session_id=request.session_id,
            final_rankings=final_rankings,
            utility_scores=utility_scores,  # All flights
            feedback_summary=optimizer.feedback_summary
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