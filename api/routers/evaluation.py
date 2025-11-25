"""
Evaluation router - Human vs LLM comparison (Person A vs Person B vs LISTEN-U vs LILO)
"""
from fastapi import APIRouter, HTTPException
from ..models.requests import (
    EvaluationStartRequest,
    PersonARankingsRequest,
    PersonBRankingsRequest,
    ManualEvaluationRequest,
    LISTENEvaluationRequest,
    LILOEvaluationRequest
)
from ..models.responses import EvaluationSessionResponse
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db import SessionLocal, EvaluationSession, SequentialEvaluation

router = APIRouter()


@router.post("/start", response_model=EvaluationSessionResponse)
async def start_evaluation(request: EvaluationStartRequest):
    """
    Start evaluation session (Person A enters prompt).

    Returns eval_session_id for tracking.
    """
    try:
        db = SessionLocal()
        try:
            # Create evaluation session
            eval_session = EvaluationSession(
                session_id=request.session_id,
                person_a_user_id=request.user_id,
                person_a_prompt=request.prompt,
                person_a_rankings=[]  # Will be filled later
            )
            db.add(eval_session)
            db.commit()
            db.refresh(eval_session)

            return EvaluationSessionResponse(
                eval_session_id=request.session_id,
                search_id=0,  # Will be set when flights are searched
                person_a_prompt=request.prompt
            )

        finally:
            db.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Start evaluation failed: {str(e)}")


@router.post("/person-a/rankings")
async def submit_person_a_rankings(request: PersonARankingsRequest):
    """
    Person A submits ground truth top-5 rankings.
    """
    try:
        db = SessionLocal()
        try:
            # Update evaluation session
            eval_session = db.query(EvaluationSession).filter_by(
                session_id=request.session_id
            ).first()

            if not eval_session:
                raise HTTPException(status_code=404, detail="Evaluation session not found")

            eval_session.person_a_rankings = request.rankings
            db.commit()

            return {"status": "ok", "message": "Person A rankings saved"}

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Submit rankings failed: {str(e)}")


@router.post("/person-b/rankings")
async def submit_person_b_rankings(request: PersonBRankingsRequest):
    """
    Person B submits guessed rankings for Person A.
    """
    try:
        db = SessionLocal()
        try:
            # Update evaluation session
            eval_session = db.query(EvaluationSession).filter_by(
                session_id=request.eval_session_id
            ).first()

            if not eval_session:
                raise HTTPException(status_code=404, detail="Evaluation session not found")

            eval_session.person_b_user_id = request.user_id
            eval_session.person_b_rankings = request.rankings
            db.commit()

            return {"status": "ok", "message": "Person B rankings saved"}

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Submit rankings failed: {str(e)}")


@router.post("/algorithm/rankings/{algorithm}")
async def submit_algorithm_rankings(
    algorithm: str,
    session_id: str,
    rankings: list
):
    """
    Save algorithm rankings (LISTEN-U, LILO, Cheapest, Fastest).

    Args:
        algorithm: 'listen_u', 'lilo', 'cheapest', or 'fastest'
        session_id: Evaluation session ID
        rankings: Top-k rankings from algorithm
    """
    try:
        db = SessionLocal()
        try:
            eval_session = db.query(EvaluationSession).filter_by(
                session_id=session_id
            ).first()

            if not eval_session:
                raise HTTPException(status_code=404, detail="Evaluation session not found")

            # Update appropriate field
            if algorithm == "listen_u":
                eval_session.listen_u_rankings = rankings
            elif algorithm == "lilo":
                eval_session.lilo_rankings = rankings
            elif algorithm == "cheapest":
                eval_session.cheapest_rankings = rankings
            elif algorithm == "fastest":
                eval_session.fastest_rankings = rankings
            else:
                raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")

            db.commit()

            return {"status": "ok", "message": f"{algorithm} rankings saved"}

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Submit algorithm rankings failed: {str(e)}")


@router.get("/session/{session_id}", response_model=EvaluationSessionResponse)
async def get_evaluation_session(session_id: str):
    """
    Get evaluation session details.
    """
    try:
        db = SessionLocal()
        try:
            eval_session = db.query(EvaluationSession).filter_by(
                session_id=session_id
            ).first()

            if not eval_session:
                raise HTTPException(status_code=404, detail="Evaluation session not found")

            return EvaluationSessionResponse(
                eval_session_id=eval_session.session_id,
                search_id=eval_session.search_id or 0,
                person_a_prompt=eval_session.person_a_prompt,
                person_a_rankings=eval_session.person_a_rankings,
                person_b_rankings=eval_session.person_b_rankings,
                listen_u_rankings=eval_session.listen_u_rankings,
                lilo_rankings=eval_session.lilo_rankings,
                comparison_results=eval_session.team_draft_results
            )

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get session failed: {str(e)}")


@router.post("/compare")
async def compare_rankings(session_id: str):
    """
    Compare Person A's ground truth with Person B and algorithms.

    Calculates:
    - Overlap (how many same flights)
    - Precision@k
    - NDCG@k
    - Team Draft results
    """
    try:
        db = SessionLocal()
        try:
            eval_session = db.query(EvaluationSession).filter_by(
                session_id=session_id
            ).first()

            if not eval_session:
                raise HTTPException(status_code=404, detail="Evaluation session not found")

            # Get ground truth
            person_a_ids = [f['id'] for f in eval_session.person_a_rankings]

            # Calculate metrics for each ranker
            results = {}

            # Person B
            if eval_session.person_b_rankings:
                person_b_ids = [f['id'] for f in eval_session.person_b_rankings]
                results['person_b'] = {
                    'overlap': len(set(person_a_ids) & set(person_b_ids)),
                    'precision': len(set(person_a_ids) & set(person_b_ids)) / len(person_b_ids)
                }

            # LISTEN-U
            if eval_session.listen_u_rankings:
                listen_ids = [f['id'] for f in eval_session.listen_u_rankings]
                results['listen_u'] = {
                    'overlap': len(set(person_a_ids) & set(listen_ids)),
                    'precision': len(set(person_a_ids) & set(listen_ids)) / len(listen_ids)
                }

            # LILO
            if eval_session.lilo_rankings:
                lilo_ids = [f['id'] for f in eval_session.lilo_rankings]
                results['lilo'] = {
                    'overlap': len(set(person_a_ids) & set(lilo_ids)),
                    'precision': len(set(person_a_ids) & set(lilo_ids)) / len(lilo_ids)
                }

            # Save results
            eval_session.metrics = results
            eval_session.completed_at = db.func.now()
            db.commit()

            return results

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compare rankings failed: {str(e)}")


# Sequential Evaluation Endpoints (Manual → LISTEN → LILO)

@router.post("/sequential/manual")
async def save_manual_evaluation(request: ManualEvaluationRequest):
    """
    Save Manual method evaluation data (first step in sequential workflow).

    Creates or updates a sequential evaluation session with manual method results.
    """
    try:
        db = SessionLocal()
        try:
            # Check if session exists
            seq_eval = db.query(SequentialEvaluation).filter_by(
                session_id=request.session_id
            ).first()

            if not seq_eval:
                # Create new sequential evaluation
                seq_eval = SequentialEvaluation(
                    session_id=request.session_id,
                    user_id=request.user_id
                )
                db.add(seq_eval)

            # Update manual method data
            seq_eval.manual_search_results = request.search_results
            seq_eval.manual_rankings = request.rankings
            seq_eval.manual_completed_at = datetime.utcnow()

            db.commit()

            return {
                "status": "ok",
                "message": "Manual evaluation saved",
                "session_id": request.session_id
            }

        finally:
            db.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save manual evaluation failed: {str(e)}")


@router.post("/sequential/listen")
async def save_listen_evaluation(request: LISTENEvaluationRequest):
    """
    Save LISTEN method evaluation data (second step in sequential workflow).

    Updates sequential evaluation session with LISTEN method results.
    """
    try:
        db = SessionLocal()
        try:
            # Get existing session
            seq_eval = db.query(SequentialEvaluation).filter_by(
                session_id=request.session_id
            ).first()

            if not seq_eval:
                raise HTTPException(status_code=404, detail="Sequential evaluation session not found. Complete Manual method first.")

            # Update LISTEN method data
            seq_eval.listen_prompt = request.prompt
            seq_eval.listen_search_results = request.search_results
            seq_eval.listen_ranked_flights = request.ranked_flights
            seq_eval.listen_rankings = request.rankings
            seq_eval.listen_completed_at = datetime.utcnow()

            db.commit()

            return {
                "status": "ok",
                "message": "LISTEN evaluation saved",
                "session_id": request.session_id
            }

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save LISTEN evaluation failed: {str(e)}")


@router.post("/sequential/lilo")
async def save_lilo_evaluation(request: LILOEvaluationRequest):
    """
    Save LILO method evaluation data (final step in sequential workflow).

    Updates sequential evaluation session with LILO method results and marks workflow complete.
    """
    try:
        db = SessionLocal()
        try:
            # Get existing session
            seq_eval = db.query(SequentialEvaluation).filter_by(
                session_id=request.session_id
            ).first()

            if not seq_eval:
                raise HTTPException(status_code=404, detail="Sequential evaluation session not found. Complete Manual and LISTEN methods first.")

            # Update LILO method data
            seq_eval.lilo_prompt = request.prompt
            seq_eval.lilo_search_results = request.search_results
            seq_eval.lilo_initial_answers = request.initial_answers
            seq_eval.lilo_iteration1_flights = request.iteration1_flights
            seq_eval.lilo_iteration1_feedback = request.iteration1_feedback
            seq_eval.lilo_iteration2_flights = request.iteration2_flights
            seq_eval.lilo_iteration2_feedback = request.iteration2_feedback
            seq_eval.lilo_iteration3_flights = request.iteration3_flights
            seq_eval.lilo_rankings = request.rankings
            seq_eval.lilo_completed_at = datetime.utcnow()

            # Mark entire evaluation as complete
            seq_eval.completed_at = datetime.utcnow()

            db.commit()

            return {
                "status": "ok",
                "message": "LILO evaluation saved. Sequential evaluation complete!",
                "session_id": request.session_id,
                "completed": True
            }

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save LILO evaluation failed: {str(e)}")


@router.get("/sequential/{session_id}")
async def get_sequential_evaluation(session_id: str):
    """
    Get sequential evaluation session details.

    Returns all data from Manual, LISTEN, and LILO methods.
    """
    try:
        db = SessionLocal()
        try:
            seq_eval = db.query(SequentialEvaluation).filter_by(
                session_id=session_id
            ).first()

            if not seq_eval:
                raise HTTPException(status_code=404, detail="Sequential evaluation not found")

            return {
                "session_id": seq_eval.session_id,
                "user_id": seq_eval.user_id,
                "manual": {
                    "search_results": seq_eval.manual_search_results,
                    "rankings": seq_eval.manual_rankings,
                    "completed_at": seq_eval.manual_completed_at.isoformat() if seq_eval.manual_completed_at else None
                },
                "listen": {
                    "prompt": seq_eval.listen_prompt,
                    "search_results": seq_eval.listen_search_results,
                    "ranked_flights": seq_eval.listen_ranked_flights,
                    "rankings": seq_eval.listen_rankings,
                    "completed_at": seq_eval.listen_completed_at.isoformat() if seq_eval.listen_completed_at else None
                },
                "lilo": {
                    "prompt": seq_eval.lilo_prompt,
                    "search_results": seq_eval.lilo_search_results,
                    "initial_answers": seq_eval.lilo_initial_answers,
                    "iteration1_flights": seq_eval.lilo_iteration1_flights,
                    "iteration1_feedback": seq_eval.lilo_iteration1_feedback,
                    "iteration2_flights": seq_eval.lilo_iteration2_flights,
                    "iteration2_feedback": seq_eval.lilo_iteration2_feedback,
                    "iteration3_flights": seq_eval.lilo_iteration3_flights,
                    "rankings": seq_eval.lilo_rankings,
                    "completed_at": seq_eval.lilo_completed_at.isoformat() if seq_eval.lilo_completed_at else None
                },
                "created_at": seq_eval.created_at.isoformat(),
                "completed_at": seq_eval.completed_at.isoformat() if seq_eval.completed_at else None,
                "comparison_metrics": seq_eval.comparison_metrics
            }

        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get sequential evaluation failed: {str(e)}")