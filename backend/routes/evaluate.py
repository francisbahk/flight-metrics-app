"""
Evaluation API routes for LISTEN and Team Draft experiments.
"""
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime

from database import get_db
from models.flight import ListenRanking, TeamDraftResult, Rating, Flight
from utils.parse_duration import interleave_rankings
from utils.listen_algorithms import ListenU, ListenT

router = APIRouter(prefix="/api/evaluate", tags=["evaluate"])


# Pydantic models for request validation
class ListenRankingRequest(BaseModel):
    user_id: str
    prompt: str
    flight_ids: List[int]
    user_ranking: List[int]  # Ordered list of flight IDs
    notes: Optional[str] = None


class TeamDraftStartRequest(BaseModel):
    user_id: str
    prompt: str
    algorithm_a: str
    algorithm_b: str
    algorithm_a_ranking: List[int]  # List of flight IDs
    algorithm_b_ranking: List[int]  # List of flight IDs
    notes: Optional[str] = None


class TeamDraftSubmitRequest(BaseModel):
    session_id: int
    user_preferences: List[bool]  # True for "yes", False for "no"


class RatingRequest(BaseModel):
    user_id: str
    flight_id: int
    rating: int  # 1-5 star rating
    prompt_giver: bool = False


class ListenURequest(BaseModel):
    user_id: str
    flight_ids: List[int]
    preference_utterance: str
    max_iterations: int = 3
    notes: Optional[str] = None


class ListenTRequest(BaseModel):
    user_id: str
    flight_ids: List[int]
    preference_utterance: str
    num_rounds: int = 3
    batch_size: int = 4
    notes: Optional[str] = None


@router.post("/listen/ranking")
async def submit_listen_ranking(
    ranking: ListenRankingRequest,
    db: Session = Depends(get_db),
):
    """
    Submit a LISTEN ranking evaluation.

    Users provide their preferred ranking of a set of flights.
    This is used to evaluate ranking algorithms.

    Request body:
    - user_id: Unique identifier for the user
    - prompt: Description of ranking criteria (e.g., "Rank by best value")
    - flight_ids: Original unordered list of flight IDs
    - user_ranking: User's ordered list of flight IDs (best to worst)
    - notes: Optional notes about the ranking
    """
    try:
        # Validate that all flights exist
        for flight_id in ranking.flight_ids:
            flight = db.query(Flight).filter(Flight.id == flight_id).first()
            if not flight:
                raise HTTPException(status_code=404, detail=f"Flight {flight_id} not found")

        # Validate that user_ranking contains same flights as flight_ids
        if set(ranking.user_ranking) != set(ranking.flight_ids):
            raise HTTPException(
                status_code=400,
                detail="user_ranking must contain the same flight IDs as flight_ids",
            )

        # Create ranking record
        listen_ranking = ListenRanking(
            user_id=ranking.user_id,
            prompt=ranking.prompt,
            flight_ids=ranking.flight_ids,
            user_ranking=ranking.user_ranking,
            mode="listen",
            notes=ranking.notes,
        )

        db.add(listen_ranking)
        db.commit()
        db.refresh(listen_ranking)

        return {
            "message": "LISTEN ranking submitted successfully",
            "ranking_id": listen_ranking.id,
            "data": listen_ranking.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error submitting ranking: {str(e)}")


@router.get("/listen/rankings")
async def get_listen_rankings(
    user_id: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    Get LISTEN rankings with optional filtering by user.
    """
    try:
        query = db.query(ListenRanking)

        if user_id:
            query = query.filter(ListenRanking.user_id == user_id)

        rankings = query.order_by(ListenRanking.timestamp.desc()).limit(limit).all()

        return {
            "count": len(rankings),
            "rankings": [r.to_dict() for r in rankings],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving rankings: {str(e)}")


@router.post("/teamdraft/start")
async def start_team_draft(
    request: TeamDraftStartRequest,
    db: Session = Depends(get_db),
):
    """
    Start a Team Draft evaluation session.

    Team Draft interleaves two algorithm rankings and presents flights
    one at a time to the user, asking for binary preferences (yes/no).

    Request body:
    - user_id: Unique identifier for the user
    - prompt: Description of preference criteria
    - algorithm_a: Name of first algorithm
    - algorithm_b: Name of second algorithm
    - algorithm_a_ranking: Ordered list of flight IDs from algorithm A
    - algorithm_b_ranking: Ordered list of flight IDs from algorithm B
    - notes: Optional notes

    Returns:
    - session_id: ID for this Team Draft session
    - interleaved_list: The interleaved flight list to present to user
    """
    try:
        # Validate that all flights exist
        all_flight_ids = set(request.algorithm_a_ranking + request.algorithm_b_ranking)
        for flight_id in all_flight_ids:
            flight = db.query(Flight).filter(Flight.id == flight_id).first()
            if not flight:
                raise HTTPException(status_code=404, detail=f"Flight {flight_id} not found")

        # Create interleaved list
        interleaved = interleave_rankings(
            request.algorithm_a_ranking, request.algorithm_b_ranking
        )

        # Create Team Draft session
        team_draft = TeamDraftResult(
            user_id=request.user_id,
            prompt=request.prompt,
            algorithm_a=request.algorithm_a,
            algorithm_b=request.algorithm_b,
            algorithm_a_ranking=request.algorithm_a_ranking,
            algorithm_b_ranking=request.algorithm_b_ranking,
            interleaved_list=interleaved,
            user_preferences=[],  # Will be filled when user submits
            a_score=0,
            b_score=0,
            notes=request.notes,
        )

        db.add(team_draft)
        db.commit()
        db.refresh(team_draft)

        # Fetch full flight data for interleaved list
        flights_data = []
        for item in interleaved:
            flight = db.query(Flight).filter(Flight.id == item["flight_id"]).first()
            if flight:
                flights_data.append({
                    "flight": flight.to_dict(),
                    "source": item["source"],
                })

        return {
            "message": "Team Draft session started",
            "session_id": team_draft.id,
            "interleaved_list": interleaved,
            "flights_data": flights_data,
            "total_flights": len(interleaved),
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error starting Team Draft: {str(e)}")


@router.post("/teamdraft/submit")
async def submit_team_draft(
    request: TeamDraftSubmitRequest,
    db: Session = Depends(get_db),
):
    """
    Submit Team Draft preferences and calculate scores.

    Request body:
    - session_id: ID of the Team Draft session
    - user_preferences: List of boolean values (True = yes/like, False = no/dislike)

    Returns:
    - Calculated scores for algorithm A and B
    - Final result data
    """
    try:
        # Get Team Draft session
        team_draft = (
            db.query(TeamDraftResult).filter(TeamDraftResult.id == request.session_id).first()
        )

        if not team_draft:
            raise HTTPException(status_code=404, detail="Team Draft session not found")

        # Validate preferences length matches interleaved list
        if len(request.user_preferences) != len(team_draft.interleaved_list):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(team_draft.interleaved_list)} preferences, got {len(request.user_preferences)}",
            )

        # Calculate scores
        a_score = 0
        b_score = 0

        for i, preference in enumerate(request.user_preferences):
            if preference:  # User liked this flight
                source = team_draft.interleaved_list[i]["source"]
                if source == "a":
                    a_score += 1
                elif source == "b":
                    b_score += 1

        # Update Team Draft record
        team_draft.user_preferences = request.user_preferences
        team_draft.a_score = a_score
        team_draft.b_score = b_score

        db.commit()
        db.refresh(team_draft)

        return {
            "message": "Team Draft submitted successfully",
            "session_id": team_draft.id,
            "results": {
                "algorithm_a": team_draft.algorithm_a,
                "algorithm_b": team_draft.algorithm_b,
                "a_score": a_score,
                "b_score": b_score,
                "total_yes": sum(request.user_preferences),
                "total_no": len(request.user_preferences) - sum(request.user_preferences),
                "winner": (
                    team_draft.algorithm_a
                    if a_score > b_score
                    else team_draft.algorithm_b if b_score > a_score else "tie"
                ),
            },
            "data": team_draft.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Error submitting Team Draft: {str(e)}"
        )


@router.get("/teamdraft/results/{session_id}")
async def get_team_draft_results(
    session_id: int,
    db: Session = Depends(get_db),
):
    """
    Get results for a specific Team Draft session.
    """
    try:
        team_draft = (
            db.query(TeamDraftResult).filter(TeamDraftResult.id == session_id).first()
        )

        if not team_draft:
            raise HTTPException(status_code=404, detail="Team Draft session not found")

        return {
            "session_id": team_draft.id,
            "data": team_draft.to_dict(),
            "results": {
                "algorithm_a": team_draft.algorithm_a,
                "algorithm_b": team_draft.algorithm_b,
                "a_score": team_draft.a_score,
                "b_score": team_draft.b_score,
                "winner": (
                    team_draft.algorithm_a
                    if team_draft.a_score > team_draft.b_score
                    else team_draft.algorithm_b
                    if team_draft.b_score > team_draft.a_score
                    else "tie"
                ),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")


@router.post("/rating")
async def submit_rating(
    rating: RatingRequest,
    db: Session = Depends(get_db),
):
    """
    Submit a rating for a specific flight.

    Request body:
    - user_id: Unique identifier for the user
    - flight_id: ID of the flight being rated
    - rating: Rating value (typically 1-5)
    - prompt_giver: Whether this user provided the ranking prompt
    """
    try:
        # Validate flight exists
        flight = db.query(Flight).filter(Flight.id == rating.flight_id).first()
        if not flight:
            raise HTTPException(status_code=404, detail="Flight not found")

        # Validate rating range
        if rating.rating < 1 or rating.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

        # Create rating record
        flight_rating = Rating(
            user_id=rating.user_id,
            flight_id=rating.flight_id,
            rating=rating.rating,
            prompt_giver=rating.prompt_giver,
        )

        db.add(flight_rating)
        db.commit()
        db.refresh(flight_rating)

        return {
            "message": "Rating submitted successfully",
            "rating_id": flight_rating.id,
            "data": flight_rating.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error submitting rating: {str(e)}")


@router.get("/ratings")
async def get_ratings(
    user_id: Optional[str] = None,
    flight_id: Optional[int] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    Get ratings with optional filtering.
    """
    try:
        query = db.query(Rating)

        if user_id:
            query = query.filter(Rating.user_id == user_id)
        if flight_id:
            query = query.filter(Rating.flight_id == flight_id)

        ratings = query.order_by(Rating.timestamp.desc()).limit(limit).all()

        return {
            "count": len(ratings),
            "ratings": [r.to_dict() for r in ratings],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving ratings: {str(e)}")


@router.post("/listen-u/run")
async def run_listen_u(
    request: ListenURequest,
    db: Session = Depends(get_db),
):
    """
    Run LISTEN-U algorithm (Utility Refinement).
    
    LISTEN-U iteratively refines a linear utility function over numerical attributes
    to find the best flight matching user preferences.
    
    Request body:
    - user_id: Unique identifier for the user
    - flight_ids: List of flight IDs to rank
    - preference_utterance: User's natural language preferences
    - max_iterations: Number of weight refinement iterations (default: 3)
    - notes: Optional notes
    
    Returns:
    - Ranked flights with utility scores
    - Final weight vector
    - Iteration history
    """
    try:
        # Fetch flights from database
        flights = db.query(Flight).filter(Flight.id.in_(request.flight_ids)).all()
        
        if not flights:
            raise HTTPException(status_code=404, detail="No flights found with provided IDs")
        
        if len(flights) != len(request.flight_ids):
            raise HTTPException(
                status_code=400,
                detail=f"Found {len(flights)} flights but {len(request.flight_ids)} IDs provided"
            )
        
        # Convert to dictionaries
        flights_data = [f.to_dict() for f in flights]
        
        # Run LISTEN-U algorithm
        listen_u = ListenU()
        results = listen_u.rank_flights(
            flights_data,
            request.preference_utterance,
            request.max_iterations
        )
        
        # Store results in database
        ranking_data = {
            "user_id": request.user_id,
            "prompt": request.preference_utterance,
            "flight_ids": request.flight_ids,
            "user_ranking": [f["id"] for f in results["ranked_flights"]],
            "mode": "listen-u",
            "notes": f"LISTEN-U weights: {results['final_weights']}. {request.notes or ''}",
        }
        
        listen_ranking = ListenRanking(**ranking_data)
        db.add(listen_ranking)
        db.commit()
        db.refresh(listen_ranking)
        
        return {
            "message": "LISTEN-U completed successfully",
            "ranking_id": listen_ranking.id,
            "algorithm": "LISTEN-U",
            "results": results,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error running LISTEN-U: {str(e)}")


@router.post("/listen-t/run")
async def run_listen_t(
    request: ListenTRequest,
    db: Session = Depends(get_db),
):
    """
    Run LISTEN-T algorithm (Tournament Selection).
    
    LISTEN-T conducts a tournament by sampling random batches of flights,
    selecting a champion from each batch, then running a final playoff.
    
    Request body:
    - user_id: Unique identifier for the user
    - flight_ids: List of flight IDs to evaluate
    - preference_utterance: User's natural language preferences
    - num_rounds: Number of preliminary tournament rounds (default: 3)
    - batch_size: Number of flights per batch (default: 4)
    - notes: Optional notes
    
    Returns:
    - Tournament winner
    - All champions from each round
    - Tournament bracket/history
    """
    try:
        # Fetch flights from database
        flights = db.query(Flight).filter(Flight.id.in_(request.flight_ids)).all()
        
        if not flights:
            raise HTTPException(status_code=404, detail="No flights found with provided IDs")
        
        if len(flights) != len(request.flight_ids):
            raise HTTPException(
                status_code=400,
                detail=f"Found {len(flights)} flights but {len(request.flight_ids)} IDs provided"
            )
        
        # Convert to dictionaries
        flights_data = [f.to_dict() for f in flights]
        
        # Run LISTEN-T algorithm
        listen_t = ListenT()
        results = listen_t.rank_flights(
            flights_data,
            request.preference_utterance,
            request.num_rounds,
            request.batch_size
        )
        
        # Store results in database
        ranking_data = {
            "user_id": request.user_id,
            "prompt": request.preference_utterance,
            "flight_ids": request.flight_ids,
            "user_ranking": [f["id"] for f in results["ranked_flights"]],
            "mode": "listen-t",
            "notes": f"LISTEN-T tournament with {request.num_rounds} rounds. {request.notes or ''}",
        }
        
        listen_ranking = ListenRanking(**ranking_data)
        db.add(listen_ranking)
        db.commit()
        db.refresh(listen_ranking)
        
        return {
            "message": "LISTEN-T completed successfully",
            "ranking_id": listen_ranking.id,
            "algorithm": "LISTEN-T",
            "results": results,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error running LISTEN-T: {str(e)}")
