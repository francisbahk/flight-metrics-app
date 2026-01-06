"""
Pydantic models for API responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class SearchResponse(BaseModel):
    """Response from flight search"""
    search_id: int
    flights: List[Dict]
    parsed_params: Dict
    total_flights: int


class RankingResponse(BaseModel):
    """Response from ranking algorithm"""
    algorithm: str
    ranked_flights: List[Dict]
    execution_time_ms: Optional[float] = None


class LILOInitResponse(BaseModel):
    """LILO initialization response"""
    session_id: str
    round_number: int = 0
    flights_shown: List[Dict] = []
    questions: Optional[List[str]] = None
    message: str = "Answer these questions to help us understand your preferences"


class LILORoundResponse(BaseModel):
    """LILO round response"""
    session_id: str
    round_number: int
    flights_shown: List[Dict]
    questions: Optional[List[str]] = None
    feedback_summary: Optional[str] = None
    is_final_round: bool = False


class LILOFinalResponse(BaseModel):
    """LILO final ranking response"""
    session_id: str
    final_rankings: List[Dict]
    utility_scores: Optional[List[float]] = None
    feedback_summary: str


class EvaluationSessionResponse(BaseModel):
    """Evaluation session info"""
    eval_session_id: str
    search_id: int
    person_a_prompt: str
    person_a_rankings: Optional[List[Dict]] = None
    person_b_rankings: Optional[List[Dict]] = None
    listen_u_rankings: Optional[List[Dict]] = None
    lilo_rankings: Optional[List[Dict]] = None
    comparison_results: Optional[Dict] = None


class TrackingResponse(BaseModel):
    """Tracking event confirmation"""
    status: str = "ok"
    event_id: Optional[int] = None