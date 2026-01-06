"""
Pydantic models for API requests
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class SearchRequest(BaseModel):
    """Request for flight search"""
    query: str = Field(..., description="Natural language flight query")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")


class RankingRequest(BaseModel):
    """Request for ranking algorithms"""
    flights: List[Dict] = Field(..., description="List of flight dictionaries")
    user_prompt: str = Field(..., description="User's original prompt")
    user_preferences: Optional[Dict] = Field(None, description="Parsed preferences")


class LILOInitRequest(BaseModel):
    """Initialize LILO session"""
    session_id: str
    flights: List[Dict]
    user_prompt: str
    user_preferences: Optional[Dict] = None


class LILORoundRequest(BaseModel):
    """Submit LILO round feedback"""
    session_id: str
    round_number: int
    user_rankings: List[int] = Field(default=[], description="Indices of flights user ranked (in order)")
    user_feedback: str = Field(default="", description="Natural language feedback")
    question_answers: Optional[Dict[str, str]] = Field(default=None, description="Answers to questions (question -> answer mapping)")


class LILOFinalRequest(BaseModel):
    """Get final LILO ranking"""
    session_id: str


class EvaluationStartRequest(BaseModel):
    """Start evaluation session (Person A enters prompt)"""
    session_id: str
    user_id: str
    prompt: str


class PersonARankingsRequest(BaseModel):
    """Person A submits ground truth rankings"""
    session_id: str
    rankings: List[Dict] = Field(..., description="Top-5 flights in ranked order")


class PersonBRankingsRequest(BaseModel):
    """Person B submits guessed rankings"""
    eval_session_id: str
    user_id: str
    rankings: List[Dict] = Field(..., description="Top-5 guesses for Person A")


class TrackingEvent(BaseModel):
    """User interaction event"""
    session_id: str
    search_id: Optional[int] = None
    event_type: str = Field(..., description="'flight_click', 'flight_view', etc.")
    event_data: Dict = Field(..., description="Flexible event data")


# Sequential Evaluation Models (Manual → LISTEN → LILO)

class ManualEvaluationRequest(BaseModel):
    """Manual method evaluation submission"""
    session_id: str
    user_id: Optional[str] = None
    search_results: Dict = Field(..., description="Complete search results with all flights")
    rankings: List[Dict] = Field(..., description="Top-5 flight rankings")


class LISTENEvaluationRequest(BaseModel):
    """LISTEN method evaluation submission"""
    session_id: str
    prompt: str = Field(..., description="Natural language prompt")
    search_results: Dict = Field(..., description="Complete search results with all flights")
    ranked_flights: List[Dict] = Field(..., description="LISTEN-U ranked flights")
    rankings: List[Dict] = Field(..., description="Top-5 flight rankings")


class LILOEvaluationRequest(BaseModel):
    """LILO method evaluation submission"""
    session_id: str
    prompt: str = Field(..., description="Natural language prompt")
    search_results: Dict = Field(..., description="Complete search results with all flights")
    initial_answers: Dict = Field(..., description="Answers to initial preference questions")
    iteration1_flights: List[Dict] = Field(..., description="15 random flights shown in iteration 1")
    iteration1_feedback: str = Field(..., description="User feedback from iteration 1")
    iteration2_flights: List[Dict] = Field(..., description="15 utility-ranked flights from iteration 2")
    iteration2_feedback: str = Field(..., description="User feedback from iteration 2")
    iteration3_flights: List[Dict] = Field(..., description="15 final flights shown in iteration 3")
    rankings: List[Dict] = Field(..., description="Top-5 final flight rankings")