"""
Ranking router - Cheapest, Fastest, LISTEN-U algorithms
"""
from fastapi import APIRouter, HTTPException
from ..models.requests import RankingRequest
from ..models.responses import RankingResponse
import sys
from pathlib import Path
import time

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.listen_main_wrapper import rank_flights_with_listen_main

router = APIRouter()


@router.post("/rank/cheapest", response_model=RankingResponse)
async def rank_cheapest(request: RankingRequest):
    """
    Rank flights by price (cheapest first).
    """
    try:
        start_time = time.time()

        # Sort by price (all flights)
        ranked = sorted(request.flights, key=lambda x: x['price'])

        execution_time = (time.time() - start_time) * 1000  # ms

        return RankingResponse(
            algorithm="Cheapest",
            ranked_flights=ranked,
            execution_time_ms=execution_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")


@router.post("/rank/fastest", response_model=RankingResponse)
async def rank_fastest(request: RankingRequest):
    """
    Rank flights by duration (fastest first).
    """
    try:
        start_time = time.time()

        # Sort by duration (all flights)
        ranked = sorted(request.flights, key=lambda x: x['duration_min'])

        execution_time = (time.time() - start_time) * 1000  # ms

        return RankingResponse(
            algorithm="Fastest",
            ranked_flights=ranked,
            execution_time_ms=execution_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")


@router.post("/rank/listen-u", response_model=RankingResponse)
async def rank_listen_u(request: RankingRequest):
    """
    Rank flights using LISTEN-U algorithm (5 iterations).

    This takes 1-2 minutes.
    """
    try:
        start_time = time.time()

        # Run LISTEN-U
        ranked = rank_flights_with_listen_main(
            flights=request.flights,
            user_prompt=request.user_prompt,
            user_preferences=request.user_preferences or {}
            # n_iterations defaults to 5 in wrapper
        )

        execution_time = (time.time() - start_time) * 1000  # ms

        return RankingResponse(
            algorithm="LISTEN-U",
            ranked_flights=ranked,
            execution_time_ms=execution_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LISTEN-U failed: {str(e)}"
        )


@router.post("/rank/all")
async def rank_all_algorithms(request: RankingRequest):
    """
    Run all 3 algorithms and return results.

    Returns:
    {
        "cheapest": {...},
        "fastest": {...},
        "listen_u": {...}
    }
    """
    try:
        # Run all algorithms
        cheapest_resp = await rank_cheapest(request)
        fastest_resp = await rank_fastest(request)
        listen_u_resp = await rank_listen_u(request)

        return {
            "cheapest": cheapest_resp.dict(),
            "fastest": fastest_resp.dict(),
            "listen_u": listen_u_resp.dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")