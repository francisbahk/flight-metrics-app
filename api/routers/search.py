"""
Search router - Flight search with Gemini + Amadeus
"""
from fastapi import APIRouter, HTTPException
from ..models.requests import SearchRequest
from ..models.responses import SearchResponse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.prompt_parser import parse_flight_prompt_with_llm
from backend.amadeus_client import AmadeusClient
from backend.utils.parse_duration import parse_duration_to_minutes
from backend.db import SessionLocal, Search, FlightShown

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_flights(request: SearchRequest):
    """
    Search for flights using natural language query.

    Workflow:
    1. Parse prompt with Gemini
    2. Search flights via Amadeus
    3. Save to database
    4. Return flights + parsed params
    """
    try:
        # Step 1: Parse prompt with Gemini
        parsed = parse_flight_prompt_with_llm(request.query)

        if not parsed.get('origins') or not parsed.get('destinations'):
            raise HTTPException(
                status_code=400,
                detail="Could not extract origin and destination from query"
            )

        # Step 2: Search flights via Amadeus
        amadeus = AmadeusClient()

        # Use first origin and destination
        origin = parsed['origins'][0]
        destination = parsed['destinations'][0]
        departure_date = parsed.get('departure_date')

        if not departure_date:
            raise HTTPException(
                status_code=400,
                detail="Could not extract departure date from query"
            )

        # Search flights
        results = amadeus.search_flights(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            adults=1,
            max_results=50
        )

        # Parse results
        if isinstance(results, list):
            flight_offers = results
        elif isinstance(results, dict) and 'data' in results:
            flight_offers = results['data']
        else:
            flight_offers = []

        all_flights = []
        for offer in flight_offers:
            itinerary = offer['itineraries'][0]
            segments = itinerary['segments']

            flight_info = {
                'id': offer['id'],
                'price': float(offer['price']['total']),
                'currency': offer['price']['currency'],
                'duration_min': parse_duration_to_minutes(itinerary['duration']),
                'stops': len(segments) - 1,
                'departure_time': segments[0]['departure']['at'],
                'arrival_time': segments[-1]['arrival']['at'],
                'airline': segments[0]['carrierCode'],
                'flight_number': segments[0]['number'],
                'origin': segments[0]['departure']['iataCode'],
                'destination': segments[-1]['arrival']['iataCode']
            }
            all_flights.append(flight_info)

        if not all_flights:
            raise HTTPException(
                status_code=404,
                detail="No flights found for this route and date"
            )

        # Step 3: Save to database
        db = SessionLocal()
        try:
            search = Search(
                session_id=request.session_id or "api_session",
                user_prompt=request.query,
                parsed_origins=parsed.get('origins'),
                parsed_destinations=parsed.get('destinations'),
                parsed_preferences=parsed.get('preferences'),
                departure_date=departure_date
            )
            db.add(search)
            db.commit()
            db.refresh(search)
            search_id = search.search_id
        finally:
            db.close()

        # Step 4: Return results
        return SearchResponse(
            search_id=search_id,
            flights=all_flights,
            parsed_params=parsed,
            total_flights=len(all_flights)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")