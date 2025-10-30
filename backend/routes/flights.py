"""
Flight search and management API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime
import json

from database import get_db
from models.flight import Flight
from amadeus_client import AmadeusClient
from utils.parse_duration import (
    parse_duration_to_minutes,
    calculate_flight_distances,
    calculate_time_of_day_seconds,
)

router = APIRouter(prefix="/api/flights", tags=["flights"])

# Initialize Amadeus client
amadeus_client = AmadeusClient()


@router.get("/search")
async def search_flights(
    origin: str = Query(..., description="Origin airport IATA code (e.g., JFK)"),
    destination: str = Query(..., description="Destination airport IATA code (e.g., LAX)"),
    departure_date: str = Query(..., description="Departure date (YYYY-MM-DD)"),
    adults: int = Query(1, ge=1, le=9, description="Number of adult passengers"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    db: Session = Depends(get_db),
):
    """
    Search for flights using Amadeus API and store results in database.

    This endpoint:
    1. Calls Amadeus Flight Offers Search API
    2. Parses and enriches flight data with computed metrics
    3. Stores flights in MySQL database
    4. Returns the stored flight records

    Computed metrics:
    - duration_min: Duration in minutes (parsed from ISO 8601)
    - dis_from_origin: Distance from reference point to origin (km)
    - dis_from_dest: Distance from reference point to destination (km)
    - departure_seconds: Seconds from midnight for departure time
    - arrival_seconds: Seconds from midnight for arrival time
    """
    try:
        # Validate date format
        try:
            datetime.strptime(departure_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        # Search flights via Amadeus API
        flight_offers = amadeus_client.search_flights(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            adults=adults,
            max_results=max_results,
        )

        if not flight_offers:
            return {
                "message": "No flights found",
                "count": 0,
                "flights": [],
            }

        # Process and store flights
        stored_flights = []

        for offer in flight_offers:
            # Parse flight offer
            parsed = amadeus_client.parse_flight_offer(offer)
            if not parsed:
                continue

            # Calculate additional metrics
            duration_min = parse_duration_to_minutes(parsed["duration"])
            dis_from_origin, dis_from_dest = calculate_flight_distances(
                parsed["origin"], parsed["destination"]
            )

            # Parse times
            departure_time = datetime.fromisoformat(
                parsed["departure_time"].replace("Z", "+00:00")
            )
            arrival_time = datetime.fromisoformat(
                parsed["arrival_time"].replace("Z", "+00:00")
            )

            # Calculate time of day in seconds
            departure_seconds = calculate_time_of_day_seconds(parsed["departure_time"])
            arrival_seconds = calculate_time_of_day_seconds(parsed["arrival_time"])

            # Create flight name (airline + route)
            carrier = parsed.get("carrier_code", "")
            name = f"{carrier} {parsed['origin']}-{parsed['destination']}"

            # Create flight record
            flight = Flight(
                name=name,
                origin=parsed["origin"],
                destination=parsed["destination"],
                departure_time=departure_time,
                arrival_time=arrival_time,
                duration=parsed["duration"],
                stops=parsed["stops"],
                price=parsed["price"],
                dis_from_origin=dis_from_origin,
                dis_from_dest=dis_from_dest,
                departure_seconds=departure_seconds,
                arrival_seconds=arrival_seconds,
                duration_min=duration_min,
                raw_data=parsed["raw_data"],
            )

            db.add(flight)
            stored_flights.append(flight)

        # Commit all flights to database
        db.commit()

        # Refresh to get IDs
        for flight in stored_flights:
            db.refresh(flight)

        return {
            "message": f"Successfully retrieved and stored {len(stored_flights)} flights",
            "count": len(stored_flights),
            "flights": [flight.to_dict() for flight in stored_flights],
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error searching flights: {str(e)}")


@router.get("/all")
async def get_all_flights(
    origin: Optional[str] = Query(None, description="Filter by origin"),
    destination: Optional[str] = Query(None, description="Filter by destination"),
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price"),
    max_stops: Optional[int] = Query(None, description="Maximum number of stops"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
    db: Session = Depends(get_db),
):
    """
    Get all stored flights with optional filters.

    Supports filtering by:
    - Origin airport
    - Destination airport
    - Price range
    - Maximum stops
    """
    try:
        query = db.query(Flight)

        # Apply filters
        if origin:
            query = query.filter(Flight.origin == origin.upper())
        if destination:
            query = query.filter(Flight.destination == destination.upper())
        if min_price is not None:
            query = query.filter(Flight.price >= min_price)
        if max_price is not None:
            query = query.filter(Flight.price <= max_price)
        if max_stops is not None:
            query = query.filter(Flight.stops <= max_stops)

        # Order by date retrieved (newest first) and apply limit
        flights = query.order_by(Flight.date_retrieved.desc()).limit(limit).all()

        return {
            "count": len(flights),
            "flights": [flight.to_dict() for flight in flights],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving flights: {str(e)}")


@router.get("/metrics")
async def get_flight_metrics(
    origin: Optional[str] = Query(None, description="Filter by origin"),
    destination: Optional[str] = Query(None, description="Filter by destination"),
    db: Session = Depends(get_db),
):
    """
    Compute aggregated metrics for stored flights.

    Returns statistics including:
    - Total number of flights
    - Average price
    - Minimum/maximum price
    - Average duration
    - Average number of stops
    - Price by number of stops
    """
    try:
        query = db.query(Flight)

        # Apply filters
        if origin:
            query = query.filter(Flight.origin == origin.upper())
        if destination:
            query = query.filter(Flight.destination == destination.upper())

        flights = query.all()

        if not flights:
            return {
                "message": "No flights found",
                "count": 0,
                "metrics": None,
            }

        # Calculate basic metrics
        prices = [f.price for f in flights]
        durations = [f.duration_min for f in flights if f.duration_min]
        stops = [f.stops for f in flights]

        metrics = {
            "count": len(flights),
            "price": {
                "average": sum(prices) / len(prices),
                "min": min(prices),
                "max": max(prices),
                "median": sorted(prices)[len(prices) // 2],
            },
            "duration": {
                "average_minutes": sum(durations) / len(durations) if durations else 0,
                "min_minutes": min(durations) if durations else 0,
                "max_minutes": max(durations) if durations else 0,
            },
            "stops": {
                "average": sum(stops) / len(stops),
                "min": min(stops),
                "max": max(stops),
            },
        }

        # Calculate price by stops
        price_by_stops = {}
        for stop_count in set(stops):
            stop_flights = [f for f in flights if f.stops == stop_count]
            stop_prices = [f.price for f in stop_flights]
            price_by_stops[stop_count] = {
                "count": len(stop_flights),
                "average_price": sum(stop_prices) / len(stop_prices),
            }

        metrics["price_by_stops"] = price_by_stops

        return {
            "message": "Metrics calculated successfully",
            "count": len(flights),
            "metrics": metrics,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing metrics: {str(e)}")


@router.get("/{flight_id}")
async def get_flight(
    flight_id: int,
    db: Session = Depends(get_db),
):
    """
    Get a specific flight by ID.
    """
    try:
        flight = db.query(Flight).filter(Flight.id == flight_id).first()

        if not flight:
            raise HTTPException(status_code=404, detail="Flight not found")

        return flight.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving flight: {str(e)}")
