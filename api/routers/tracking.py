"""
Tracking router - User interaction events (clicks, views, hovers, etc.)
"""
from fastapi import APIRouter, HTTPException
from ..models.requests import TrackingEvent
from ..models.responses import TrackingResponse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db import SessionLocal, InteractionEvent

router = APIRouter()


@router.post("/event", response_model=TrackingResponse)
async def track_event(event: TrackingEvent):
    """
    Track user interaction event.

    Event types:
    - 'flight_click': User clicked on a flight
    - 'flight_view': Flight entered viewport
    - 'flight_hover': User hovered over flight
    - 'filter_applied': User applied a filter
    - 'sort_applied': User sorted results
    """
    try:
        db = SessionLocal()
        try:
            # Save event to database
            interaction = InteractionEvent(
                session_id=event.session_id,
                search_id=event.search_id,
                event_type=event.event_type,
                event_data=event.event_data
            )
            db.add(interaction)
            db.commit()
            db.refresh(interaction)

            return TrackingResponse(
                status="ok",
                event_id=interaction.id
            )

        finally:
            db.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Track event failed: {str(e)}")


@router.get("/events/{session_id}")
async def get_session_events(session_id: str, event_type: str = None):
    """
    Get all events for a session, optionally filtered by type.
    """
    try:
        db = SessionLocal()
        try:
            query = db.query(InteractionEvent).filter_by(session_id=session_id)

            if event_type:
                query = query.filter_by(event_type=event_type)

            events = query.order_by(InteractionEvent.created_at).all()

            return {
                "session_id": session_id,
                "total_events": len(events),
                "events": [
                    {
                        "id": e.id,
                        "event_type": e.event_type,
                        "event_data": e.event_data,
                        "created_at": e.created_at.isoformat()
                    }
                    for e in events
                ]
            }

        finally:
            db.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get events failed: {str(e)}")


@router.get("/analytics/{search_id}")
async def get_search_analytics(search_id: int):
    """
    Get analytics for a search:
    - Which flights were viewed
    - Which were clicked
    - How long each was viewed
    """
    try:
        db = SessionLocal()
        try:
            events = db.query(InteractionEvent).filter_by(search_id=search_id).all()

            # Aggregate by flight_id
            flight_stats = {}

            for event in events:
                flight_id = event.event_data.get('flight_id')
                if not flight_id:
                    continue

                if flight_id not in flight_stats:
                    flight_stats[flight_id] = {
                        'flight_id': flight_id,
                        'views': 0,
                        'clicks': 0,
                        'hovers': 0,
                        'total_viewport_time_ms': 0
                    }

                if event.event_type == 'flight_view':
                    flight_stats[flight_id]['views'] += 1
                    viewport_time = event.event_data.get('viewport_time_ms', 0)
                    flight_stats[flight_id]['total_viewport_time_ms'] += viewport_time
                elif event.event_type == 'flight_click':
                    flight_stats[flight_id]['clicks'] += 1
                elif event.event_type == 'flight_hover':
                    flight_stats[flight_id]['hovers'] += 1

            return {
                "search_id": search_id,
                "total_events": len(events),
                "flight_stats": list(flight_stats.values())
            }

        finally:
            db.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get analytics failed: {str(e)}")