# backend/

Core backend logic for the flight app. Handles database, flight APIs, and prompt parsing.

## Files

| File | Purpose |
|------|---------|
| `db.py` | SQLAlchemy ORM models, database session management, and all DB read/write functions |
| `flight_search.py` | Unified flight search client — wraps Amadeus and SerpAPI behind a single interface |
| `amadeus_client.py` | Amadeus Flight Offers API client (search, airline lookups) |
| `serpapi_client.py` | SerpAPI Google Flights client (alternative search provider) |
| `prompt_parser.py` | Uses Gemini LLM to parse natural language flight prompts into structured search parameters |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `utils/` | Small utility functions (e.g., duration parsing) |

## Data Flow

```
User prompt → prompt_parser.py → structured params → flight_search.py → amadeus/serpapi → flight results
                                                                                              ↓
                                                                              db.py (save search + results)
```
