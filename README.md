# Flight Ranking Research Application

A Streamlit-based flight search and evaluation system that collects user preference data to compare ranking algorithms. Users search for flights using natural language, view results from three different algorithms (Cheapest, Fastest, LISTEN-U), and submit their top 5 preferences.

## Overview

This application helps research how well different flight ranking algorithms match user preferences by:
1. Parsing natural language flight queries using Gemini
2. Fetching flight data from Amadeus API
3. Ranking flights using three algorithms
4. Collecting user feedback (top 5 selections)
5. Storing data in Railway MySQL for analysis

## Tech Stack

- **Streamlit** - Web interface
- **Amadeus API** - Flight search data
- **Gemini API** - Natural language parsing and LISTEN-U preference learning
- **LISTEN** - Utility-based preference learning algorithm
- **Railway MySQL** - Persistent data storage
- **Python 3.11+** - Required for LISTEN compatibility

## Architecture

### Data Flow

1. User enters natural language query (e.g., "Fly from NYC to LA on Nov 20")
2. Gemini parses query to extract airports, dates, preferences
3. Amadeus API returns ~50 flights for the route
4. Three ranking algorithms process the same 50 flights:
   - **Cheapest**: Sorts by price
   - **Fastest**: Sorts by duration
   - **LISTEN-U**: Learns utility function over 25 iterations using Gemini
5. Top 10 from each algorithm are interleaved (up to 30 unique flights shown)
6. User selects and ranks their top 5 flights
7. Data saved to Railway MySQL for analysis

### Project Structure

```
flight_app/
├── app.py                          # Main Streamlit application
├── backend/
│   ├── amadeus_client.py           # Amadeus API client
│   ├── prompt_parser.py            # Gemini-based query parser
│   ├── listen_main_wrapper.py      # LISTEN algorithm wrapper
│   ├── listen_data_converter.py    # Convert flights to LISTEN format
│   ├── db.py                       # Railway MySQL database
│   └── utils/
│       └── parse_duration.py       # ISO 8601 duration parsing
├── LISTEN/                         # Symlink to LISTEN repository
├── view_data.py                    # View collected data from terminal
├── test_listen.py                  # Test LISTEN integration
├── .env                            # API keys and database credentials
├── .python-version                 # Python 3.11.0 (required for LISTEN)
└── requirements.txt                # Python dependencies
```

## Setup

### Prerequisites

- Python 3.11 or higher (required for LISTEN)
- Amadeus API credentials (https://developers.amadeus.com/)
- Gemini API key (https://aistudio.google.com/app/apikey)
- Railway MySQL database (or local MySQL)
- LISTEN repository cloned locally

### Installation

1. Clone and configure environment:

```bash
cd flight_app
cp .env.example .env
```

2. Edit `.env` with your credentials:

```bash
# Amadeus API
AMADEUS_API_KEY=your_key_here
AMADEUS_API_SECRET=your_secret_here
AMADEUS_BASE_URL=https://api.amadeus.com

# Gemini API (used for parsing and LISTEN)
GEMINI_API_KEY=your_gemini_key_here
GOOGLE_API_KEY=your_gemini_key_here

# Railway MySQL
DB_TYPE=mysql
MYSQL_HOST=maglev.proxy.rlwy.net
MYSQL_PORT=50981
MYSQL_DATABASE=railway
MYSQL_USER=root
MYSQL_PASSWORD=your_railway_password
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create symbolic link to LISTEN repository:

```bash
ln -s /path/to/LISTEN LISTEN
```

5. Initialize database:

```bash
python backend/db.py
```

### Running Locally

```bash
streamlit run app.py
```

Access at http://localhost:8501

### Deployment

Deployed on Streamlit Cloud with:
- Python 3.11 runtime (`.python-version`)
- Railway MySQL for persistence
- Environment variables configured in Streamlit Cloud settings

## Data Collection

### Database Schema

**searches** - Each flight search query
- search_id, session_id, user_prompt, parsed_origins, parsed_destinations, departure_date, created_at

**flights_shown** - All 30 flights displayed to user
- id, search_id, flight_data (JSON), algorithm (Cheapest/Fastest/LISTEN-U), algorithm_rank, display_position

**user_rankings** - User's top 5 selections
- id, search_id, flight_id, user_rank (1-5), submitted_at

### Viewing Data

Terminal:
```bash
python3 view_data.py          # All searches
python3 view_data.py latest   # Latest search only
```

SQL (direct Railway MySQL access):
```bash
mysql -h maglev.proxy.rlwy.net -P 50981 -u root -p railway
```

Example queries:
```sql
-- Algorithm performance
SELECT
  f.algorithm,
  COUNT(r.id) as times_selected,
  AVG(r.user_rank) as avg_rank
FROM flights_shown f
JOIN user_rankings r ON f.id = r.flight_id
GROUP BY f.algorithm;

-- Position bias analysis
SELECT display_position, COUNT(*) as selections
FROM flights_shown f
JOIN user_rankings r ON f.id = r.flight_id
GROUP BY display_position;
```

## Algorithm Details

### LISTEN-U (Utility-based Preference Learning)

Uses the official LISTEN repository (https://github.com/AdamJovine/LISTEN) with:
- **Algorithm**: `utility` (main algorithm from paper)
- **Iterations**: 25 (learns utility function)
- **LLM**: Gemini via `--api-model gemini`
- **Runtime**: ~2-3 minutes per search

LISTEN learns a utility function by querying Gemini 25 times, then uses the learned function to rank all 50 flights and return the top 10.

Test LISTEN locally:
```bash
python3 test_listen.py
```

## Current Limitations

1. **Single origin/destination** - Only searches first airport from parsed list (e.g., JFK from ["JFK", "LGA", "EWR"])
2. **Test API** - Amadeus test environment has limited airport coverage
3. **No layover details in UI** - Full segment data stored but not displayed
4. **25 iteration runtime** - LISTEN-U takes 2-3 minutes per search

## Research Questions

This system can answer:
- Does LISTEN-U rank flights better than simple heuristics (price/duration)?
- Do users exhibit position bias (prefer top-ranked results)?
- What's the correlation between algorithm rank and user rank?
- How do user preferences vary across different routes/contexts?

## API Costs

Per search:
- 1 Amadeus API call (free tier)
- 1 Gemini call for parsing (~$0.00001)
- 25 Gemini calls for LISTEN (~$0.00025)
- Total: ~$0.00026 per search

## Testing

Test LISTEN integration:
```bash
python3 test_listen.py
```

Test database connection:
```bash
python3 backend/db.py
```

## Resources

- LISTEN Repository: https://github.com/AdamJovine/LISTEN
- Amadeus API Docs: https://developers.amadeus.com/self-service/category/flights
- Gemini API Docs: https://ai.google.dev/gemini-api/docs
- Streamlit Docs: https://docs.streamlit.io

## License

This project is provided for educational and research purposes.
