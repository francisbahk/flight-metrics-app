# Flight Ranking App - Complete System Context

## Project Overview

This is a **flight ranking evaluation research platform** that:
1. Takes natural language flight queries from users
2. Searches real flights via Amadeus API
3. Ranks flights using 3 algorithms (Cheapest, Fastest, LISTEN-U)
4. Collects user feedback on rankings for algorithm evaluation
5. Stores all data in MySQL for offline analysis

**Goal**: Evaluate how well LISTEN-U (LLM-based ranking) performs compared to simple heuristics.

---

## Complete Data Flow (Prompt → SQL Storage)

```
┌─────────────────────────────────────────────────────────────────────┐
│ USER TYPES PROMPT                                                   │
│ "I want to fly from Houston to NYC on Nov 20. I like cheap flights"│
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: GEMINI PROMPT PARSING (backend/prompt_parser.py)           │
│                                                                     │
│ Input:  Natural language string                                    │
│ Model:  Gemini 2.0 Flash                                           │
│ Output: Structured JSON                                            │
│         {                                                           │
│           "origins": ["IAH", "HOU"],                               │
│           "destinations": ["JFK", "LGA", "EWR"],                   │
│           "departure_date": "2025-11-20",                          │
│           "preferences": {"prefer_cheap": true}                    │
│         }                                                           │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: AMADEUS FLIGHT SEARCH (backend/amadeus_client.py)          │
│                                                                     │
│ For each origin/destination pair:                                  │
│   1. OAuth2 authentication (get access token)                      │
│   2. GET /v2/shopping/flight-offers?origin=IAH&destination=JFK... │
│   3. Parse response → ~50 flight offers                            │
│                                                                     │
│ Each flight has:                                                   │
│   - id, price, currency, duration_min, stops                       │
│   - departure_time, arrival_time, airline, flight_number          │
│   - origin, destination                                            │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: THREE RANKING ALGORITHMS (app_new.py:222-258)              │
│                                                                     │
│ Algorithm 1: CHEAPEST                                              │
│   sorted(flights, key=lambda x: x['price'])[:10]                   │
│                                                                     │
│ Algorithm 2: FASTEST                                               │
│   sorted(flights, key=lambda x: x['duration_min'])[:10]            │
│                                                                     │
│ Algorithm 3: LISTEN-U (via LISTEN main.py framework)              │
│   See detailed LISTEN flow below ↓                                 │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: ROUND-ROBIN INTERLEAVING (app_new.py:259-281)             │
│                                                                     │
│ Combine 3 algorithms fairly:                                       │
│   Position 1: Cheapest #1                                          │
│   Position 2: Fastest #1                                           │
│   Position 3: LISTEN-U #1                                          │
│   Position 4: Cheapest #2                                          │
│   Position 5: Fastest #2                                           │
│   Position 6: LISTEN-U #2                                          │
│   ...                                                               │
│   Position 30: LISTEN-U #10                                        │
│                                                                     │
│ Result: 30 flights total (10 from each algorithm)                  │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: USER INTERACTION (Streamlit UI)                            │
│                                                                     │
│ User sees 30 flights in interleaved order                          │
│ User clicks "+" to add 5 flights to shortlist                      │
│ User drags to reorder their top 5                                  │
│ User clicks "Submit Rankings"                                      │
│                                                                     │
│ Data collected:                                                    │
│   - Original prompt                                                │
│   - 30 flights shown (with algorithm labels)                       │
│   - User's top 5 (in ranked order)                                 │
│   - Timestamp, session_id                                          │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: MYSQL STORAGE (backend/db.py - TO BE IMPLEMENTED)          │
│                                                                     │
│ Tables:                                                            │
│   - searches (search_id, session_id, prompt, timestamp)           │
│   - flights_shown (flight_id, search_id, flight_data, algorithm)  │
│   - user_rankings (ranking_id, search_id, flight_id, rank)        │
│                                                                     │
│ Enables offline analysis:                                          │
│   - Algorithm comparison (which ranks better?)                     │
│   - Position bias (do users prefer top results?)                   │
│   - NDCG, Precision@K metrics                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## LISTEN-U Algorithm Flow (Detailed)

LISTEN-U uses the **official LISTEN repository** (`https://github.com/AdamJovine/LISTEN`) via its `main.py` framework. We do NOT reimplement LISTEN - we just prepare data in their format and call their code.

### How LISTEN Integration Works:

```
50 Amadeus Flights
    ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3a: Convert to LISTEN CSV Format                           │
│ (backend/listen_data_converter.py)                              │
│                                                                  │
│ flights_to_listen_csv(flights) →                                │
│   LISTEN/input/flight_20251110_174532.csv                       │
│                                                                  │
│ CSV columns (matches LISTEN's expected format):                 │
│   - name, origin, destination                                   │
│   - departure time, arrival time, duration                      │
│   - stops, price                                                │
│   - dis_from_origin, dis_from_dest                              │
│   - departure_seconds, arrival_seconds, duration_min            │
└──────────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3b: Generate LISTEN YAML Config                            │
│ (backend/listen_data_converter.py)                              │
│                                                                  │
│ generate_listen_config(user_prompt, csv_filename) →             │
│   LISTEN/configs/flight_20251110_174532.yml                     │
│                                                                  │
│ Config includes:                                                 │
│   - tag: "flight_20251110_174532"                               │
│   - data_csv: "input/flight_20251110_174532.csv"                │
│   - metric_columns: [name, origin, ..., duration_min]           │
│   - metric_signs: {stops: -1, price: -1, ...}                   │
│   - prompts: {scenario_header, comparison_base, ...}            │
│   - modes: {User: {prompt: user_prompt, weights: {...}}}        │
└──────────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3c: Call LISTEN main.py                                    │
│ (backend/listen_main_wrapper.py)                                │
│                                                                  │
│ subprocess.run([                                                 │
│   "python3", "main.py",                                          │
│   "--scenario", "flight_20251110_174532",                       │
│   "--algo", "comparison",                                        │
│   "--mode", "User",                                              │
│   "--max-iters", "5",                                            │
│   "--model-type", "gp",                                          │
│   "--acq", "eubo",                                               │
│   "--api-model", "gemini",                                       │
│   "--comparison-batch-size", "4"                                 │
│ ], cwd=LISTEN_DIR)                                               │
└──────────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3d: LISTEN Executes (inside LISTEN repo)                   │
│                                                                  │
│ LISTEN/main.py:                                                  │
│   1. Loads CSV and config                                        │
│   2. Initializes BatchDuelingBanditOptimizer                    │
│   3. FOR each of 5 iterations:                                   │
│        a. select_next_batch() → 4 flight indices                 │
│        b. Build comparison prompt with 4 flights                 │
│        c. Call Gemini (via LISTEN/gemini.py):                    │
│           "Which flight is BEST? A, B, C, or D?"                 │
│        d. Gemini responds: "FINAL: B"                            │
│        e. Record winner in BatchComparisonResult                 │
│        f. Train Gaussian Process on all comparisons so far       │
│   4. Get final utilities for ALL flights (GP prediction)        │
│   5. Rank flights by utility (highest first)                     │
│   6. Save results to LISTEN/output/flight_20251110_174532/      │
└──────────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────────┐
│ STEP 3e: Parse LISTEN Output                                    │
│ (backend/listen_main_wrapper.py)                                │
│                                                                  │
│ Read LISTEN output files and extract top 10 ranked flights      │
│ Map indices back to original flight objects                     │
│ Return ranked_flights[:10]                                       │
└──────────────────────────────────────────────────────────────────┘
```

**Key Point**: We're using LISTEN's actual published code. We just:
1. Convert our flight data → their CSV format
2. Generate a YAML config with the user's prompt
3. Call their `main.py`
4. Parse their output

This ensures we get the **real LISTEN algorithm**, not a reimplementation.

---

## File Structure

```
flight_app/
├── app_new.py                          # Main Streamlit UI
├── .env                                # API keys (AMADEUS_API_KEY, GEMINI_API_KEY)
├── requirements.txt                    # Python dependencies
│
├── backend/
│   ├── amadeus_client.py              # Amadeus API wrapper (OAuth2 + search)
│   ├── prompt_parser.py               # Gemini-based natural language parsing
│   ├── listen_data_converter.py       # Convert flights → LISTEN CSV/YAML
│   ├── listen_main_wrapper.py         # Wrapper to call LISTEN main.py
│   ├── listen_wrapper.py              # OLD custom LISTEN implementation (NOT USED)
│   ├── gemini_llm_client.py           # Gemini client for LISTEN (NOT USED - LISTEN has its own)
│   ├── db.py                           # Database functions (TO BE IMPLEMENTED)
│   └── utils/
│       └── parse_duration.py          # Parse ISO 8601 durations
│
├── LISTEN/                             # Symbolic link → /Users/francisbahk/LISTEN
│   ├── main.py                        # LISTEN experimental framework ← WE CALL THIS
│   ├── LISTEN.py                      # BatchDuelingBanditOptimizer class
│   ├── GP.py                          # Gaussian Process model
│   ├── gemini.py                      # Gemini LLM client (used by LISTEN)
│   ├── Experiment.py                  # Experiment runner
│   ├── configs/                       # Our generated configs go here
│   │   └── flight_TIMESTAMP.yml
│   └── input/                         # Our generated CSVs go here
│       └── flight_TIMESTAMP.csv
│
└── Documentation/
    ├── CONTEXT.md                      # THIS FILE - Complete system overview
    ├── SYSTEM_EXPLANATION.md           # Detailed technical explanation (OLD)
    ├── LISTEN_INTEGRATION_EXPLAINED.md # LISTEN integration details (OLD)
    ├── PLAN.md                         # Original development plan
    └── plan2.md                        # LISTEN submodule setup guide
```

---

## Key Components Explained

### 1. Prompt Parsing (backend/prompt_parser.py)

**Purpose**: Convert natural language → structured JSON

**How it works**:
- Uses Gemini 2.0 Flash
- Sends detailed prompt with examples (e.g., "Houston" → ["IAH", "HOU"])
- Parses JSON response
- Falls back to regex if Gemini fails

**Critical Fix**: Must include **explicit city→airport examples** or Gemini extracts random 3-letter words from the sentence (e.g., "FLY" instead of "IAH").

**Example Input**:
```
"I want to fly from Houston to NYC on Nov 20. I like cheap flights"
```

**Example Output**:
```json
{
  "origins": ["IAH", "HOU"],
  "destinations": ["JFK", "LGA", "EWR"],
  "departure_date": "2025-11-20",
  "preferences": {"prefer_cheap": true}
}
```

### 2. Amadeus API (backend/amadeus_client.py)

**Authentication**: OAuth2 client credentials flow
- POST to `https://test.api.amadeus.com/v1/security/oauth2/token`
- Get access token (cached for 30 minutes)

**Flight Search**:
- GET to `https://test.api.amadeus.com/v2/shopping/flight-offers`
- Query params: `origin`, `destination`, `departureDate`, `adults`, `max`
- Returns JSON with `data` array of flight offers

**Response Structure**:
```json
{
  "data": [
    {
      "id": "1",
      "price": {"total": "204.00", "currency": "USD"},
      "itineraries": [{
        "duration": "PT1H22M",
        "segments": [{
          "departure": {"iataCode": "ITH", "at": "2024-11-15T10:28:00"},
          "arrival": {"iataCode": "IAD", "at": "2024-11-15T11:50:00"},
          "carrierCode": "UA",
          "number": "1234"
        }]
      }]
    }
  ]
}
```

### 3. LISTEN Integration (3-file system)

#### File 1: listen_data_converter.py
- `flights_to_listen_csv()`: Converts Amadeus dicts → CSV
- `generate_listen_config()`: Creates YAML config from user prompt

#### File 2: listen_main_wrapper.py
- Creates timestamped files (e.g., `flight_20251110_174532.csv/yml`)
- Runs `subprocess` to call LISTEN's `main.py`
- Parses output and returns top 10 flights

#### File 3: LISTEN/main.py (external repo)
- Loads our CSV and YAML
- Runs comparison-based dueling bandit algorithm
- Uses Gemini for pairwise/batch comparisons
- Trains Gaussian Process
- Outputs ranked flights

### 4. User Interface (app_new.py)

**Session State Variables**:
- `session_id`: Unique ID for this session
- `flights`: All flights from Amadeus
- `interleaved_results`: 30 flights (10 from each algorithm)
- `shortlist`: User's top 5 selected flights
- `parsed_params`: Parsed search parameters

**UI Components**:
1. **Prompt input**: Text area for natural language query
2. **Flight list**: 30 flights with algorithm labels
3. **Shortlist**: Draggable list of top 5 (using streamlit-sortables)
4. **Submit button**: Saves rankings to database

---

## Database Schema (TO BE IMPLEMENTED)

### Table: `searches`
Stores each search query.

```sql
CREATE TABLE searches (
  search_id INT AUTO_INCREMENT PRIMARY KEY,
  session_id VARCHAR(255) NOT NULL,
  user_prompt TEXT NOT NULL,
  parsed_origins JSON,
  parsed_destinations JSON,
  parsed_preferences JSON,
  departure_date DATE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_session (session_id),
  INDEX idx_created (created_at)
);
```

### Table: `flights_shown`
Stores all 30 flights shown to user.

```sql
CREATE TABLE flights_shown (
  id INT AUTO_INCREMENT PRIMARY KEY,
  search_id INT NOT NULL,
  flight_data JSON NOT NULL,         -- Full flight object
  algorithm VARCHAR(50) NOT NULL,    -- 'Cheapest', 'Fastest', 'LISTEN-U'
  algorithm_rank INT NOT NULL,       -- Position in that algorithm (1-10)
  display_position INT NOT NULL,     -- Position shown to user (1-30)
  FOREIGN KEY (search_id) REFERENCES searches(search_id) ON DELETE CASCADE,
  INDEX idx_search (search_id)
);
```

### Table: `user_rankings`
Stores user's top 5 ranked flights.

```sql
CREATE TABLE user_rankings (
  id INT AUTO_INCREMENT PRIMARY KEY,
  search_id INT NOT NULL,
  flight_id INT NOT NULL,            -- FK to flights_shown
  user_rank INT NOT NULL,            -- User's ranking (1-5)
  submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (search_id) REFERENCES searches(search_id) ON DELETE CASCADE,
  FOREIGN KEY (flight_id) REFERENCES flights_shown(id) ON DELETE CASCADE,
  INDEX idx_search (search_id)
);
```

### Example Query: Algorithm Performance

```sql
-- Which algorithm appears most in user's top 5?
SELECT
  f.algorithm,
  COUNT(*) as times_selected,
  AVG(r.user_rank) as avg_rank
FROM user_rankings r
JOIN flights_shown f ON r.flight_id = f.id
GROUP BY f.algorithm
ORDER BY times_selected DESC;
```

---

## Environment Variables

Required in `.env`:

```bash
# Amadeus API (get from https://developers.amadeus.com)
AMADEUS_API_KEY=your_api_key_here
AMADEUS_API_SECRET=your_api_secret_here

# Gemini API (get from https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_key_here
# OR
GOOGLE_API_KEY=your_gemini_key_here

# MySQL Database (for storing rankings)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=flight_rankings
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
```

---

## Dependencies (requirements.txt)

```
streamlit>=1.28.0              # Web UI framework
sqlalchemy>=2.0.35             # Database ORM
pymysql==1.1.0                 # MySQL driver
cryptography==41.0.7           # Required by pymysql
requests==2.31.0               # HTTP client for Amadeus
python-dotenv==1.0.0           # Load .env files
google-generativeai>=0.3.0     # Gemini API
streamlit-sortables>=0.2.0     # Drag-and-drop UI

# LISTEN dependencies
numpy<2
scipy
scikit-learn
groq                           # (LISTEN supports multiple LLM providers)
pyyaml                         # For YAML config files
```

---

## Performance Characteristics

### Latency Breakdown (per search):

| Step | Time | Notes |
|------|------|-------|
| Gemini prompt parsing | 1-2s | API call |
| Amadeus OAuth | 0.1s | Cached after first call |
| Amadeus flight search | 2-5s | Network + API processing |
| Cheapest algorithm | <0.01s | Simple sort |
| Fastest algorithm | <0.01s | Simple sort |
| **LISTEN-U** | **30-60s** | **Calls LISTEN main.py** |
| ├─ CSV/YAML generation | 0.1s | File I/O |
| ├─ LISTEN subprocess | 25-55s | Python startup + algorithm |
| │  ├─ 5 iterations × Gemini | ~15s | 5 comparisons × 3s each |
| │  └─ GP training | ~10s | Gaussian Process fitting |
| └─ Parse output | 0.1s | Read files |
| Interleaving | <0.01s | List operations |
| **Total** | **35-70 seconds** | Dominated by LISTEN |

### Cost per Search:

| Service | Calls | Cost |
|---------|-------|------|
| Gemini (parsing) | 1 | $0.00001 |
| Gemini (LISTEN comparisons) | 5 | $0.00005 |
| Amadeus (test API) | 1 | Free |
| **Total** | | **~$0.00006** |

---

## Current Limitations

1. **No database storage yet** - Rankings not saved (TODO in app_new.py:396)
2. **Test API only** - Amadeus test API has limited routes (ITH not available)
3. **Single origin/dest** - Only searches first origin/destination pair
4. **No error recovery** - If LISTEN fails, falls back to simple ranking
5. **No user authentication** - Anonymous sessions only
6. **Local deployment only** - Not configured for cloud deployment

---

## Next Steps

1. ✅ Integrate LISTEN main.py framework
2. ✅ Fix Gemini airport parsing
3. **⏳ Implement MySQL database storage**
4. **⏳ Deploy to cloud (Streamlit Cloud + MySQL instance)**
5. **⏳ Add user authentication (optional)**
6. **⏳ Switch to Amadeus production API**
7. **⏳ Add feedback analysis dashboard**

---

## Research Questions This System Can Answer

With the database populated:

1. **Algorithm Comparison**: Does LISTEN-U rank flights better than simple heuristics?
2. **Position Bias**: Do users prefer flights shown at the top regardless of algorithm?
3. **Preference Alignment**: Do user rankings match LISTEN's predicted utilities?
4. **Feature Importance**: Which flight attributes (price, duration, stops) matter most?
5. **Gemini Accuracy**: How well does Gemini parse complex flight queries?

**Metrics to Compute**:
- NDCG@5 (Normalized Discounted Cumulative Gain)
- Precision@K
- Spearman rank correlation
- Algorithm selection rate

---

## Troubleshooting

### "FLY" and "NEW" extracted as airports
**Problem**: Gemini extracting random 3-letter words instead of airport codes

**Solution**: Prompt needs explicit examples (Houston → IAH/HOU). See prompt_parser.py lines 69-85.

### LISTEN times out
**Problem**: LISTEN main.py takes >5 minutes

**Solution**: Reduce `--max-iters` or `--comparison-batch-size`. Currently: 5 iterations, batch size 4.

### Amadeus "airport not found"
**Problem**: Small airports (ITH, ELM) not in test API

**Solution**: `get_test_api_fallback()` maps small airports to nearby major ones (ITH → SYR).

### Rankings not saved
**Problem**: "TODO: Save to database" placeholder

**Solution**: Implement `save_ranking()` function in backend/db.py (see Database Schema section above).

---

## Contact & Resources

- **LISTEN Repository**: https://github.com/AdamJovine/LISTEN
- **Amadeus API Docs**: https://developers.amadeus.com/self-service/category/flights
- **Gemini API Docs**: https://ai.google.dev/gemini-api/docs
- **Streamlit Docs**: https://docs.streamlit.io

---

**Last Updated**: November 10, 2025
**Version**: 2.0 (with LISTEN main.py integration)