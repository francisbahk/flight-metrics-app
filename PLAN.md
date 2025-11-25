# 🚀 Flight Evaluation Web App - Implementation Plan

**Last Updated:** Nov 10, 2025

---

## 📋 Executive Summary

Transform the current flight_app into a clean, Suno-like interface for collecting user feedback on flight rankings. Integrate LISTEN-U algorithm alongside simple algorithms (cheapest, fastest) and display interleaved results for unbiased evaluation.

---

## 🏗️ Architecture Overview

### Repository Structure (NEW)

```
flight-evaluation/  (NEW public repo)
├── .gitmodules
├── LISTEN/  (submodule → https://github.com/AdamJovine/LISTEN)
│   ├── main.py
│   ├── LISTEN.py
│   ├── LLM.py
│   └── requirements.txt
├── streamlit_app.py  (NEW clean UI)
├── core/
│   ├── interleave.py  (the "R" file)
│   ├── algorithms.py  (cheapest, fastest)
│   └── listen_wrapper.py  (calls LISTEN/LISTEN.py)
├── backend/
│   ├── amadeus_client.py  (from flight_app)
│   ├── prompt_parser.py  (NEW - uses Gemini)
│   └── database.py  (MySQL for feedback)
├── requirements.txt
└── .env  (Amadeus, Gemini API keys)

flight_app/  (CURRENT - keep separate for development)
└── (current state preserved)
```

---

## 🎯 Core Features (MVP for Meeting)

### 1. Clean Suno-Like UI ✨
**Priority:** CRITICAL
**Time:** 1 hour

**Design:**
```
┌──────────────────────────────────────────────────────┐
│  ✈️ Flight Ranker                                    │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │ Describe your flight needs...                  │ │
│  │                                                │ │
│  │ (large text area, 200px height)                │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│          [🔍 Search Flights]  (large button)        │
└──────────────────────────────────────────────────────┘
```

**Implementation:**
- Single page, no tabs
- Minimal styling, focus on functionality
- Text area for prompt (multiline, 200px height)
- Large search button
- Loading spinner during search

---

### 2. Prompt Parsing with LLM 🧠
**Priority:** CRITICAL
**Time:** 2 hours

**Use:** Gemini API (from LISTEN repo pattern)
**File:** `backend/prompt_parser.py`

**Function:**
```python
def parse_flight_prompt(prompt: str) -> dict:
    """
    Parse natural language prompt into structured flight search.

    Args:
        prompt: User's natural language query

    Returns:
        {
            'origin': 'ITH',  # or list ['ITH', 'SYR', 'ELM', 'BGM']
            'destination': 'IAD',  # or list ['DCA', 'IAD']
            'date': '2024-10-20',
            'preferences': {
                'prefer_direct': True,
                'avoid_early_departures': True,  # no flights before 7am
                'min_connection_time': 45,  # minutes
                'preferred_airlines': ['UA'],
                'price_sensitivity': 'low',  # low, medium, high
            },
            'constraints': {
                'max_layover_min': 90,
                'avoid_airports': ['JFK', 'EWR'],
                'fly_america_act': True,
            }
        }
    """
```

**LLM Prompt Template:**
```
Extract flight search parameters from this query:
"{user_prompt}"

Return JSON with:
- origin (airport code or list)
- destination (airport code or list)
- date (YYYY-MM-DD)
- preferences (dict of user preferences)
- constraints (hard requirements)

Focus on: price sensitivity, timing preferences, connection requirements, airline preferences.
```

**Fallback:** If parsing fails, use simple regex (like current nl_parser.py)

---

### 3. Flight Search (Existing)
**Priority:** MEDIUM
**Time:** 30 min (just wire up)

**Use existing:** `backend/amadeus_client.py`

**Enhancement:** Handle multiple origins/destinations
```python
# If prompt gives multiple airports
for origin in origins:
    for destination in destinations:
        flights = amadeus.search_flights(origin, destination, date)
        all_flights.extend(flights)
```

**Test API constraints:**
- Use major airports (JFK, LAX, IAD, etc.)
- If small airport (ITH) requested, fallback to nearby major (SYR, BGM)

---

### 4. Algorithm Integration 🤖
**Priority:** CRITICAL
**Time:** 3 hours

#### A. Simple Algorithms (Existing)
**File:** `core/algorithms.py`

```python
def cheapest(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """Sort by price ascending"""
    return sorted(flights, key=lambda x: x['price'])

def fastest(flights: List[Dict], preferences: Dict) -> List[Dict]:
    """Sort by duration ascending"""
    return sorted(flights, key=lambda x: x['duration_min'])
```

#### B. LISTEN-U Integration (NEW)
**File:** `core/listen_wrapper.py`

**Challenge:** LISTEN.py is designed for essays, not flights. Need to adapt.

**Approach:**
```python
import sys
sys.path.insert(0, 'LISTEN/')
from LISTEN import DuelingBanditOptimizer
from LLM import FreeLLMPreferenceClient

def listen_u(flights: List[Dict], preferences: Dict, prompt: str) -> List[Dict]:
    """
    Rank flights using LISTEN-U algorithm.

    Process:
    1. Convert flights to feature vectors
    2. Run LISTEN dueling bandit for N iterations
    3. Get final utility weights
    4. Rank all flights by learned utility
    5. Return top-k
    """

    # Convert flights to feature matrix
    features = flights_to_features(flights)  # price, duration, stops, etc.

    # Initialize LISTEN optimizer
    optimizer = DuelingBanditOptimizer(
        all_options=list(range(len(flights))),
        features=features,
        client=FreeLLMPreferenceClient(
            provider="groq",  # or gemini
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-120b"
        ),
        batch_size=4,  # small for speed
        n_champions=10,  # top-10
    )

    # Run LISTEN iterations (fewer for speed)
    for iteration in range(5):  # instead of 25
        batch = optimizer.select_next_batch()
        # Let LLM compare flights
        results = optimizer.process_batch_comparisons(batch, prompt)
        optimizer.update_model(results)

    # Get final rankings
    utilities = optimizer.get_final_utilities()
    ranked_indices = np.argsort(utilities)[::-1]

    return [flights[i] for i in ranked_indices]

def flights_to_features(flights: List[Dict]) -> np.ndarray:
    """
    Convert flight dicts to feature vectors.

    Features:
    - price (normalized 0-1)
    - duration (normalized)
    - stops (0, 1, 2+)
    - departure_hour (0-23)
    - arrival_hour
    - connection_time (if stops > 0)
    """
    features_list = []
    for flight in flights:
        features_list.append([
            flight['price'] / 1000,  # normalize
            flight['duration_min'] / 600,
            min(flight['stops'], 2),  # cap at 2
            # Add more features as needed
        ])
    return np.array(features_list)
```

**Simplification for MVP:**
- Reduce iterations from 25 → 5 (for speed)
- Use simulated preferences instead of LLM for first version?
- Or use cached LLM responses for common comparisons?

**DECISION NEEDED:** Should LISTEN-U use LLM or simulated utility for MVP?

---

### 5. Interleaving Logic 🔀
**Priority:** CRITICAL
**Time:** 1 hour

**File:** `core/interleave.py`

**Method:** Simple round-robin (Team Draft)

```python
def interleave_rankings(
    algorithms: Dict[str, List[Dict]],  # {'Cheapest': [flights], 'LISTEN-U': [flights]}
    k: int = 10  # top-k from each
) -> List[Dict]:
    """
    Interleave top-k flights from each algorithm.

    Args:
        algorithms: Dict mapping algorithm name to ranked flights
        k: Number of flights to take from each algorithm

    Returns:
        Interleaved list with metadata about source
    """
    interleaved = []

    # Take top-k from each
    for i in range(k):
        for algo_name, flights in algorithms.items():
            if i < len(flights):
                interleaved.append({
                    'flight': flights[i],
                    'source_algorithm': algo_name,
                    'rank_in_algorithm': i + 1,
                    'display_position': len(interleaved) + 1
                })

    return interleaved  # Length = 3 algorithms × 10 = 30 flights
```

**Unbiased property:** Round-robin ensures no algorithm gets preferential positions.

---

### 6. Results Display & Shortlist 📊
**Priority:** CRITICAL
**Time:** 2 hours

**UI Design:**
```
┌──────────────────────────────────────────────────────┐
│  30 Flights Found                                    │
│                                                      │
│  ☐ Flight 1: AA123 JFK→LAX $299 • 6h • Direct      │
│     Source: Cheapest (#1)                            │
│                                                      │
│  ☐ Flight 2: UA456 SYR→IAD $450 • 4h • 1 stop      │
│     Source: Fastest (#1)                             │
│                                                      │
│  ☐ Flight 3: DL789 ITH→DCA $320 • 5h • Direct      │
│     Source: LISTEN-U (#1)                            │
│                                                      │
│  [... 27 more flights ...]                           │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │ Your Shortlist (0/5 selected)                  │ │
│  │                                                │ │
│  │ Drag to rank your top 5 choices:               │ │
│  │ (empty - click checkboxes above to add)        │ │
│  └────────────────────────────────────────────────┘ │
│                                                      │
│  [Submit Feedback]  (disabled until 5 selected)     │
└──────────────────────────────────────────────────────┘
```

**Functionality:**
- Checkbox to add flight to shortlist (max 5)
- Drag-and-drop to reorder shortlist
- Submit button saves to database

**Streamlit Components:**
- Use `st.checkbox()` for selection
- Use `st.container()` for shortlist
- For drag-drop: Use streamlit-sortables library OR manual up/down buttons

**Simple version (no drag-drop):**
```python
# Let user number their preferences 1-5
for flight in shortlisted_flights:
    rank = st.number_input(
        f"Rank for {flight['airline']}{flight['flight_number']}",
        min_value=1,
        max_value=5,
        key=f"rank_{flight['id']}"
    )
```

---

### 7. Data Collection 💾
**Priority:** CRITICAL
**Time:** 1 hour

**Database Schema:**

```sql
CREATE TABLE feedback_sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_prompt TEXT,
    parsed_params JSON,
    num_flights_returned INT
);

CREATE TABLE feedback_flights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(36),
    flight_id VARCHAR(50),
    source_algorithm VARCHAR(50),
    rank_in_algorithm INT,
    display_position INT,
    flight_data JSON,  -- full flight details
    FOREIGN KEY (session_id) REFERENCES feedback_sessions(session_id)
);

CREATE TABLE feedback_rankings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(36),
    flight_id VARCHAR(50),
    user_rank INT,  -- 1-5
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES feedback_sessions(session_id)
);
```

**Python Code:**
```python
def save_feedback(session_id, prompt, flights, user_rankings):
    """Save user feedback to MySQL for offline evaluation."""

    # Save session
    db.execute("""
        INSERT INTO feedback_sessions (session_id, user_prompt, num_flights_returned)
        VALUES (%s, %s, %s)
    """, (session_id, prompt, len(flights)))

    # Save all flights shown
    for flight in flights:
        db.execute("""
            INSERT INTO feedback_flights
            (session_id, flight_id, source_algorithm, rank_in_algorithm, display_position, flight_data)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (session_id, flight['flight']['id'], flight['source_algorithm'],
              flight['rank_in_algorithm'], flight['display_position'], json.dumps(flight['flight'])))

    # Save user's ranking
    for flight_id, rank in user_rankings.items():
        db.execute("""
            INSERT INTO feedback_rankings (session_id, flight_id, user_rank)
            VALUES (%s, %s, %s)
        """, (session_id, flight_id, rank))
```

---

## ⏱️ Implementation Timeline (12 hours)

### Hour 1-2: Setup & Infrastructure
- [ ] Create new `flight-evaluation` directory
- [ ] Add LISTEN as git submodule
- [ ] Set up requirements.txt (merge flight_app + LISTEN dependencies)
- [ ] Configure .env with Gemini API key
- [ ] Test LISTEN imports work

### Hour 3-4: Prompt Parsing
- [ ] Create `backend/prompt_parser.py`
- [ ] Implement Gemini API integration
- [ ] Test with example prompt
- [ ] Add fallback to simple regex parsing

### Hour 5-6: Clean UI
- [ ] Create new `streamlit_app.py` (Suno-style)
- [ ] Single text area for prompt
- [ ] Wire up to prompt parser
- [ ] Connect to Amadeus search
- [ ] Display basic flight list

### Hour 7-8: Algorithm Integration
- [ ] Create `core/algorithms.py` (cheapest, fastest)
- [ ] Create `core/listen_wrapper.py`
- [ ] Adapt LISTEN for flights (feature extraction)
- [ ] Test LISTEN-U with sample flights
- [ ] DECISION: Use simulated utility or real LLM?

### Hour 9: Interleaving
- [ ] Create `core/interleave.py`
- [ ] Implement round-robin interleaving
- [ ] Test with 3 algorithms × 10 flights = 30 results

### Hour 10: Shortlist & Ranking UI
- [ ] Add checkbox selection for flights
- [ ] Create shortlist container
- [ ] Add ranking interface (number inputs or drag-drop)
- [ ] Add submit button

### Hour 11: Data Collection
- [ ] Create database tables
- [ ] Implement save_feedback() function
- [ ] Test end-to-end: search → rank → save
- [ ] Verify data appears in MySQL

### Hour 12: Testing & Polish
- [ ] End-to-end test with real prompt
- [ ] Fix any bugs
- [ ] Add loading indicators
- [ ] Deploy to Streamlit Cloud
- [ ] Test deployed version

---

## 🚨 Critical Decisions Needed

### Decision 1: LISTEN-U for MVP?
**Options:**
- **A:** Use simulated utility (fast, reliable)
  - Pro: Guaranteed to work
  - Con: Not "real" LISTEN-U
- **B:** Use LLM with reduced iterations (5 instead of 25)
  - Pro: Real LISTEN-U behavior
  - Con: Slow (5-10 sec per query), API costs
- **C:** Mock LISTEN-U for demo (random + noise)
  - Pro: Fast demo
  - Con: Not functional

**Recommendation:** Option A for MVP (simulated), upgrade to B post-meeting

### Decision 2: Drag-Drop Ranking?
**Options:**
- **A:** Use streamlit-sortables library (drag-drop)
  - Pro: Nice UX
  - Con: Extra dependency, might be buggy
- **B:** Use number inputs (1-5)
  - Pro: Simple, reliable
  - Con: Less intuitive
- **C:** Use up/down buttons
  - Pro: Mobile-friendly
  - Con: More clicks

**Recommendation:** Option B for MVP (number inputs)

### Decision 3: Prompt Parsing Quality?
**Options:**
- **A:** Full Gemini parsing (like example prompt)
  - Pro: Handles complex queries
  - Con: Slow, costs money
- **B:** Simple regex + keywords
  - Pro: Fast, free
  - Con: Limited understanding
- **C:** Hybrid (regex first, LLM if ambiguous)
  - Pro: Best of both
  - Con: More code

**Recommendation:** Option A (Gemini) - it's core to the research

---

## 📦 Dependencies

### New Dependencies
```txt
# LLM for prompt parsing
google-generativeai>=0.3.0
groq  # for LISTEN-U

# For drag-drop (optional)
streamlit-sortables

# Existing from flight_app
streamlit>=1.28.0
sqlalchemy>=2.0.35
pymysql==1.1.0
requests==2.31.0
python-dotenv==1.0.0

# From LISTEN
numpy<2
scipy
scikit-learn
torch  # (heavy - needed for LISTEN)
transformers  # (heavy - needed for LISTEN)
```

---

## 🧪 Testing Checklist

### Before Meeting:
- [ ] Can parse complex prompt (like example)
- [ ] Returns 30 flights (10 from each algorithm)
- [ ] Can select 5 flights for shortlist
- [ ] Can rank shortlist 1-5
- [ ] Submit saves to database
- [ ] Can query database to see saved feedback
- [ ] Works on Streamlit Cloud deployment

### Demo Flow:
1. Paste example prompt about Ithaca → Reston flight
2. Click search (show loading spinner)
3. See 30 flights displayed
4. Click 5 interesting flights
5. Number them 1-5
6. Submit feedback
7. (Behind scenes) Query database to show data was saved

---

## 🎯 Success Criteria

**Must Have (for meeting):**
- ✅ Clean single-page UI (like Suno)
- ✅ LLM prompt parsing (Gemini)
- ✅ 3 algorithms working (cheapest, fastest, LISTEN-U)
- ✅ 30 interleaved results displayed
- ✅ Shortlist selection (5 flights)
- ✅ Ranking interface (1-5)
- ✅ Data saved to MySQL

**Nice to Have:**
- Drag-and-drop ranking
- Beautiful styling
- Multiple search sessions in UI
- Export data as CSV
- Admin dashboard to view collected data

---

## 🔑 API Keys Needed

Add to `.env`:
```bash
# Amadeus (existing)
AMADEUS_API_KEY=your_key
AMADEUS_API_SECRET=your_secret

# Gemini (for prompt parsing)
GEMINI_API_KEY=your_gemini_key  # or GOOGLE_API_KEY

# Groq (for LISTEN-U)
GROQ_API_KEY=your_groq_key

# MySQL (existing)
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DB=flights
```

---

## 📝 Notes

- Flight_app repo stays separate for experimentation
- New flight-evaluation repo is clean, focused on research
- LISTEN submodule keeps that code private
- Can share evaluation repo publicly without exposing LISTEN
- Data collection format matches LISTEN paper methods for offline evaluation

---

## ❓ Open Questions

1. **LISTEN-U speed:** 5 iterations acceptable or need caching?
2. **Feature engineering:** Which flight features matter most for utility learning?
3. **Prompt parsing:** Should we show user the parsed parameters for confirmation?
4. **Error handling:** What if Amadeus returns < 30 flights total?
5. **Session management:** Allow multiple searches per session or one-shot?

---

**Status:** ⏳ AWAITING APPROVAL
**Next Step:** Review this plan, answer open questions, then begin implementation!
