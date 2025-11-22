# LILO Implementation Plan - Feature Branch

**Branch:** `feature/lilo-implementation`
**Status:** Planning Phase
**Paper:** https://arxiv.org/pdf/2510.17671

---

## Overview

This document outlines the plan to implement LILO (Bayesian Optimization with Interactive Natural Language Feedback) alongside major architectural changes and a human vs LLM evaluation system.

## Major Components

1. **LILO Algorithm Implementation** - Interactive preference learning with natural language feedback
2. **Offline Data Collection** - Track user interactions (clicks, scrolls, hovers, time spent)
3. **Streamlit → FastAPI/Uvicorn Migration** - Complete architectural rewrite
4. **Human vs LLM Evaluation** - Compare human recommendations vs LISTEN-U vs LILO
5. **New Database Schema** - Store all interaction data and evaluation results

---

## Part 1: LILO Algorithm Implementation

### What is LILO?

**LILO** = "Bayesian Optimization with Interactive Natural Language Feedback"

Unlike LISTEN-U (which uses initial prompt only), LILO:
- Presents candidate flights to user
- Asks user to explain preferences in natural language
- Uses LLM to interpret feedback
- Refines recommendations iteratively
- Learns utility function from conversational feedback

### LILO Workflow (Proposed)

```
Round 1:
  1. User provides initial prompt
  2. Search flights (Amadeus API → 50 flights)
  3. LILO presents N candidate flights
  4. User ranks top-k and provides natural language feedback
  5. LLM interprets feedback → extract preference signals

Round 2 (Refinement):
  1. LILO uses learned preferences to generate new candidates
  2. Present M flights (mix of new + previous)
  3. User ranks top-k and provides more feedback
  4. LLM refines preference model

Round 3+ (Optional):
  Continue refinement until convergence or max rounds

Final:
  LILO ranks ALL flights using learned utility
  Return top 10 recommendations
```

### Questions for LILO Implementation

#### Q1: Paper-Specific Details
- [ ] **How many refinement rounds does the paper recommend?** (Need to read Appendix)
- [ ] **How many flights (N) to show per round?** (Paper experiments?)
- [ ] **How many flights (k) should user rank per round?** (Top 3? Top 5?)
- [ ] **Do we show ALL 50 flights or subset?**

#### Q2: Prompts
The paper mentions "Prompt 2-6 in appendices" but I need exact wording:
- [ ] **Prompt to ask user for natural language feedback** (after they rank)
- [ ] **Prompt for LLM to extract preference signals** from user feedback
- [ ] **Prompt for LLM to update utility function** based on feedback
- [ ] **Should I extract these from the paper PDF Appendix?**

#### Q3: Implementation Details
- [ ] **Bayesian optimization backend**: Use GPyOpt? scikit-optimize? Custom implementation?
- [ ] **How to represent flights as feature vectors?** (price, duration, stops, departure_time, airline, etc.)
- [ ] **How to convert natural language → preference signals?** (LLM extracts weights?)
- [ ] **Example feedback**: "I prefer direct flights even if more expensive" → Update utility weights?

#### Q4: Integration with Existing Code
- [ ] **New file structure**:
  ```
  backend/
    lilo/
      __init__.py
      lilo_optimizer.py          # Main LILO algorithm
      preference_extractor.py     # LLM feedback parser
      bayesian_backend.py         # BO implementation
      prompts.py                  # All LILO prompts
  ```
- [ ] **Use existing Gemini client** from listen_main_wrapper.py?
- [ ] **Store LILO state between rounds** (session-based)

### Proposed Default Parameters (Subject to Change)

Based on research context (pending paper confirmation):
- **Initial candidates shown**: 10 flights
- **User ranks**: Top 5 per round
- **Refinement rounds**: 2-3 rounds
- **Final ranking**: All 50 flights using learned utility
- **Total LLM calls**: ~3-5 per round (feedback parsing + utility update)

---

## Part 2: Offline Data Collection

### Goal
Understand which flights users actually look at/consider (beyond just their final top-5 ranking).

### Data to Collect

#### 2.1 Interaction Events
```javascript
// Frontend tracking events
{
  "event_type": "flight_view",        // Flight entered viewport
  "flight_id": "abc123",
  "timestamp": "2025-11-17T22:30:00",
  "scroll_depth": 0.45,               // 45% down page
  "viewport_time_ms": 2300            // Viewed for 2.3 seconds
}

{
  "event_type": "flight_click",       // User clicked expand/details
  "flight_id": "abc123",
  "timestamp": "2025-11-17T22:30:05"
}

{
  "event_type": "flight_hover",       // Mouse hovered over flight
  "flight_id": "abc123",
  "hover_duration_ms": 800
}

{
  "event_type": "filter_applied",     // User filtered results
  "filter_type": "max_price",
  "filter_value": 300,
  "timestamp": "2025-11-17T22:29:00"
}

{
  "event_type": "sort_applied",       // User sorted results
  "sort_by": "duration",
  "sort_order": "asc",
  "timestamp": "2025-11-17T22:29:30"
}
```

#### 2.2 Session-Level Metrics
- Total time spent on results page
- Number of flights viewed (entered viewport)
- Scroll depth percentiles (25%, 50%, 75%, 100%)
- Number of filter/sort interactions
- Flights clicked vs flights ranked (did they rank flights they never clicked?)

#### 2.3 Flight-Level Engagement Scores
For each flight shown, calculate:
- **View score**: Time in viewport / Total session time
- **Interaction score**: Boolean (clicked or hovered)
- **Consideration score**: Composite of view + interaction
- **Final ranking**: Position in user's top-k (if selected)

### Implementation Approach

#### Option A: FastAPI + JavaScript (Recommended with Uvicorn migration)
```javascript
// Frontend (React/vanilla JS)
const trackEvent = async (eventType, eventData) => {
  await fetch('/api/tracking/event', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session_id: sessionId,
      search_id: searchId,
      event_type: eventType,
      event_data: eventData,
      timestamp: new Date().toISOString()
    })
  });
};

// Intersection Observer for viewport tracking
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      trackEvent('flight_view', {
        flight_id: entry.target.dataset.flightId,
        scroll_depth: window.scrollY / document.body.scrollHeight
      });
    }
  });
});
```

```python
# Backend (FastAPI)
@app.post("/api/tracking/event")
async def track_event(event: TrackingEvent):
    # Save to database
    await db.save_tracking_event(event)
    return {"status": "ok"}
```

#### Option B: Streamlit (Current - Limited)
Streamlit has limited JavaScript access, so tracking would be basic:
- Track button clicks
- Track which flights were added to shortlist
- Track ranking order
- **Cannot easily track**: scrolling, viewport time, hover events

**Decision**: Since you're migrating to Uvicorn/FastAPI anyway, use Option A.

### Questions for Offline Data Collection

- [ ] **Which events are highest priority?** (Click? Scroll? Hover?)
- [ ] **Storage**: Separate table `interaction_events` or append to existing tables?
- [ ] **Privacy**: Any PII concerns with tracking scroll/click behavior?
- [ ] **Real-time vs Batch**: Send events immediately or batch every N seconds?

---

## Part 3: Streamlit → FastAPI/Uvicorn Migration

### Current Architecture (Streamlit)
```
User → Streamlit (app.py) → Backend modules → Railway MySQL
                 ↓
         LISTEN/LILO (subprocess)
```

### New Architecture (FastAPI + React)
```
User → React Frontend (port 3000) → FastAPI Backend (port 8000) → Railway MySQL
                                          ↓
                                  LISTEN/LILO (subprocess)
                                          ↓
                                  Gemini API
```

### Migration Strategy

#### Phase 1: Backend API (FastAPI)
Create new FastAPI backend that exposes REST endpoints:

```python
# backend_api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Flight Ranking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.post("/api/search")
async def search_flights(query: SearchQuery):
    """
    1. Parse prompt with Gemini
    2. Search Amadeus
    3. Return 50 flights
    """
    pass

@app.post("/api/rank/cheapest")
async def rank_cheapest(flights: List[Flight]):
    """Return flights sorted by price"""
    pass

@app.post("/api/rank/fastest")
async def rank_fastest(flights: List[Flight]):
    """Return flights sorted by duration"""
    pass

@app.post("/api/rank/listen")
async def rank_listen(request: ListenRequest):
    """Run LISTEN-U algorithm (25 iterations)"""
    pass

@app.post("/api/rank/lilo/init")
async def lilo_init(request: LiloInitRequest):
    """Start LILO session, return initial candidates"""
    pass

@app.post("/api/rank/lilo/refine")
async def lilo_refine(request: LiloRefineRequest):
    """
    Round N of LILO:
    - User feedback (text + rankings)
    - Return refined candidates
    """
    pass

@app.post("/api/rank/lilo/final")
async def lilo_final(session_id: str):
    """Return LILO final top-10 ranking"""
    pass

@app.post("/api/evaluation/human-vs-llm/start")
async def start_evaluation(request: EvalStartRequest):
    """
    Person A submits prompt
    Return search results
    """
    pass

@app.post("/api/evaluation/human-vs-llm/submit-rankings")
async def submit_rankings(request: RankingsRequest):
    """
    Person A, Person B, LISTEN-U submit rankings
    """
    pass

@app.post("/api/tracking/event")
async def track_event(event: TrackingEvent):
    """Save user interaction event"""
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Phase 2: Frontend (React)
Create React frontend that replaces Streamlit UI:

```
frontend_react/
├── public/
│   └── index.html
├── src/
│   ├── App.jsx                    # Main app component
│   ├── index.js                   # Entry point
│   ├── api/
│   │   └── client.js              # API client (fetch wrapper)
│   ├── components/
│   │   ├── SearchForm.jsx         # Natural language search
│   │   ├── FlightList.jsx         # Display 30 interleaved flights
│   │   ├── FlightCard.jsx         # Individual flight card
│   │   ├── ShortlistPanel.jsx     # Drag-drop ranking (react-beautiful-dnd)
│   │   ├── LiloRound.jsx          # LILO refinement round UI
│   │   └── HumanVsLLM.jsx         # Evaluation interface
│   ├── hooks/
│   │   └── useTracking.js         # Custom hook for event tracking
│   └── utils/
│       └── tracking.js            # Tracking utility functions
├── package.json
└── tailwind.config.js
```

Example component:
```jsx
// FlightCard.jsx
import { useTracking } from '../hooks/useTracking';
import { useRef, useEffect } from 'react';

export default function FlightCard({ flight, onSelect }) {
  const cardRef = useRef(null);
  const { trackView, trackClick, trackHover } = useTracking();

  useEffect(() => {
    // Viewport tracking
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            trackView(flight.id);
          }
        });
      },
      { threshold: 0.5 }
    );

    if (cardRef.current) {
      observer.observe(cardRef.current);
    }

    return () => observer.disconnect();
  }, [flight.id]);

  return (
    <div
      ref={cardRef}
      onClick={() => {
        trackClick(flight.id);
        onSelect(flight);
      }}
      onMouseEnter={() => trackHover(flight.id, 'start')}
      className="flight-card"
    >
      <h3>{flight.airline}{flight.flight_number}</h3>
      <p>${flight.price} • {flight.duration_min}min • {flight.stops} stops</p>
      <span className="algorithm-badge">{flight.algorithm} #{flight.rank}</span>
    </div>
  );
}
```

#### Phase 3: Deployment
- **Backend**: Deploy FastAPI on Railway/Render/Fly.io with Uvicorn
- **Frontend**: Deploy React on Vercel/Netlify/Cloudflare Pages
- **Database**: Keep Railway MySQL

### Questions for Migration

- [ ] **Timeline**: When should migration start? (After LILO implementation? Parallel?)
- [ ] **Keep Streamlit version?** (In `main` branch while we develop FastAPI in feature branch)
- [ ] **Frontend framework**: React (recommended)? Vue? Vanilla JS?
- [ ] **Styling**: Tailwind CSS (match current design)? Material-UI? Chakra UI?
- [ ] **Deployment platforms**: Railway for backend? Vercel for frontend?

---

## Part 4: Human vs LLM Evaluation Interface

### Workflow

#### Setup Phase (Person A)
1. Person A enters natural language prompt
2. System searches flights (Amadeus → 50 flights)
3. Person A browses all 50 flights
4. Person A selects and ranks their top-k flights
5. System stores:
   - Person A's prompt
   - Person A's ground-truth ranking
   - All interaction data (which flights Person A viewed/clicked)

#### Recommendation Phase (Parallel)
**Person B (Human Recommender):**
1. Sees Person A's prompt (NOT Person A's ranking)
2. Browses same 50 flights
3. Tries to predict what Person A would like
4. Submits top-k recommendations for Person A

**LISTEN-U:**
1. Takes Person A's prompt
2. Runs 25 iterations with Gemini
3. Returns top-k recommendations

**LILO:**
1. Takes Person A's prompt
2. Runs interactive refinement rounds (using Person A's feedback from earlier)
3. Returns top-k recommendations

#### Evaluation Phase (Team Draft)
**Team Draft Comparison:**
Display flights from Person B and LISTEN-U in alternating order:
```
Position 1: Person B's #1
Position 2: LISTEN-U's #1
Position 3: Person B's #2
Position 4: LISTEN-U's #2
...
Position 2k: LISTEN-U's #k
```

Person A reviews interleaved list and indicates which flights they prefer.

**Metrics to Calculate:**
- **Overlap**: How many flights appear in both Person B and LISTEN-U top-k?
- **Precision@k**: How many of Person B/LISTEN-U recommendations match Person A's ground truth?
- **NDCG@k**: Normalized discounted cumulative gain
- **Win Rate**: Team Draft - does Person B or LISTEN-U get more "picks"?

### UI Mockup (FastAPI + React)

```
┌─────────────────────────────────────────────────────────────┐
│  EVALUATION MODE: Human vs LLM                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Step 1: Person A's Prompt                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ "I need to fly from Houston to NYC on Nov 20.          │  │
│  │  I prefer cheap flights but not too early."            │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  Step 2: Person A Ranks Top 5                                │
│  ┌──────────────────────────────────────────────┐           │
│  │ 1. United UA1234 - $250 (direct)              │           │
│  │ 2. Delta DL5678 - $189 (1 stop)               │           │
│  │ 3. American AA9012 - $165 (1 stop)            │           │
│  │ 4. United UA3456 - $245 (direct)              │           │
│  │ 5. Delta DL7890 - $155 (1 stop)               │           │
│  └──────────────────────────────────────────────┘           │
│  [Submit Person A's Rankings]                                │
│                                                               │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                                               │
│  Step 3: Person B Recommends (Hidden from Person A)          │
│  Prompt shown to Person B: [Same prompt as above]            │
│  Person B picks top 5 flights they think Person A wants      │
│  [Person B submits rankings]                                 │
│                                                               │
│  Step 4: LISTEN-U Recommends (Running...)                    │
│  [Spinner: Running LISTEN-U 25 iterations...]                │
│                                                               │
│  Step 5: LILO Recommends (Running...)                        │
│  [LILO interactive refinement with Person A...]              │
│                                                               │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                                               │
│  Step 6: Team Draft - Person A Evaluates                     │
│  Which recommendation set is better?                          │
│                                                               │
│  Person B vs LISTEN-U:                                        │
│  ┌─────────────────────────────────────────────┐            │
│  │ [Person B #1] vs [LISTEN-U #1]              │ A picks → │
│  │ [Person B #2] vs [LISTEN-U #2]              │ B picks → │
│  │ [Person B #3] vs [LISTEN-U #3]              │ A picks → │
│  │ [Person B #4] vs [LISTEN-U #4]              │ A picks → │
│  │ [Person B #5] vs [LISTEN-U #5]              │ B picks → │
│  └─────────────────────────────────────────────┘            │
│  Winner: Person B (3-2)                                      │
│                                                               │
│  Person B vs LILO:                                           │
│  [Repeat team draft...]                                      │
│                                                               │
│  LISTEN-U vs LILO:                                           │
│  [Repeat team draft...]                                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Questions for Evaluation Interface

- [ ] **How many flights (k) for evaluation?** Top 5? Top 10?
- [ ] **Should Person B see Person A's interaction data?** (Which flights A clicked/viewed?)
- [ ] **Team Draft presentation**: Side-by-side? Or show flight details fully?
- [ ] **Multiple evaluations**: Can same Person A prompt be used with different Person B users?
- [ ] **LILO refinement in evaluation**: Does Person A do LILO refinement BEFORE ranking, or is LILO separate?

---

## Part 5: Database Schema Updates

### New Tables

#### 5.1 `lilo_sessions` - Track LILO refinement rounds
```sql
CREATE TABLE lilo_sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    search_id INT,  -- Links to searches table
    user_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    num_rounds INT DEFAULT 0,
    final_utility_function JSON,  -- Learned weights
    FOREIGN KEY (search_id) REFERENCES searches(id)
);
```

#### 5.2 `lilo_rounds` - Each refinement round
```sql
CREATE TABLE lilo_rounds (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(36),
    round_number INT,
    flights_shown JSON,  -- List of flight IDs shown this round
    user_rankings JSON,  -- User's top-k ranking for this round
    user_feedback TEXT,  -- Natural language feedback
    extracted_preferences JSON,  -- LLM-extracted preference signals
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES lilo_sessions(session_id)
);
```

#### 5.3 `interaction_events` - Offline data collection
```sql
CREATE TABLE interaction_events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(36),
    search_id INT,
    event_type VARCHAR(50),  -- 'flight_view', 'flight_click', 'hover', 'filter', 'sort'
    event_data JSON,  -- Flexible: flight_id, timestamp, scroll_depth, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_search (search_id),
    INDEX idx_event_type (event_type)
);
```

#### 5.4 `evaluation_sessions` - Human vs LLM experiments
```sql
CREATE TABLE evaluation_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(36) UNIQUE,
    person_a_user_id VARCHAR(100),  -- Ground truth provider
    person_a_prompt TEXT,
    person_a_rankings JSON,  -- Ground truth top-k
    person_b_user_id VARCHAR(100),  -- Human recommender
    person_b_rankings JSON,  -- Person B's guesses
    listen_u_rankings JSON,  -- LISTEN-U recommendations
    lilo_rankings JSON,  -- LILO recommendations
    team_draft_results JSON,  -- Who won each comparison
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 5.5 `flight_engagement` - Aggregated engagement per flight
```sql
CREATE TABLE flight_engagement (
    id INT AUTO_INCREMENT PRIMARY KEY,
    search_id INT,
    flight_id VARCHAR(100),
    view_count INT DEFAULT 0,
    click_count INT DEFAULT 0,
    hover_count INT DEFAULT 0,
    total_viewport_time_ms INT DEFAULT 0,
    scroll_depth_max FLOAT,  -- Highest scroll depth when flight was visible
    was_ranked BOOLEAN DEFAULT FALSE,
    rank_position INT NULL,
    engagement_score FLOAT,  -- Composite score
    FOREIGN KEY (search_id) REFERENCES searches(id)
);
```

### Schema Migration

```sql
-- migrations/add_lilo_and_tracking.sql
START TRANSACTION;

CREATE TABLE lilo_sessions (...);
CREATE TABLE lilo_rounds (...);
CREATE TABLE interaction_events (...);
CREATE TABLE evaluation_sessions (...);
CREATE TABLE flight_engagement (...);

-- Add indexes for performance
CREATE INDEX idx_lilo_search ON lilo_sessions(search_id);
CREATE INDEX idx_lilo_round ON lilo_rounds(session_id, round_number);
CREATE INDEX idx_interaction_session ON interaction_events(session_id);
CREATE INDEX idx_flight_engagement ON flight_engagement(search_id, flight_id);

COMMIT;
```

---

## Part 6: File Structure (New Branch)

```
flight_app/
├── main                           # Branch: Old Streamlit version (preserved)
└── feature/lilo-implementation    # Branch: New FastAPI + LILO version
    ├── backend_api/               # NEW: FastAPI backend
    │   ├── main.py                # Uvicorn entrypoint
    │   ├── routers/
    │   │   ├── search.py          # /api/search endpoints
    │   │   ├── ranking.py         # /api/rank/* endpoints
    │   │   ├── lilo.py            # /api/rank/lilo/* endpoints
    │   │   ├── evaluation.py      # /api/evaluation/* endpoints
    │   │   └── tracking.py        # /api/tracking/* endpoints
    │   ├── models/
    │   │   ├── requests.py        # Pydantic request models
    │   │   └── responses.py       # Pydantic response models
    │   └── dependencies.py        # Shared dependencies (DB, auth, etc.)
    │
    ├── backend/                   # EXISTING: Shared backend logic
    │   ├── amadeus_client.py
    │   ├── prompt_parser.py
    │   ├── listen_main_wrapper.py
    │   ├── lilo/                  # NEW: LILO implementation
    │   │   ├── __init__.py
    │   │   ├── lilo_optimizer.py  # Main LILO algorithm
    │   │   ├── preference_extractor.py  # LLM feedback parser
    │   │   ├── bayesian_backend.py      # Bayesian optimization
    │   │   └── prompts.py         # LILO prompts from paper
    │   └── db.py
    │
    ├── frontend_react/            # NEW: React frontend
    │   ├── public/
    │   ├── src/
    │   │   ├── App.jsx
    │   │   ├── components/
    │   │   ├── hooks/
    │   │   └── utils/
    │   └── package.json
    │
    ├── LISTEN/                    # Submodule (unchanged)
    ├── migrations/                # NEW: Database migrations
    │   └── add_lilo_and_tracking.sql
    ├── README.md
    ├── LILO_IMPLEMENTATION_PLAN.md  # This file
    └── requirements.txt           # Add: fastapi, uvicorn, pydantic, bayesian-optimization
```

---

## Part 7: Research Questions & Hypotheses

### Online Data (User Experiments)

**Hypothesis 1: LISTEN/LILO helps discovery**
- **Metric**: Do users select flights they didn't initially view/click?
- **Measurement**: Compare `user_rankings` with `interaction_events.flight_view`
- **Expected**: LISTEN/LILO surfaces flights users wouldn't find by simple sort

**Hypothesis 2: LISTEN/LILO outperforms humans**
- **Metric**: Person B recommendations vs LISTEN-U recommendations vs LILO
- **Measurement**: NDCG@k, Precision@k, Team Draft win rate
- **Expected**: LILO > LISTEN-U > Person B (for personalized preferences)

**Hypothesis 3: Additional feedback improves LILO**
- **Metric**: LILO (with refinement) vs LISTEN-U (initial prompt only)
- **Measurement**: NDCG@k comparing final rankings to Person A ground truth
- **Expected**: LILO outperforms LISTEN-U due to interactive refinement

### Offline Data (Benchmark Dataset)

**Dataset Components:**
1. **Ground truth rankings**: Person A's top-k for each prompt
2. **Interaction logs**: What Person A viewed/clicked/hovered
3. **Natural language feedback**: Person A's explanations during LILO rounds
4. **Flight features**: Price, duration, stops, times, airlines

**Potential Uses for Researchers:**
- **Ranking algorithms**: Test new preference learning methods
- **LLM prompting**: Experiment with different feedback extraction prompts
- **UI/UX research**: Analyze scroll depth, click patterns, filter usage
- **Personalization**: Build models to predict user preferences

**Dataset Format (CSV export):**
```csv
search_id,user_id,prompt,flight_id,price,duration,stops,algorithm,algorithm_rank,user_rank,was_viewed,was_clicked,viewport_time_ms
123,userA,"cheap NYC flight",f1,250,180,0,cheapest,1,2,TRUE,TRUE,3200
123,userA,"cheap NYC flight",f2,189,235,1,listen-u,1,1,TRUE,TRUE,5800
...
```

---

## Part 8: Implementation Phases (Proposed Timeline)

### Phase 0: Planning & Questions (Current)
- [ ] Review this document
- [ ] Answer all questions marked with [ ]
- [ ] Read LILO paper Appendix for exact prompts
- [ ] Decide on architecture details

### Phase 1: LILO Core Algorithm (Week 1-2)
- [ ] Implement `lilo_optimizer.py` with Bayesian optimization
- [ ] Implement `preference_extractor.py` with Gemini LLM
- [ ] Extract and implement prompts from paper
- [ ] Test LILO with sample flight data
- [ ] Validate against paper's experiments

### Phase 2: FastAPI Backend (Week 2-3)
- [ ] Create FastAPI app structure
- [ ] Implement all `/api/*` endpoints
- [ ] Migrate existing backend modules (Amadeus, LISTEN)
- [ ] Integrate LILO endpoints
- [ ] Add tracking endpoints
- [ ] Test with Postman/curl

### Phase 3: Database Schema (Week 3)
- [ ] Create migration SQL file
- [ ] Test migration on local MySQL
- [ ] Apply to Railway MySQL
- [ ] Implement new database queries in `db.py`

### Phase 4: React Frontend (Week 4-6)
- [ ] Set up React app with Vite/Create React App
- [ ] Implement flight search UI
- [ ] Implement LILO refinement rounds UI
- [ ] Implement human vs LLM evaluation UI
- [ ] Add interaction tracking (scroll, click, hover)
- [ ] Style with Tailwind CSS

### Phase 5: Integration & Testing (Week 6-7)
- [ ] Connect React frontend to FastAPI backend
- [ ] End-to-end testing of LILO workflow
- [ ] End-to-end testing of evaluation workflow
- [ ] Verify all tracking data is saved correctly
- [ ] Performance testing (LILO speed, API latency)

### Phase 6: Deployment (Week 7-8)
- [ ] Deploy FastAPI backend to Railway/Render
- [ ] Deploy React frontend to Vercel/Netlify
- [ ] Configure CORS, environment variables
- [ ] Test production deployment
- [ ] Run pilot user study

---

## Key Questions Summary (Please Answer)

### LILO Algorithm
1. How many refinement rounds? (Paper default?)
2. How many flights to show per round?
3. How many flights should user rank per round?
4. Should I extract exact prompts from paper PDF Appendix, or do you have them?
5. Which Bayesian optimization library? (GPyOpt? scikit-optimize? Custom?)

### Architecture
6. Timeline for FastAPI migration? (Immediate? After LILO works in Streamlit?)
7. Frontend framework preference? (React recommended, but open to others)
8. Keep Streamlit version in main branch while developing feature branch?
9. Deployment platforms? (Railway backend + Vercel frontend?)

### Evaluation
10. Top-k size for evaluation? (5? 10?)
11. Should Person B see Person A's interaction data, or just the prompt?
12. LILO in evaluation: Does Person A do LILO refinement separately, or is it automatic?

### Data Collection
13. Priority events for offline tracking? (Click > Scroll > Hover?)
14. Real-time event sending or batched?
15. Privacy/IRB considerations for tracking user behavior?

### Timeline
16. What's the deadline/timeline for this entire project?
17. Should we do phased rollout (LILO in Streamlit first, then migrate to FastAPI)?

---

## Next Steps

Once you answer the questions above, I can:
1. Read the LILO paper Appendix for exact prompts
2. Start implementing `backend/lilo/` module
3. Create FastAPI backend structure
4. Build React frontend prototype
5. Set up new database tables

Let me know your preferences and I'll proceed with implementation!