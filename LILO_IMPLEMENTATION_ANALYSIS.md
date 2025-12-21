# LILO Implementation Analysis

## ⚠️ Critical Finding: NOT Using `language_bo_code`

**I created a custom implementation (`backend/lilo_flight_optimizer.py`) instead of using the existing `language_bo_code/` folder.**

---

## What's in `language_bo_code/`?

The `language_bo_code` folder contains a **full Bayesian Optimization framework** for synthetic multi-objective optimization:

### Key Components:
- **`bo_loop.py`**: Main BO loop with Gaussian Process models, acquisition functions (EI, NEI, EUBO, etc.)
- **`utility_approximator.py`**: LLM-based utility estimation using prompts
- **`gp_models.py`**: Gaussian Process surrogate models
- **`prompts.py`**: Prompts for synthetic environments (DTLZ2, OSY, etc.)
- **`environments.py`**: Synthetic test environments (CarCab, VehicleSafety, Thermo, etc.)
- **`llm_utils.py`**: LLM API calling utilities

### What it does:
- **Continuous optimization**: Searches over continuous parameter spaces (not discrete choices)
- **GP surrogate models**: Builds Gaussian Process models of the utility function
- **Acquisition functions**: Uses sophisticated BO acquisition functions (Expected Improvement, etc.)
- **Synthetic environments**: Designed for benchmark problems, not real-world applications

### Why I didn't use it:
1. **Different problem type**: It's for continuous multi-objective optimization, flights are discrete choices
2. **No flight-specific code**: All prompts are for synthetic utility functions (piecewise linear, thresholds, slopes)
3. **Complex dependencies**: Requires BoTorch, GPyTorch, Hydra config system

---

## What I Actually Implemented

### File: `backend/lilo_flight_optimizer.py`

A **simplified adapter** that:
1. Takes the LILO paper's core idea (LLM-based utility learning)
2. Adapts it for **discrete flight selection**
3. Uses **Gemini 2.0 Flash** instead of the language_bo_code's LLM client
4. Implements only the **essential LILO prompts** from the paper

---

## Complete LILO Flow (Current Implementation)

### 📍 Position in Workflow
```
User enters with access token
    ↓
Regular Flight Rankings (unranked mode)
    ↓ (after CSV download)
🎯 LILO PREFERENCE LEARNING (2 rounds)
    ↓ (after final recommendations)
Cross-Validation Rankings
    ↓
Survey
    ↓
Completion token generated
```

---

## Round-by-Round Breakdown

### 🔵 Round 1: Initial Preference Collection

#### Step 1: Candidate Selection
**Function**: `select_candidates(all_flights, round_num=1, n_candidates=15)`

**Strategy for Round 1**: **Diversity-based (NOT utility-based)**
```python
def select_candidates(self, all_flights, round_num=1, n_candidates=15):
    if round_num == 1:
        # DIVERSITY STRATEGY
        cheapest = sorted(all_flights, key=lambda x: x.get('price', 999999))[:5]
        fastest = sorted(all_flights, key=lambda x: x.get('duration_min', 999999))[:5]
        remaining = [f for f in all_flights if f not in cheapest and f not in fastest]
        random.shuffle(remaining)
        candidates = cheapest + fastest + remaining[:max(0, n_candidates-10)]
        return candidates[:n_candidates]
```

**Result**: 15 flights shown
- 5 cheapest flights
- 5 fastest flights
- 5 random flights from the remaining pool

#### Step 2: User Selection & Ranking
- User selects **exactly 5 flights** from the 15 shown
- User **drags to rank** them (1 = best, 5 = worst)

#### Step 3: User Feedback
- User provides **free-text feedback** about preferences
- Example: "I prioritize price over speed, hate layovers, prefer afternoon arrivals"

#### Step 4: LLM Question Generation
**Function**: `generate_questions(flights_shown, user_rankings, round_num=1, n_questions=3)`

**Uses**: **Prompt 2 from LILO paper** (adapted for flights)

**Prompt structure**:
```
You are an expert in determining whether a human decision maker (DM)
is going to be satisfied with a set of flight options.

## Experimental outcomes:
So far, we have obtained the following flight options:
[15 flights formatted as arm_0, arm_1, ... arm_14]

## Human feedback messages:
The decision maker has indicated interest in these flights: [user's top 5]

## Your task:
Generate 3 questions to better understand the DM's utility function.

Questions can be:
- Clarifying optimization objective (price vs time vs convenience)
- Ranking two or more flights
- How to improve certain flights
- Rating specific flights
- About their priorities (price, duration, stops, airlines, times)

Return as JSON: {"q1": "...", "q2": "...", "q3": "..."}
```

**Result**: 3 questions stored for Round 2
- Example: "What is more important to you: saving $50 or saving 1 hour of travel time?"

#### Step 5: Feedback Summarization
**Function**: `summarize_feedback(flights_shown)`

**Uses**: **Prompt 5 from LILO paper**

**Prompt structure**:
```
## Experimental outcomes:
[All flights shown so far]

## Human feedback messages:
Round 1:
User ranked: [indices of top 5]
Feedback: [user's free text]

## Your task:
Summarize the human feedback messages into a clear description
of the DM's optimization goals.

Make your summary as quantitative as possible so that it can be
easily used for utility estimation.

Return as JSON: {"summary": "..."}
```

**Result**: Summary like "User prioritizes flights under $300, prefers direct flights, values afternoon arrivals"

#### Step 6: Database Save
- Create `LILOSession` record
- Create `LILORound` record with:
  - `round_number=1`
  - `flights_shown=[0,1,2,...,14]` (indices)
  - `user_rankings=[3,7,1,9,12]` (user's top 5 in ranked order)
  - `user_feedback="I prioritize..."`
  - `generated_questions=["Q1?", "Q2?", "Q3?"]`

---

### 🟢 Round 2: Refinement & Question Answering

#### Step 1: Question Answering
- User answers the **3 questions** generated in Round 1
- Each answer is a text area input
- Example answers:
  - Q1: "Saving time is more important"
  - Q2: "I'd rank Flight A > Flight B because..."
  - Q3: "I prefer Delta over United"

#### Step 2: Utility-Based Candidate Selection
**Function**: `select_candidates(all_flights, round_num=2, n_candidates=15)`

**THIS IS THE KEY PART - Round 2 uses utility-based selection!**

```python
def select_candidates(self, all_flights, round_num=1, n_candidates=15):
    if round_num == 1:
        # [diversity strategy from above]
    else:
        # UTILITY-BASED STRATEGY (Round 2+)
        utilities = self.estimate_utilities(all_flights)  # ← LLM estimates utilities!

        flights_with_utility = list(zip(all_flights, utilities))
        flights_with_utility.sort(key=lambda x: x[1], reverse=True)  # Sort by utility

        return [f for f, u in flights_with_utility[:n_candidates]]  # Top 15
```

#### Step 2a: Utility Estimation (for candidate selection)
**Function**: `estimate_utilities(all_flights)`

**Uses**: **Prompt 4 from LILO paper**

**Prompt structure**:
```
You are an expert in determining whether a human decision maker (DM)
is going to be satisfied with a set of flight options.

## Experimental outcomes:
[ALL flights - could be 50+ flights from original search]

## Human feedback messages:
Round 1:
User ranked: [indices]
Feedback: [text]

## Feedback Summary:
[Summarized preferences from Prompt 5]

## Your task:
Predict the probability of the decision maker being satisfied
with each flight option.

First, analyze the human feedback messages to understand
the DM's preferences.

Then, provide predictions for ALL flights.

Return as JSONL:
{"arm_index": "arm_0", "reasoning": "...", "p_accept": 0.85}
{"arm_index": "arm_1", "reasoning": "...", "p_accept": 0.12}
...
```

**Result**: Utility scores (0-1) for **every flight** in the original search
- Example: `[0.85, 0.12, 0.93, 0.45, ...]` for all 50+ flights
- **Top 15 by utility score** are selected as Round 2 candidates

**⚠️ Important**: This happens BEFORE the user answers Q&A questions!
- Round 2 candidates are selected based on:
  - ✅ Round 1 rankings
  - ✅ Round 1 free-text feedback
  - ✅ Feedback summary
  - ❌ NOT the Q&A answers (those come after candidate selection)

#### Step 3: User Selection & Ranking (Again)
- User sees the **top 15 utility-ranked flights**
- User selects **exactly 5** from these 15
- User **drags to rank** them

#### Step 4: Database Save
- Update `LILOSession.num_rounds = 2`
- Create `LILORound` record with:
  - `round_number=2`
  - `flights_shown=[indices of top 15 by utility]`
  - `user_rankings=[new top 5]`
  - `user_feedback=[concatenated Q&A answers]`
  - `extracted_preferences={"q1": "...", "q2": "...", "q3": "..."}`

---

### ⭐ Final Recommendations

#### Step 1: Final Utility Estimation
**Function**: `get_final_ranking(all_flights, top_k=10)`

**Uses**: `estimate_utilities()` again with **ALL accumulated feedback**

Now includes:
- Round 1 rankings + feedback
- Round 1 Q&A answers (from Round 2)
- Round 2 rankings + Q&A answers
- Feedback summary

**Prompt includes**:
```
## Human feedback messages:
Round 1:
User ranked: [...]
Feedback: [...]
Q&A: {"q1": "...", "q2": "...", "q3": "..."}

Round 2:
User ranked: [...]
Feedback: [Q&A answers]

## Feedback Summary:
[Updated summary]
```

**Result**: Final utility scores for all flights

#### Step 2: Show Top 10
- Display top 10 flights ranked by final utility
- First 3 expanded, rest collapsed
- User **does not rank or select** - just views
- User clicks **"Continue to Validation →"**

#### Step 3: Database Save
- Update `LILOSession`:
  - `completed_at = now()`
  - `final_utility_scores = [0.91, 0.87, ...]` (for all flights)
  - `feedback_summary = "..."`

---

## Key Differences from Expected LILO

### ❌ What's Missing/Different:

1. **No Gaussian Process models**
   - Paper/code uses GP surrogate models
   - I use pure LLM utility estimation (no statistical model)

2. **No acquisition functions**
   - Paper uses sophisticated BO acquisition (EI, EUBO, etc.)
   - I use simple utility ranking (greedy selection)

3. **No iterative convergence**
   - Paper suggests running until convergence
   - I hard-stop at 2 rounds

4. **Fixed candidate set**
   - Paper can generate new candidates in continuous space
   - I only re-rank the original flight search results

5. **Q&A answers not used for Round 2 candidate selection**
   - Questions generated in Round 1
   - But candidates for Round 2 selected BEFORE user answers
   - Answers only affect final ranking, not Round 2 selection
   - **This seems like a bug!**

6. **No pairwise comparisons**
   - Paper mentions pairwise preference learning
   - I only use rankings (top-k selection)

---

## Data Flow Summary

### What gets stored:
```
LILOSession:
  - session_id
  - search_id (links to initial flight search)
  - num_rounds = 2
  - completed_at
  - final_utility_scores = [utilities for all flights]
  - feedback_summary = "User prioritizes..."

LILORound (Round 1):
  - round_number = 1
  - flights_shown = [0,1,2,...,14] (indices of 15 diverse flights)
  - user_rankings = [3,7,1,9,12] (top 5 ranked)
  - user_feedback = "I prioritize price..."
  - generated_questions = ["Q1?", "Q2?", "Q3?"]

LILORound (Round 2):
  - round_number = 2
  - flights_shown = [8,3,21,9,...] (indices of top 15 by utility)
  - user_rankings = [8,21,9,45,3] (new top 5 ranked)
  - user_feedback = "Q1: ...\nQ2: ...\nQ3: ..."
  - extracted_preferences = {"q1": "...", "q2": "...", "q3": "..."}
```

---

## Round 2 Candidate Selection - Detailed Trace

### Timeline:
1. **Round 1 completes** → user submits rankings + feedback
2. **LLM calls Prompt 5** → generates feedback summary
3. **LLM calls Prompt 2** → generates 3 questions
4. **Store to database** → LILOSession + LILORound(1)
5. **User proceeds to Round 2**
6. **UI shows 3 questions** (from step 3)
7. **LLM calls Prompt 4** → estimates utilities for ALL flights using:
   - ✅ Round 1 rankings
   - ✅ Round 1 feedback text
   - ✅ Feedback summary
   - ❌ Q&A answers (not yet provided!)
8. **Select top 15 by utility** → show to user
9. **User answers questions** → provides Q&A responses
10. **User selects & ranks 5 flights** from the 15 shown
11. **User clicks "See Final Recommendations"**

### The Problem:
- **Questions are generated but answers aren't used** for Round 2 candidate selection!
- Round 2 candidates depend on Round 1 data only
- Q&A answers only feed into the **final ranking**, not Round 2

---

## Questions for Verification

1. **Should I be using `language_bo_code/` modules?**
   - If yes: How do I adapt GP-based BO for discrete flight choice?
   - If no: Is my simplified LLM-only approach acceptable?

2. **Should Q&A answers affect Round 2 candidate selection?**
   - Currently: Questions generated → Candidates selected → User answers
   - Should be: Questions generated → User answers → Candidates selected?

3. **Should there be more than 2 rounds?**
   - Run until convergence?
   - Fixed 3-4 rounds?
   - Adaptive based on uncertainty?

4. **Should I use pairwise comparisons instead of rankings?**

5. **Is showing final top 10 without collecting feedback okay?**
   - Or should user rank the final 10?
