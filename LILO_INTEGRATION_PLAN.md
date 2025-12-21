# LILO Integration Plan using `language_bo_code`

## Current Problem
I created a custom `backend/lilo_flight_optimizer.py` instead of using the existing `language_bo_code/` modules. This was wrong.

## Solution: Use language_bo_code Modules

### Key Modules to Use:

1. **`utility_approximator.py`** - Core LILO functions:
   - `get_questions()` - Generate follow-up questions (Prompt 2)
   - `get_proxy_utilities_qa()` - Estimate utilities with Q&A feedback
   - `_get_human_feedback()` - Format feedback context
   - `_summarize_feedback()` - Summarize user preferences

2. **`llm_utils.py`** - LLM client interface (abstract class)
   - Need to implement Gemini backend

3. **`prompts.py`** - Prompt templates
   - UAPPROX_QUESTIONS_SCALAR/PAIRWISE - Question generation prompts
   - UAPPROX_QA_LABEL - Utility estimation with Q&A
   - QA_SUMMARIZER - Feedback summarization

---

## Implementation Strategy

### Step 1: Create Gemini LLM Client

**File**: `backend/gemini_llm_client.py`

```python
from language_bo_code.llm_utils import LLMClient
import google.generativeai as genai

class GeminiLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp"):
        super().__init__(api_key=api_key, model=model)
        genai.configure(api_key=api_key)
        self.genai_model = genai.GenerativeModel(model)

    async def get_llm_response(self, prompt: str, num_responses: int = 1, kwargs=None):
        # Implement Gemini API calls
        responses = []
        for _ in range(num_responses):
            response = self.genai_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0.7) if kwargs else 0.7,
                    max_output_tokens=kwargs.get('max_tokens', 2048) if kwargs else 2048
                )
            )
            responses.append(response.text)
        return responses

    async def get_batch_llm_responses(self, prompts: list, num_responses: int = 1, kwargs=None):
        # Call get_llm_response for each prompt
        results = []
        for prompt in prompts:
            results.append(await self.get_llm_response(prompt, num_responses, kwargs))
        return results
```

---

### Step 2: Create Flight Environment Adapter

**File**: `backend/flight_environment.py`

The `language_bo_code` expects an environment with:
- `y_names`: List of outcome attribute names
- `x_names`: Input parameter names (not needed for discrete choice)
- `cfg`: Configuration object

For flights, we don't have continuous parameters to optimize - we have discrete flight choices.
Each flight is an "arm" with multiple attributes.

```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class FlightEnvironmentConfig:
    """Minimal config for flight environment"""
    pass

class FlightEnvironment:
    """
    Adapter to make flights work with language_bo_code utilities.
    Treats each flight as an 'arm' in a multi-armed bandit problem.
    """

    def __init__(self):
        self.cfg = FlightEnvironmentConfig()

        # Flight attributes (these are the "outcomes" y in their framework)
        self.y_names = [
            'price',           # Cost in dollars
            'duration_min',    # Flight duration in minutes
            'stops',           # Number of stops
            'departure_hour',  # Departure time (0-23)
            'arrival_hour',    # Arrival time (0-23)
        ]

        # No input parameters (x) since we're not optimizing a function
        # Just selecting from discrete options
        self.x_names = []

    def flights_to_dataframe(self, flights: list) -> pd.DataFrame:
        """
        Convert flight dictionaries to DataFrame format expected by language_bo_code.
        Each row is an 'arm' (flight option).
        """
        rows = []
        for i, flight in enumerate(flights):
            # Extract departure/arrival hour from time strings
            dept_time = flight.get('departure_time', '12:00:00')
            arr_time = flight.get('arrival_time', '12:00:00')

            dept_hour = int(dept_time.split(':')[0]) if ':' in dept_time else 12
            arr_hour = int(arr_time.split(':')[0]) if ':' in arr_time else 12

            row = {
                'arm_index': i,  # REQUIRED by language_bo_code
                'price': flight.get('price', 0),
                'duration_min': flight.get('duration_min', 0),
                'stops': flight.get('stops', 0),
                'departure_hour': dept_hour,
                'arrival_hour': arr_hour,

                # Store full flight data for later retrieval
                '_flight_data': flight
            }
            rows.append(row)

        return pd.DataFrame(rows)
```

---

### Step 3: Create LILO Flight Coordinator

**File**: `backend/lilo_coordinator.py` (replaces `lilo_flight_optimizer.py`)

```python
import pandas as pd
from typing import List, Dict
from language_bo_code.utility_approximator import (
    get_questions,
    get_proxy_utilities_qa,
    _get_human_feedback,
    _summarize_feedback
)
from .gemini_llm_client import GeminiLLMClient
from .flight_environment import FlightEnvironment

class LILOFlightCoordinator:
    """
    Coordinates LILO process using language_bo_code modules.
    Manages the interaction between flights, LLM, and LILO algorithms.
    """

    def __init__(self, gemini_api_key: str):
        self.llm_client = GeminiLLMClient(api_key=gemini_api_key)
        self.env = FlightEnvironment()

        # DataFrames tracking state
        self.exp_df = pd.DataFrame()      # All flights shown across rounds
        self.context_df = pd.DataFrame()  # Flights with user feedback/rankings

        # Round tracking
        self.round_num = 0

    def initialize_flights(self, all_flights: List[Dict]):
        """Convert all flights to DataFrame format"""
        self.all_flights_df = self.env.flights_to_dataframe(all_flights)

    def select_round1_candidates(self, n_candidates: int = 15) -> pd.DataFrame:
        """
        Round 1: Diversity-based selection (same as before)
        Returns DataFrame of selected flights
        """
        cheapest = self.all_flights_df.nsmallest(5, 'price')
        fastest = self.all_flights_df.nsmallest(5, 'duration_min')

        remaining = self.all_flights_df[
            ~self.all_flights_df['arm_index'].isin(
                pd.concat([cheapest, fastest])['arm_index']
            )
        ]

        random_sample = remaining.sample(min(5, len(remaining)))

        candidates = pd.concat([cheapest, fastest, random_sample]).head(n_candidates)
        candidates = candidates.reset_index(drop=True)

        return candidates

    def add_user_feedback(self,
                          shown_flights_df: pd.DataFrame,
                          user_rankings: List[int],  # Indices into shown_flights_df
                          user_feedback_text: str,
                          round_num: int):
        """
        Add user feedback to context_df for utility estimation.

        Args:
            shown_flights_df: The flights shown to user this round
            user_rankings: Indices of top-k flights in ranked order
            user_feedback_text: Natural language feedback
            round_num: Current round number
        """
        # Get the ranked flights
        ranked_flights = shown_flights_df.iloc[user_rankings].copy()

        # Add feedback column (required by language_bo_code)
        ranked_flights['feedback'] = user_feedback_text
        ranked_flights['round'] = round_num
        ranked_flights['rank_position'] = range(len(user_rankings))

        # Append to context (history of all feedback)
        self.context_df = pd.concat([self.context_df, ranked_flights], ignore_index=True)

        # Also add to exp_df (all shown flights)
        shown_flights_df['round'] = round_num
        self.exp_df = pd.concat([self.exp_df, shown_flights_df], ignore_index=True)

    def generate_questions(self, n_questions: int = 3) -> List[str]:
        """
        Use language_bo_code to generate follow-up questions.
        Calls get_questions() from utility_approximator.py
        """
        selected_arms = self.context_df['arm_index'].tolist()

        questions = get_questions(
            exp_df=self.exp_df,
            context_df=self.context_df,
            env=self.env,
            selected_arm_index_ls=selected_arms,
            n_questions=n_questions,
            llm_client=self.llm_client,
            include_goals=True,
            pre_select_data=True,
            prompt_type="scalar"  # Use scalar prompts (not pairwise)
        )

        return questions

    def estimate_utilities(self, flights_to_rank: pd.DataFrame) -> pd.DataFrame:
        """
        Use language_bo_code to estimate p_accept for each flight.
        Calls get_proxy_utilities_qa() from utility_approximator.py

        Returns:
            DataFrame with added 'p_accept' column
        """
        labeled_df = get_proxy_utilities_qa(
            to_label_df=flights_to_rank,
            context_df=self.context_df,
            env=self.env,
            llm_client=self.llm_client,
            num_responses=1,  # Can increase for ensemble
            include_goals=True,
            summarize_feedback=True  # Uses QA_SUMMARIZER prompt
        )

        return labeled_df

    def select_utility_ranked_candidates(self, n_candidates: int = 15) -> pd.DataFrame:
        """
        Select top N flights by estimated utility (for Round 2+)
        """
        # Estimate utilities for ALL flights
        all_with_utilities = self.estimate_utilities(self.all_flights_df)

        # Sort by p_accept (descending) and take top N
        top_candidates = all_with_utilities.nlargest(n_candidates, 'p_accept')

        return top_candidates.reset_index(drop=True)

    def get_final_ranking(self, top_k: int = 10) -> List[Dict]:
        """
        Get final top-k flights ranked by utility.
        Returns list of flight dictionaries.
        """
        final_ranked = self.estimate_utilities(self.all_flights_df)
        top_flights = final_ranked.nlargest(top_k, 'p_accept')

        # Convert back to flight dicts
        return [row['_flight_data'] for _, row in top_flights.iterrows()]
```

---

### Step 4: Update app.py Integration

**Changes to** `app.py`:

Replace the current LILO section (lines 1851-2196) with calls to the new coordinator:

```python
# Initialize LILO coordinator
if st.session_state.lilo_coordinator is None:
    from backend.lilo_coordinator import LILOFlightCoordinator
    st.session_state.lilo_coordinator = LILOFlightCoordinator(
        gemini_api_key=os.getenv('GEMINI_API_KEY')
    )
    st.session_state.lilo_coordinator.initialize_flights(
        st.session_state.get('all_flights_data', [])
    )

# Round 1
if st.session_state.lilo_round == 0:
    # Select candidates using coordinator
    if not st.session_state.lilo_round1_flights_df:
        candidates_df = st.session_state.lilo_coordinator.select_round1_candidates(n_candidates=15)
        st.session_state.lilo_round1_flights_df = candidates_df

    # ... user selects and ranks ...

    # After user submits:
    st.session_state.lilo_coordinator.add_user_feedback(
        shown_flights_df=st.session_state.lilo_round1_flights_df,
        user_rankings=user_selected_indices,
        user_feedback_text=user_feedback_text,
        round_num=1
    )

    # Generate questions
    questions = st.session_state.lilo_coordinator.generate_questions(n_questions=3)
    st.session_state.lilo_questions = questions

# Round 2
elif st.session_state.lilo_round == 1:
    # Select utility-ranked candidates
    if not st.session_state.lilo_round2_flights_df:
        candidates_df = st.session_state.lilo_coordinator.select_utility_ranked_candidates(n_candidates=15)
        st.session_state.lilo_round2_flights_df = candidates_df

    # ... user answers questions and ranks ...

    # After user submits with Q&A answers:
    combined_feedback = "\\n".join([f"Q{i+1}: {answer}" for i, answer in enumerate(qa_answers)])

    st.session_state.lilo_coordinator.add_user_feedback(
        shown_flights_df=st.session_state.lilo_round2_flights_df,
        user_rankings=user_selected_indices,
        user_feedback_text=combined_feedback,
        round_num=2
    )

# Final ranking
elif st.session_state.lilo_round == 2:
    final_flights = st.session_state.lilo_coordinator.get_final_ranking(top_k=10)
    # Display final_flights
```

---

## Key Benefits of This Approach

✅ **Uses actual language_bo_code modules** instead of reimplementing
✅ **Leverages tested prompts** from the LILO paper
✅ **Proper feedback accumulation** via context_df
✅ **Feedback summarization** using QA_SUMMARIZER prompt
✅ **Extensible** - can add GP models later if needed
✅ **Maintains compatibility** with their framework

---

## What We're NOT Using (and why it's okay)

❌ **Gaussian Process models** - Not needed for discrete choice, LLM-only is simpler
❌ **Acquisition functions** - Not needed, we use greedy utility ranking
❌ **Continuous optimization** - Flights are discrete, we just rank existing options
❌ **Synthetic environments** - We have real flights, not test functions

---

## Data Flow with New Approach

### Round 1:
1. `select_round1_candidates()` → 15 diverse flights
2. User ranks top 5, provides feedback
3. `add_user_feedback()` → stores in context_df
4. `generate_questions()` → uses `get_questions()` from language_bo_code
5. Store questions for Round 2

### Round 2:
1. `estimate_utilities(all_flights)` → uses `get_proxy_utilities_qa()`
2. `select_utility_ranked_candidates()` → top 15 by p_accept
3. Show 15 flights, user answers questions from Round 1
4. User ranks top 5 from the 15 shown
5. `add_user_feedback()` with Q&A answers as feedback text

### Final:
1. `estimate_utilities(all_flights)` → with ALL accumulated feedback
2. `get_final_ranking()` → top 10 by final p_accept
3. Display results

---

## Files to Create/Modify

### New Files:
1. `backend/gemini_llm_client.py` - Gemini implementation of LLMClient
2. `backend/flight_environment.py` - Flight environment adapter
3. `backend/lilo_coordinator.py` - Main LILO orchestrator (replaces lilo_flight_optimizer.py)

### Files to Modify:
1. `app.py` - Update LILO section (lines 1851-2196) to use coordinator
2. `backend/lilo_flight_optimizer.py` - DELETE this file

### Files to Keep:
1. `language_bo_code/` - Use as-is (no modifications)

---

## Question for User:

**Does this plan make sense?**

The key insight is that `language_bo_code` is a **framework** with:
- **Abstract LLM client** → We implement Gemini backend
- **Utility estimation functions** → We call them with flight DataFrames
- **Prompt templates** → We use their prompts (adapted via environment)
- **Data format** → We convert flights to their DataFrame format (arm_index + attributes)

We're NOT using the GP/BO parts (which are for continuous optimization), just the LLM-based utility approximation parts.

Is this the right direction?
