"""
LILO Integration for Streamlit Flight App
Uses language_bo_code BOAgenticOptimizer for proper LILO implementation
"""
import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# LILO imports from language_bo_code
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'language_bo_code'))

from language_bo_code.bo_loop import BOAgenticOptimizer
from language_bo_code.environments import SimulEnvironment
from language_bo_code.llm_utils import LLMClient
from language_bo_code.utility_approximator import get_questions
from omegaconf import DictConfig, OmegaConf
import torch
from botorch.models.transforms.input import Normalize


class FlightEnvironment(SimulEnvironment):
    """
    Environment for flight optimization using LILO.

    For flights, x and y are the same - we directly observe flight properties.
    x (parameters) = [price, duration, stops, departure_hour, arrival_hour]
    y (outcomes) = same as x (we observe these directly)

    LILO learns user preferences over these combinations via language feedback.
    """

    def __init__(self, cfg: DictConfig, all_flights: List[Dict]):
        """
        Args:
            cfg: LILO config
            all_flights: List of all available flight dictionaries
        """
        self.all_flights = all_flights

        # Define parameter/outcome names
        self.x_names = [
            "price",
            "duration_min",
            "stops",
            "departure_hour",
            "arrival_hour"
        ]

        # For flights, y = x (we observe flight properties directly)
        self.y_names = self.x_names.copy()

        self.n_var = len(self.x_names)
        self.n_obj = len(self.y_names)

        # Set config with environment-specific parameters
        default_cfg = {
            "n_var": self.n_var,
            "n_obj": self.n_obj,
            "outcome_func": "flight_search",
        }
        default_cfg.update(cfg)
        self.cfg = DictConfig(default_cfg)

        self.contributions_available = False
        self.utility_func = None  # User utility is learned via LILO

        # Normalize flight data
        self._normalize_flights()

    def _normalize_flights(self):
        """Extract and normalize flight parameters."""
        self.flight_params = []

        for flight in self.all_flights:
            # Extract parameters
            params = {
                "price": flight.get("price", 0),
                "duration_min": flight.get("duration_min", 0),
                "stops": flight.get("stops", 0),
                "departure_hour": self._extract_hour(flight.get("departure_time", "00:00")),
                "arrival_hour": self._extract_hour(flight.get("arrival_time", "00:00"))
            }
            self.flight_params.append(params)

        # Get min/max for normalization
        df = pd.DataFrame(self.flight_params)
        self.param_mins = df.min().to_dict()
        self.param_maxs = df.max().to_dict()

    def _extract_hour(self, time_str: str) -> float:
        """Extract hour from time string like '14:30:00' or '2:30 PM'."""
        try:
            if 'T' in time_str:  # ISO format
                time_str = time_str.split('T')[1].split('.')[0]

            parts = time_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            return hour + minute / 60.0
        except:
            return 0.0

    def get_outcomes(self, x: np.ndarray) -> np.ndarray:
        """
        For flights, y = x (we observe flight properties directly).

        Args:
            x: Array of shape (n, d) with normalized flight parameters

        Returns:
            y: Same as x (observed flight outcomes)
        """
        return x.copy()

    def get_random_x(self, seed: int, N: int = 1) -> np.ndarray:
        """
        Sample N random flights from available flights.

        Returns normalized parameters.
        """
        np.random.seed(seed)
        indices = np.random.choice(len(self.all_flights), size=N, replace=False)

        x = np.zeros((N, self.n_var))
        for i, idx in enumerate(indices):
            params = self.flight_params[idx]
            # Normalize to [0, 1]
            for j, name in enumerate(self.x_names):
                min_val = self.param_mins[name]
                max_val = self.param_maxs[name]
                if max_val > min_val:
                    x[i, j] = (params[name] - min_val) / (max_val - min_val)
                else:
                    x[i, j] = 0.0

        return x

    def get_problem_bounds(self) -> List[List[float]]:
        """Bounds for optimization (all parameters in [0, 1] after normalization)."""
        return [[0.0] * self.n_var, [1.0] * self.n_var]

    def get_goal_message(self) -> str:
        """Returns a message describing the flight search goal."""
        return (
            "Find the best flight option that balances price, duration, "
            "number of stops, and convenient departure/arrival times based on your preferences."
        )

    def get_input_transform(self) -> Normalize:
        """Returns botorch input transform for normalization."""
        bounds = torch.stack([torch.zeros(self.n_var), torch.ones(self.n_var)])
        return Normalize(d=self.n_var, bounds=bounds)

    def denormalize_params(self, x_normalized: np.ndarray) -> Dict:
        """
        Convert normalized parameters back to original scale.

        Args:
            x_normalized: Single row of normalized parameters

        Returns:
            Dictionary with denormalized flight parameters
        """
        params = {}
        for i, name in enumerate(self.x_names):
            min_val = self.param_mins[name]
            max_val = self.param_maxs[name]
            params[name] = x_normalized[i] * (max_val - min_val) + min_val
        return params

    def find_matching_flight(self, x_normalized: np.ndarray) -> Optional[Dict]:
        """
        Find the actual flight that best matches these parameters.

        Args:
            x_normalized: Single row of normalized parameters

        Returns:
            Flight dictionary from all_flights
        """
        target = self.denormalize_params(x_normalized)

        # Find closest matching flight
        best_idx = 0
        best_dist = float('inf')

        for idx, params in enumerate(self.flight_params):
            # Compute distance
            dist = sum([
                ((params[name] - target[name]) / (self.param_maxs[name] - self.param_mins[name] + 1e-6)) ** 2
                for name in self.x_names
            ])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        return self.all_flights[best_idx]


class StreamlitLILOBridge:
    """
    Bridge between LILO optimizer and Streamlit UI.
    Uses real BOAgenticOptimizer from language_bo_code.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.sessions: Dict[str, 'LILOSession'] = {}

    def create_session(
        self,
        session_id: str,
        flights_data: List[Dict],
        config_overrides: Optional[Dict] = None
    ) -> 'LILOSession':
        """
        Create a new LILO optimization session.

        Args:
            session_id: Unique identifier
            flights_data: List of flight dictionaries
            config_overrides: Optional config overrides
        """
        # Create LILO config with random seed for varied questions
        import time
        random_seed = int(time.time() * 1000) % 10000  # Use timestamp for variation
        print(f"[LILO DEBUG] Creating session with random seed: {random_seed}")

        cfg = OmegaConf.create({
            "seed": random_seed,
            "environment": {
                "seed": random_seed,  # Use same random seed for environment
                "name": "flight_search",
            },
            "N_iter": 2,  # 2 LILO iterations
            "bs_exp": 5,  # Show 5 flight options per round
            "bs_exp_init": 5,  # Show 5 options in first round
            "bs_feedback": 2,  # Ask 2 questions per round
            "save_outputs": False,
            "debug_mode": False,
            "feedback_type": "nl_qa",  # Natural language Q&A
            "uprox_prompt_type": "pairwise",  # Pairwise comparisons
            "acquisition_method": "log_nei",  # Log NEI acquisition
            "summarize_feedback": True,
            "include_goals": True,
            "init_method": "random",
            "use_prior_knowledge": False,
            "num_llm_samples": 2,
            "uprox_llm_model": "gemini/gemini-2.0-flash-exp",
            "hf_llm_model": "gemini/gemini-2.0-flash-exp",
            "api_key": self.api_key,
            "feedback_acquisition_method": "none",  # No active learning for feedback
            "pairwise_pref_model_input_type": "y"
        })

        # Apply overrides
        if config_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(config_overrides))

        # Create environment with actual flight data
        env = FlightEnvironment(cfg.environment, flights_data)

        # Create optimizer
        optimizer = BOAgenticOptimizer(env, cfg)

        # Create session
        session = LILOSession(
            session_id=session_id,
            optimizer=optimizer,
            env=env,
            current_iteration=0,
            flights_data=flights_data
        )

        self.sessions[session_id] = session
        return session

    def get_initial_questions(self, session_id: str) -> List[str]:
        """
        Get initial goal-understanding questions before showing any flights.

        This uses LILO's question generation with empty experimental data
        to ask high-level preference questions.
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        optimizer = session.optimizer

        print("[LILO DEBUG] Attempting to generate LLM questions...")
        # Generate initial goal questions (trial_index=-1 means initialization)
        questions = get_questions(
            exp_df=pd.DataFrame(),  # Empty - no experiments yet
            context_df=pd.DataFrame(),  # Empty - no feedback yet
            env=optimizer.env,
            selected_arm_index_ls=[],
            n_questions=optimizer.cfg.bs_feedback,
            llm_client=optimizer.uprox_client,
            include_goals=optimizer.cfg.include_goals,
            pre_select_data=False,
            prompt_type="pairwise"
        )

        print(f"[LILO DEBUG] LLM generated {len(questions) if questions else 0} questions")
        if questions and len(questions) > 0:
            print(f"[LILO DEBUG] First question: {questions[0][:100]}...")

        return questions

    def run_iteration(
        self,
        session_id: str,
        user_feedback: Dict[str, str]
    ) -> Tuple[List[Dict], List[str]]:
        """
        Run one LILO iteration:
        1. Record user feedback from previous round
        2. Generate new candidate flights
        3. Evaluate them
        4. Generate questions for next round

        Args:
            session_id: Session ID
            user_feedback: Dict mapping questions to answers

        Returns:
            (flights_to_show, next_questions)
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        optimizer = session.optimizer
        trial_index = session.current_iteration

        # 1. Update context with user feedback
        optimizer.update_context(
            trial_index=trial_index,
            feedback=user_feedback,
            arm_index_ls=[]  # Not doing active learning for feedback selection
        )

        # 2. Generate candidate x values (flight parameters)
        n_candidates = optimizer.bs_exp if trial_index > 0 else optimizer.bs_exp_init
        x = optimizer.acqf(seed=optimizer.cfg.seed + trial_index + 1, N=n_candidates)

        # 3. Evaluate to get y (for flights, y = x)
        y = optimizer.env.get_outcomes(x)

        # 4. Update experimental dataset
        optimizer.update_exp_data(x=x, y=y, trial_index=trial_index + 1)

        # 5. Generate questions for next round
        next_questions = get_questions(
            exp_df=optimizer.exp_df,
            context_df=optimizer.context_df,
            env=optimizer.env,
            selected_arm_index_ls=[],
            n_questions=optimizer.cfg.bs_feedback,
            llm_client=optimizer.uprox_client,
            include_goals=optimizer.cfg.include_goals,
            pre_select_data=False,
            prompt_type="pairwise"
        )

        # 6. Find actual flights matching the generated x values
        flights_to_show = []
        for i in range(len(x)):
            flight = session.env.find_matching_flight(x[i])
            if flight:
                flights_to_show.append(flight)

        session.current_iteration += 1

        return flights_to_show, next_questions

    def compute_final_rankings(
        self,
        session_id: str,
        all_flights: List[Dict]
    ) -> List[Dict]:
        """
        Compute final utility scores for all flights and return ranked list.

        Args:
            session_id: Session ID
            all_flights: All available flights to rank

        Returns:
            List of dicts: [{'flight': flight_dict, 'utility_score': float, 'rank': int}, ...]
            Sorted by utility (best first)
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        optimizer = session.optimizer
        env = session.env

        # Convert all flights to normalized x values
        all_x = []
        for flight in all_flights:
            # Extract parameters from flight
            params = {
                "price": flight.get("price", 0),
                "duration_min": flight.get("duration_min", 0),
                "stops": flight.get("stops", 0),
                "departure_hour": env._extract_hour(flight.get("departure_time", "00:00")),
                "arrival_hour": env._extract_hour(flight.get("arrival_time", "00:00"))
            }

            # Normalize to [0, 1]
            x = np.zeros(env.n_var)
            for j, name in enumerate(env.x_names):
                min_val = env.param_mins[name]
                max_val = env.param_maxs[name]
                if max_val > min_val:
                    x[j] = (params[name] - min_val) / (max_val - min_val)
                else:
                    x[j] = 0.0

            all_x.append(x)

        all_x = np.array(all_x)

        # Get utility scores from LILO model
        # The label_exp_df contains p_accept_mean which is the utility
        if optimizer.label_exp_df is not None and 'p_accept_mean' in optimizer.label_exp_df.columns:
            # Use the trained preference model to predict utilities
            from language_bo_code.utility_approximator import get_pairwise_llm_proxy_utilities

            # Create a temporary dataframe with all flights
            temp_df = pd.DataFrame()
            for i, name in enumerate(env.x_names):
                temp_df[name] = all_x[:, i]

            # Get outcomes (for flights, y = x)
            all_y = env.get_outcomes(all_x)
            for i, name in enumerate(env.y_names):
                temp_df[name] = all_y[:, i]

            # Get utility predictions using the same method as LILO
            try:
                labeled_df, _ = get_pairwise_llm_proxy_utilities(
                    to_label_df=temp_df,
                    context_df=optimizer.context_df,
                    env=optimizer.env,
                    llm_client=optimizer.uprox_client,
                    include_goals=optimizer.cfg.include_goals,
                    num_responses=optimizer.cfg.num_llm_samples,
                    pref_model=optimizer.pref_model,
                    pref_model_input_type="y",
                    summarize_feedback=False
                )

                utilities = labeled_df['p_accept_mean'].values
            except Exception as e:
                print(f"[LILO] Error computing utilities with model: {e}")
                # Fallback: use simple scoring based on last iteration's best
                utilities = -np.sum((all_x - all_x.mean(axis=0))**2, axis=1)
        else:
            # Fallback: use simple scoring
            utilities = -np.sum((all_x - all_x.mean(axis=0))**2, axis=1)

        # Create ranked list
        ranked_flights = []
        for i, flight in enumerate(all_flights):
            ranked_flights.append({
                'flight': flight,
                'utility_score': float(utilities[i]),
                'rank': 0  # Will be set after sorting
            })

        # Sort by utility (descending)
        ranked_flights.sort(key=lambda x: x['utility_score'], reverse=True)

        # Assign ranks
        for rank, item in enumerate(ranked_flights, 1):
            item['rank'] = rank

        return ranked_flights


@dataclass
class LILOSession:
    """Tracks a LILO optimization session."""
    session_id: str
    optimizer: BOAgenticOptimizer
    env: FlightEnvironment
    current_iteration: int
    flights_data: List[Dict]
