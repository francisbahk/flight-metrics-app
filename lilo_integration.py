"""
LILO Integration for Streamlit Flight App
Replaces simulated human feedback with real user input
"""
import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# LILO imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'language_bo_code'))

from language_bo_code.bo_loop import BOAgenticOptimizer
from language_bo_code.environments import SimulEnvironment
from language_bo_code.llm_utils import LLMClient
from omegaconf import DictConfig, OmegaConf
import torch
from botorch.models.transforms.input import InputTransform, Normalize


@dataclass
class LILOSession:
    """Tracks a LILO optimization session for a user"""
    session_id: str
    optimizer: BOAgenticOptimizer
    current_trial: int
    pending_questions: List[str]
    flight_data: pd.DataFrame
    created_at: datetime


class FlightUtilityFunc:
    """Simple utility function for flight preferences"""
    def get_goal_message(self) -> str:
        return ("My goal is to find flights that balance price, duration, and convenience. "
                "I want to minimize cost and travel time while maximizing convenience.")


class FlightEnvironment(SimulEnvironment):
    """
    Custom environment for flight optimization using LILO.
    Maps flight parameters (x) to flight outcomes (y).
    """

    def __init__(self, cfg: DictConfig):
        # Flight-specific parameter names
        self.x_names = [
            "price_weight",
            "duration_weight",
            "departure_time_weight",
            "stops_weight",
            "airline_weight"
        ]

        # Flight outcome metrics
        self.y_names = [
            "total_price",
            "total_duration",
            "departure_score",
            "num_stops",
            "airline_score"
        ]

        self.n_var = len(self.x_names)
        self.n_obj = len(self.y_names)

        # Set required config parameters
        default_cfg = {
            "n_var": self.n_var,
            "n_obj": self.n_obj,
            "outcome_func": "flight_search",
        }
        default_cfg.update(cfg)
        self.cfg = DictConfig(default_cfg)

        self.contributions_available = False
        self.utility_func = FlightUtilityFunc()

    def get_outcomes(self, x: np.ndarray) -> np.ndarray:
        """
        Given preference weights (x), return flight outcomes (y).
        In practice, this would rank flights and return metrics of top flights.
        """
        # Placeholder: In real implementation, this would:
        # 1. Use weights to rank available flights
        # 2. Return metrics of the top-ranked flights
        # For now, return dummy data
        n = x.shape[0]
        y = np.random.randn(n, self.n_obj)
        return y

    def get_random_x(self, seed: int, N: int = 1) -> np.ndarray:
        """Generate random weight configurations"""
        np.random.seed(seed)
        # Random weights that sum to 1
        x = np.random.dirichlet(np.ones(self.n_var), size=N)
        return x

    def get_problem_bounds(self) -> List[List[float]]:
        """Bounds for optimization (weights between 0 and 1)"""
        return [[0.0] * self.n_var, [1.0] * self.n_var]

    def get_input_transform(self) -> InputTransform:
        """Returns botorch input transform for normalization"""
        bounds = torch.stack([torch.zeros(self.n_var), torch.ones(self.n_var)])
        return Normalize(d=self.n_var, bounds=bounds)

    def get_utility_from_y(self, y: np.ndarray) -> np.ndarray:
        """
        Compute utility from outcomes.
        For flights, lower price/duration/stops is better.
        This is the TRUE utility function we're trying to learn.
        """
        # Normalize and invert (lower is better for these metrics)
        utility = -(y[:, 0] + y[:, 1] + y[:, 3]) / 3.0
        return utility.reshape(-1, 1)

    def get_utility_gradient(self, y: np.ndarray) -> np.ndarray:
        """Gradient of utility w.r.t outcomes"""
        n = y.shape[0]
        grad = np.zeros((n, self.n_obj))
        grad[:, [0, 1, 3]] = -1/3.0
        return grad


class StreamlitLILOBridge:
    """
    Bridge between LILO optimizer and Streamlit UI.
    Manages optimization sessions and handles real user feedback.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.sessions: Dict[str, LILOSession] = {}

    def create_session(
        self,
        session_id: str,
        flights_df: pd.DataFrame,
        config_overrides: Optional[Dict] = None
    ) -> LILOSession:
        """
        Create a new LILO optimization session for a user.

        Args:
            session_id: Unique identifier for this session
            flights_df: Available flights to optimize over
            config_overrides: Optional config overrides
        """
        # Load default LILO config
        config_path = os.path.join(
            os.path.dirname(__file__),
            'language_bo_code/config/bo_loop.yaml'
        )

        # Create config
        cfg = OmegaConf.create({
            "seed": 0,
            "environment": {
                "seed": 0,
                "name": "flight_search",
                "utility_func": "user_preference"
            },
            "N_iter": 2,  # 2 rounds as requested
            "bs_exp": 3,  # Show 3 flight options per round
            "bs_exp_init": 5,  # Show 5 options in first round
            "bs_feedback": 2,  # 2 questions per round as requested
            "save_outputs": False,
            "debug_mode": True,
            "feedback_type": "nl_qa",
            "uprox_prompt_type": "pairwise_direct",
            "acquisition_method": "log_nei",
            "summarize_feedback": True,
            "include_goals": True,
            "init_method": "random",
            "use_prior_knowledge": False,
            "num_llm_samples": 2,  # Reduced for speed (was 3)
            "uprox_llm_model": "gemini-2.0-flash-exp",
            "hf_llm_model": "gemini-2.0-flash-exp",
            "api_key": self.api_key,
            "feedback_acquisition_method": "none",
            "pairwise_pref_model_input_type": "y"
        })

        # Apply overrides
        if config_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(config_overrides))

        # Create environment
        env = FlightEnvironment(cfg.environment)

        # Create optimizer
        optimizer = BOAgenticOptimizer(env, cfg)

        # Create session
        session = LILOSession(
            session_id=session_id,
            optimizer=optimizer,
            current_trial=0,
            pending_questions=[],
            flight_data=flights_df,
            created_at=datetime.now()
        )

        self.sessions[session_id] = session
        return session

    def get_initial_questions(self, session_id: str) -> List[str]:
        """
        Get initial preference questions for the user.
        This replaces get_human_answers with Streamlit UI collection.
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        optimizer = session.optimizer

        # Get initial questions (trial_index=-1 means initial goals)
        from language_bo_code.utility_approximator import get_questions

        questions = get_questions(
            exp_df=pd.DataFrame(),  # Empty for initial questions
            context_df=pd.DataFrame(),
            env=optimizer.env,
            selected_arm_index_ls=[],
            n_questions=optimizer.cfg.bs_feedback,
            llm_client=optimizer.uprox_client,
            include_goals=optimizer.cfg.include_goals,
            pre_select_data=False,
            prompt_type="pairwise" if optimizer.cfg.uprox_prompt_type != "scalar" else "scalar"
        )

        session.pending_questions = questions
        return questions

    def submit_user_answers(
        self,
        session_id: str,
        answers: List[str]
    ) -> Dict:
        """
        Submit real user answers (from Streamlit UI) and update optimizer.

        Args:
            session_id: Session identifier
            answers: User's answers to the pending questions

        Returns:
            Dict with next flight options to present
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        optimizer = session.optimizer

        # Create feedback dict from questions and answers
        feedback = dict(zip(session.pending_questions, answers))

        # Update context with real user feedback
        trial_index = session.current_trial if session.current_trial > 0 else -1
        optimizer.update_context(
            trial_index=trial_index,
            feedback=feedback,
            arm_index_ls=[]
        )

        # Clear pending questions
        session.pending_questions = []

        return {"status": "success", "feedback_recorded": len(answers)}

    def get_next_flight_options(self, session_id: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Run one iteration of LILO and get next flight options to present.

        Returns:
            Tuple of (flights_dataframe, questions_for_user)
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        optimizer = session.optimizer
        session.current_trial += 1

        # Run one LILO iteration
        # This generates candidate flights based on learned preferences
        n_exp = optimizer.bs_exp if session.current_trial > 1 else optimizer.bs_exp_init

        # Get candidate flight configurations
        x = optimizer.acqf(seed=optimizer.cfg.seed + session.current_trial, N=n_exp)

        # Get outcomes for these configurations
        y = optimizer.env.get_outcomes(x)

        # Update experimental dataset
        optimizer.update_exp_data(x=x, y=y, trial_index=session.current_trial)

        # Generate questions for this round
        from language_bo_code.utility_approximator import get_questions

        arm_index_ls = optimizer.feedback_acqf(
            n=optimizer.cfg.bs_feedback * 2,
            pref_model=optimizer.pref_model
        )

        questions = get_questions(
            exp_df=optimizer.exp_df,
            context_df=optimizer.context_df,
            env=optimizer.env,
            selected_arm_index_ls=arm_index_ls,
            n_questions=optimizer.cfg.bs_feedback,
            llm_client=optimizer.uprox_client,
            include_goals=optimizer.cfg.include_goals,
            pre_select_data=True,
            prompt_type="pairwise" if optimizer.cfg.uprox_prompt_type != "scalar" else "scalar"
        )

        session.pending_questions = questions

        # Return flight options
        flights_df = optimizer.exp_df[
            optimizer.exp_df.trial_index == session.current_trial
        ].copy()

        return flights_df, questions

    def get_session_status(self, session_id: str) -> Dict:
        """Get current status of optimization session"""
        session = self.sessions.get(session_id)
        if not session:
            return {"exists": False}

        return {
            "exists": True,
            "current_trial": session.current_trial,
            "pending_questions": len(session.pending_questions),
            "total_experiments": len(session.optimizer.exp_df),
            "created_at": session.created_at.isoformat()
        }
