import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import torch

from botorch.acquisition import (
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    qExpectedUtilityOfBestOption,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.acquisition.preference import (
    AnalyticExpectedUtilityOfBestOption,
    PairwiseBayesianActiveLearningByDisagreement,
)
from botorch.models import SingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pyre_extensions import assert_is_instance
from scipy.stats import spearmanr

from .environments import (
    CarCabEnvironment,
    DTLZ2Environment,
    OSYEnvironment,
    SimulEnvironment,
    ThermoEnvironment,
    VehicleSafetyEnvironment,
)

from .gp_models import SimpleGPProxyModel

from .human_feedback_simulator import get_human_answers, get_pairwise_comparisons
from .llm_utils import LLMClient
from .prompts import CANDIDATE_SAMPLER, FULL_CANDIDATE_SAMPLER
from .utility_approximator import (
    _get_human_feedback,
    get_gp_proxy_utilities,
    get_pairwise_gp_proxy_utilities,
    get_pairwise_llm_proxy_utilities,
    get_pairwise_proxy_and_pref_models,
    get_prior_proxy_utilities,
    get_proxy_utilities_qa,
    get_questions,
)
from .utils import extract_json_from_text


class BOOptimizer:
    def __init__(self, environment: SimulEnvironment, cfg: DictConfig):
        self.env = environment
        self.cfg = cfg
        self.input_dim = len(self.env.x_names)
        self.get_random_x = self.env.get_random_x
        if cfg.bs_exp is None:
            self.bs_exp = self.env.n_var
        else:
            self.bs_exp = cfg.bs_exp
        if cfg.bs_exp_init is None:
            self.bs_exp_init = self.bs_exp
        else:
            self.bs_exp_init = cfg.bs_exp_init
        self.bs_feedback = cfg.bs_feedback
        self.proxy_model = None
        self.pref_model = None
        self.best_value = -np.inf
        self.seed = cfg.seed
        self.latest_experiment = {"x": None, "y": None}

        self.acqf = None
        self._initialize_acqf(cfg)
        self.feedback_acqf = None
        self._initialize_feedback_acqf(cfg)

        # Instantiate the context and experimental dataset
        self.context_df = pd.DataFrame()
        self.exp_df = pd.DataFrame()
        self.label_exp_df = pd.DataFrame()
        self.prior_df = pd.DataFrame()
        self.results = {"cfg": dict(cfg)}

    def _initialize_acqf(self, cfg):
        # Set the acquisition function for experiments
        if cfg.acquisition_method == "random":
            self.acqf = self._random_acqf
        elif cfg.acquisition_method == "log_ei":
            self.acqf = self._log_ei_acqf
        elif cfg.acquisition_method == "log_nei":
            self.acqf = self._log_nei_acqf
        else:
            raise ValueError(f"Unknown acquisition method: {cfg.acquisition_method}")

    def _initialize_feedback_acqf(self, cfg):
        # Set the acquisition function for human feedback
        if cfg.feedback_type != "pairwise_comp":
            if cfg.feedback_acquisition_method == "latest_experiment":
                self.feedback_acqf = self._latest_exp_acqf
            elif cfg.feedback_acquisition_method == "pred_mean":
                self.feedback_acqf = self._pred_mean_acqf
            elif cfg.feedback_acquisition_method == "pred_var":
                self.feedback_acqf = self._pred_var_acqf
            elif cfg.feedback_acquisition_method in ["eubo", "max_es"]:
                self.feedback_acqf = self._botorch_feedback_acqf
            elif cfg.feedback_acquisition_method == "none":

                def dummy_acqf(n=None, pref_model=None) -> list:
                    return []

                self.feedback_acqf = dummy_acqf
            elif cfg.feedback_acquisition_method == "random":
                self.feedback_acqf = self._random_feedback_acqf
            else:
                raise ValueError(
                    f"Unknown feedback acquisition method: {cfg.feedback_acquisition_method}"
                )
        else:
            if cfg.feedback_acquisition_method == "random":
                self.feedback_acqf = self._random_pairwise_feedback_acqf
            elif cfg.feedback_acquisition_method == "eubo":
                self.feedback_acqf = self._eubo_pairwise_feedback_acqf
            elif cfg.feedback_acquisition_method == "bald":
                self.feedback_acqf = self._bald_pairwise_feedback_acqf
            else:
                raise ValueError(
                    f"Unknown pairwise feedback acquisition method: {cfg.feedback_acquisition_method}"
                )

    def _random_acqf(self, seed: int, N: int) -> np.ndarray:
        """
        Random acquisition function that generates random hyperparameter configurations.
        Ignores the model and best value, returning purely random samples.

        Args:
            seed: Random seed for reproducibility.
            N: Number of candidates to generate.

        Returns:
            Random hyperparameter configuration as numpy array
        """
        return self.get_random_x(seed, N=N)

    def _log_ei_acqf(self, seed: int, N: int) -> np.ndarray:
        """
        Log Expected Improvement acquisition function for Bayesian optimization.
        Uses the GP model to find the next most promising point to evaluate.

        Args:
            seed: Random seed for reproducibility.
            N: Number of candidates to acquire.

        Returns:
            Next candidate point to evaluate as numpy array.
        """
        if self.proxy_model is None:
            return self.get_random_x(seed, N=N)
        else:
            bounds = self.env.get_problem_bounds()
            bounds = torch.tensor(bounds)
            if N > 1:
                qmc_sampler = SobolQMCNormalSampler(
                    sample_shape=torch.Size([256]), seed=seed
                )
                acqf = qLogExpectedImprovement(
                    model=self.proxy_model.model,
                    best_f=self.best_value,
                    sampler=qmc_sampler,
                )
                candidate, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    q=N,
                    num_restarts=10,
                    raw_samples=512,
                    sequential=False,
                )
            elif N == 1:
                acqf = LogExpectedImprovement(
                    model=self.proxy_model.model,
                    best_f=self.best_value,
                )
                candidate, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    q=N,
                    num_restarts=10,
                    raw_samples=512,
                )
            return candidate.detach().numpy()

    def _log_nei_acqf(self, seed: int, N: int) -> np.ndarray:
        """
        Log Noisy Expected Improvement acquisition function for Bayesian optimization.
        Uses the GP model to find the next most promising point to evaluate.

        Args:
            seed: Random seed for reproducibility.
            N: Number of candidates to acquire.

        Returns:
            Next candidate point to evaluate as numpy array.
        """
        if self.proxy_model is None:
            return self.get_random_x(seed, N=N)
        else:
            if self.label_exp_df.empty and not self.prior_df.empty:
                X_observed = torch.tensor(self.prior_df[self.env.x_names].values)
            elif not self.label_exp_df.empty:
                X_observed = torch.tensor(self.label_exp_df[self.env.x_names].values)
            else:
                raise ValueError("No data to compute the LogNEI acqf.")
            bounds = self.env.get_problem_bounds()
            bounds = torch.tensor(bounds)
            if self.proxy_model.multi_task:
                posterior_transform = ScalarizedPosteriorTransform(
                    weights=torch.tensor([0, 1], dtype=torch.float64)
                )  # optimize with respect to the main task
            else:
                posterior_transform = None
            if N > 1:
                qmc_sampler = SobolQMCNormalSampler(
                    sample_shape=torch.Size([256]), seed=seed
                )
                acqf = qLogNoisyExpectedImprovement(
                    model=self.proxy_model.model,
                    X_baseline=X_observed,
                    sampler=qmc_sampler,
                    posterior_transform=posterior_transform,
                )
                candidate, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    q=N,
                    num_restarts=10,
                    raw_samples=512,
                    sequential=False,
                )
            elif N == 1:
                acqf = LogNoisyExpectedImprovement(
                    model=self.proxy_model.model,
                    X_observed=X_observed,
                    posterior_transform=posterior_transform,
                )
                candidate, _ = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    q=N,
                    num_restarts=10,
                    raw_samples=512,
                )
            return candidate.detach().numpy()

    def _get_pool(self) -> pd.DataFrame:
        """
        Returns the pool of points to select from for the next batch of experiments.
        """
        # Filter exp_df to points not in context_df
        context_arms_indicies = list(self.context_df["arm_index_ls"].values)
        context_arms_indicies = [
            item for sublist in context_arms_indicies for item in sublist
        ]

        pool_df = self.label_exp_df[
            ~self.label_exp_df.arm_index.isin(context_arms_indicies)
        ]

        latest_exp_df = self.exp_df[
            self.exp_df.trial_index == self.exp_df.trial_index.max()
        ].copy()

        # Obtain labels for the latest experiment
        latest_label_exp_df, _ = self.get_label_exp_df(latest_exp_df, self.context_df)

        pool_df = pd.concat([pool_df, latest_label_exp_df], ignore_index=True)

        return pool_df

    def _latest_exp_acqf(self, n=None, pref_model=None) -> list:
        """
        Returns a batch of arm_indicies from exp_df setminus context_df to obtain feedback.
        """
        if n is None:
            n = self.bs_feedback
        latest_experiment = self.exp_df[
            self.exp_df.trial_index == self.exp_df.trial_index.max()
        ]
        return list(latest_experiment["arm_index"][:n])

    def _random_feedback_acqf(self, n=None, pref_model=None) -> list:
        if n is None:
            n = self.bs_feedback
        if self.label_exp_df.empty and self.context_df.empty:
            return self._latest_exp_acqf(n=n)
        else:
            pool_df = self._get_pool()
            selected = pool_df.sample(n=n)
        return list(selected["arm_index"].values)

    def _pred_var_acqf(self, n=None, pref_model=None) -> list:
        """
        Returns the point in exp_df setminus context_df
        with the largest variance in sampled p_accept.
        """
        if n is None:
            n = self.bs_feedback
        if self.label_exp_df.empty and self.context_df.empty:
            return self._latest_exp_acqf(n=n)

        else:
            pool_df = self._get_pool()
            selected = pool_df.sort_values(by="p_accept_var", ascending=False)[:n]
            return list(selected["arm_index"].values)

    def _pred_mean_acqf(self, n=None, pref_model=None) -> list:
        """
        Returns the point in exp_df setminus context_df
        with the largest variance in sampled p_accept.
        """
        if n is None:
            n = self.bs_feedback
        if self.label_exp_df.empty and self.context_df.empty:
            return self._latest_exp_acqf(n=n)

        else:
            pool_df = self._get_pool()
            selected = pool_df.sort_values(by="p_accept_mean", ascending=False)[:n]
            return list(selected["arm_index"].values)

    def _botorch_feedback_acqf(self, n=None, pref_model=None) -> list:
        """
        Returns a batch of points from exp_df setminus context_df to obtain feedback on
        using a botorch acquisition function.

        Args:
            botorch_acqf: The botorch acquisition function to use.
        """
        if n is None:
            n = self.bs_feedback
        if self.label_exp_df.empty and self.context_df.empty:
            return self._latest_exp_acqf(n=n)

        pool_df = self._get_pool()
        train_df = pool_df

        if self.cfg.feedback_acq_model_input_type == "y":
            if pref_model is None:
                print("Fitting Y->U model")
                pref_model = SimpleGPProxyModel(
                    input_names=self.env.y_names,
                    target_col="p_accept_mean",
                    input_transform=Normalize(d=self.env.n_obj),
                )
                pref_model.fit(train_df)
                pref_model = pref_model.model
            choices = torch.tensor(pool_df[self.env.y_names].values)
        elif self.cfg.feedback_acq_model_input_type == "x":
            if pref_model is None:
                pref_model = SimpleGPProxyModel(
                    input_names=self.env.x_names,
                    target_col="p_accept_mean",
                    input_transform=self.env.get_input_transform(),
                )
                pref_model.fit(train_df)
                pref_model = pref_model.model
            choices = torch.tensor(pool_df[self.env.x_names].values)
        else:
            raise ValueError(
                "Unknown preference model input type for feedback acquisition function."
            )

        if self.cfg.feedback_acquisition_method == "eubo":
            print("Optimizing EUBO ...")
            acqf = qExpectedUtilityOfBestOption(pref_model=pref_model)
        elif self.cfg.feedback_acquisition_method == "max_es":
            print("Optimizing Max Value Entropy Search ...")
            acqf = qMaxValueEntropy(
                model=pref_model,
                candidate_set=choices,
            )
        else:
            raise ValueError("Unknown acquisition function for feedback.")

        selected, _ = optimize_acqf_discrete(acq_function=acqf, q=n, choices=choices)

        def is_selected(x, selected, input_type="y"):
            if input_type == "x":
                cols = self.env.x_names
            elif input_type == "y":
                cols = self.env.y_names
            else:
                raise ValueError("Unknown input type for feedback acquisition.")
            diffs = np.abs(selected.numpy() - x[cols].values).sum(axis=1)
            return any(diffs < 1e-8)

        selected = train_df[
            train_df.apply(
                lambda x: is_selected(
                    x, selected, input_type=self.cfg.feedback_acq_model_input_type
                ),
                axis=1,
            )
        ]
        return list(selected.arm_index)

    def _random_pairwise_feedback_acqf(self, pref_model=None) -> list:
        arm_idx_ls = self.exp_df.arm_index.tolist()
        random_pairs = np.random.choice(
            arm_idx_ls, size=(self.bs_feedback, 2), replace=False
        )
        random_pairs = [tuple(pair) for pair in random_pairs]
        return random_pairs

    def _eubo_pairwise_feedback_acqf(self, pref_model) -> list:
        if self.label_exp_df.empty and self.context_df.empty:
            return self._random_pairwise_feedback_acqf()
        Y = torch.tensor(self.exp_df[self.env.y_names].values)
        arm_idx_ls = self.exp_df.arm_index.tolist()
        pair_idx_ls = []
        pair_data = []
        for i in range(len(arm_idx_ls)):
            for j in range(i + 1, len(arm_idx_ls)):
                pair_idx_ls.append((arm_idx_ls[i], arm_idx_ls[j]))
                pair_data.append(torch.stack([Y[[i]], Y[[j]]], dim=1))
        pair_data = torch.concatenate(pair_data, dim=0)  # [n_pairs, 2, d]
        acqf = AnalyticExpectedUtilityOfBestOption(pref_model=pref_model)
        acqf_vals = acqf(pair_data).detach().numpy()
        selected_pairs = np.argsort(acqf_vals)[::-1][: self.bs_feedback]
        selected_pairs = [pair_idx_ls[i] for i in selected_pairs]
        return selected_pairs

    def _bald_pairwise_feedback_acqf(self, pref_model) -> list:
        if self.label_exp_df.empty and self.context_df.empty:
            return self._random_pairwise_feedback_acqf()
        Y = torch.tensor(self.exp_df[self.env.y_names].values)
        arm_idx_ls = self.exp_df.arm_index.tolist()
        pair_idx_ls = []
        pair_data = []
        for i in range(len(arm_idx_ls)):
            for j in range(i + 1, len(arm_idx_ls)):
                pair_idx_ls.append((arm_idx_ls[i], arm_idx_ls[j]))
                pair_data.append(torch.stack([Y[[i]], Y[[j]]], dim=1))
        pair_data = torch.concatenate(pair_data, dim=0)  # [n_pairs, 2, d]
        acqf = PairwiseBayesianActiveLearningByDisagreement(pref_model=pref_model)
        acqf_vals = acqf(pair_data).detach().numpy()
        selected_pairs = np.argsort(acqf_vals)[::-1][: self.bs_feedback]
        selected_pairs = [pair_idx_ls[i] for i in selected_pairs]
        return selected_pairs

    def update_context(self, trial_index: int, feedback: dict, arm_index_ls: list):
        """
        Updates the context dataset with new experimental data.

        Args:
            x: The inputs to the experiment with shape (bs_feedback, d)
            y: The outcomes of the experiment with shape (bs_feedback, k)
            feedback: The list of feedbacks received from the DM for each outcome.
        """
        new_context = pd.DataFrame({"trial_index": [trial_index]}, index=[0])
        new_context["arm_index_ls"] = None
        new_context["feedback"] = None
        new_context.at[0, "arm_index_ls"] = arm_index_ls
        new_context.at[0, "feedback"] = feedback
        self.context_df = pd.concat([self.context_df, new_context]).reset_index(
            drop=True
        )

    def update_exp_data(self, x: np.ndarray, y: np.ndarray, trial_index: int):
        """
        Updates the experimental dataset with new experimental data.

        Args:
            x: The inputs to the experiment with shape (bs_feedback, d)
            y: The outcomes of the experiment with shape (bs_feedback, k)
            trial_index: The index of the trial in the BO loop.
        """
        X = pd.DataFrame(x)
        X.columns = self.env.x_names
        Y = pd.DataFrame(y)
        Y.columns = self.env.y_names
        new_exp = pd.concat([X, Y], axis=1)
        new_exp["trial_index"] = trial_index
        new_exp["arm_index"] = [f"{trial_index}_{i}" for i in range(x.shape[0])]
        columns = ["trial_index", "arm_index"] + self.env.x_names + self.env.y_names
        new_exp = new_exp[columns]
        self.exp_df = pd.concat([self.exp_df, new_exp]).reset_index(drop=True)

    def get_label_exp_df(
        self, to_label_df: pd.DataFrame, context_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[Union[SingleTaskGP, PairwiseGP]]]:
        """
        Labels the experimental dataset with the utility approximator.

        Args:
            label_df: The experimental dataset to be labeled
            context_df: The contextual dataset
        """
        label_exp_df = pd.DataFrame()
        pref_model = None
        if self.cfg.feedback_type == "true_utility_regress":
            # Label the exp data with a GP regression
            label_exp_df, pref_model = get_gp_proxy_utilities(
                to_label_df=to_label_df,
                context_df=context_df,
                exp_df=self.exp_df,
                env=self.env,
            )
        elif self.cfg.feedback_type == "pairwise_comp":
            label_exp_df, pref_model = get_pairwise_gp_proxy_utilities(
                to_label_df=to_label_df,
                context_df=context_df,
                exp_df=self.exp_df,
                env=self.env,
            )
        elif self.cfg.feedback_type == "none":
            label_exp_df = self.exp_df.copy()
            label_exp_df["true_utility"] = self.env.get_utility_from_y(
                label_exp_df[self.env.y_names].values
            )
            pref_model = None

        return label_exp_df, pref_model

    def fit_proxy_model(self):
        train_df = self.label_exp_df
        self.proxy_model = SimpleGPProxyModel(
            input_names=self.env.x_names,
            target_col="p_accept_mean",
            input_transform=self.env.get_input_transform(),
        )
        self.proxy_model.fit(train_df)

    def _run(self, trial_index: int, save_path: Optional[str] = None):
        """
        Runs the Bayesian optimization loop.

        Args:
            save_path: Optional path to save the results.
        """
        n_exp = self.bs_exp if trial_index > 1 else self.bs_exp_init
        # Sample a batch of experimental points
        x = self.acqf(seed=self.cfg.seed + trial_index + 1, N=n_exp)

        # Update the experimental dataset
        y = self.env.get_outcomes(x)
        self.latest_experiment = {"x": x, "y": y}
        self.update_exp_data(x=x, y=y, trial_index=trial_index)

        print("Experimental Dataset:")
        print(self.exp_df)

        arm_index_ls = []
        feedback = {}

        if self.cfg.feedback_type == "true_utility_regress":
            # Sample a batch of outcomes for feedback
            arm_index_ls = self.feedback_acqf(pref_model=self.pref_model)
            y_subset = (
                self.exp_df.set_index("arm_index")
                .loc[arm_index_ls, self.env.y_names]
                .values
            )
            # Get the true utility feedback
            true_utility = list(self.env.get_utility_from_y(y_subset))
            feedback = dict(zip(arm_index_ls, true_utility))
        else:
            # Sample a batch of pairwise comparisons for feedback
            arm_index_ls = self.feedback_acqf(
                pref_model=self.pref_model
            )  # list of tuples of arm indices
            comparisons = get_pairwise_comparisons(arm_index_ls, self.exp_df, self.env)
            feedback = dict(zip(arm_index_ls, comparisons))

        # Update the context dataset
        self.update_context(
            trial_index=trial_index, feedback=feedback, arm_index_ls=arm_index_ls
        )

        print("New Feedback:")
        print(feedback)

        print("Contextual Dataset:")
        print(self.context_df)

        # Label exp data with the Y -> U model
        label_exp_df, pref_model = self.get_label_exp_df(self.exp_df, self.context_df)
        self.pref_model = pref_model
        if "p_accept_mean" in label_exp_df.columns:
            best_value = label_exp_df["p_accept_mean"].max()
        else:
            best_value = None
        self.label_exp_df = label_exp_df
        self.best_value = best_value

        print("Labeled exp data:")
        print_cols = self.env.y_names + ["true_utility"]
        if "p_accept_mean" in label_exp_df.columns:
            print_cols += ["p_accept_mean"]
            print(
                "Spearman correlation: "
                + str(
                    spearmanr(
                        self.label_exp_df["true_utility"],
                        self.label_exp_df["p_accept_mean"],
                    )
                )
            )
        print(self.label_exp_df[print_cols])

        if "p_accept_mean" in label_exp_df.columns:
            # Fit the proxy GP model
            self.fit_proxy_model()

        result = {
            "trial_index": trial_index,
            "context_df": self.context_df.to_dict(),
            "label_exp_df": self.label_exp_df.to_dict(),
            "best_value": self.best_value,
        }

        self.results[trial_index] = result

        if self.cfg.save_outputs:
            with open(f"{save_path}/results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

    def run(self, save_path: Optional[str] = None):
        """
        Runs the Bayesian optimization loop.

        Args:
            save_path: Optional path to save the results.
        """
        for trial_index in range(1, self.cfg.N_iter + 1):
            if self.cfg.debug_mode:
                self._run(trial_index=trial_index, save_path=save_path)
            else:
                try:
                    self._run(trial_index=trial_index, save_path=save_path)
                except Exception as e:
                    print(f"Error: {e}")
                    return self.results

        return self.results


class BOAgenticOptimizer(BOOptimizer):
    def __init__(self, env: SimulEnvironment, cfg: DictConfig):
        super().__init__(env, cfg)
        self._initialize_acqf(cfg)
        self.pref_model = None
        self.prior_df = pd.DataFrame()

        if cfg.uprox_llm_model == cfg.hf_llm_model:
            # Create only one end point
            self.llm_client = LLMClient(
                model=cfg.uprox_llm_model,
                api_key=cfg.api_key,
            )
            self.uprox_client = self.llm_client
            self.hf_client = self.llm_client

        else:
            # Create two separate end points
            self.uprox_client = LLMClient(
                model=cfg.uprox_llm_model,
                api_key=cfg.api_key,
            )
            self.hf_client = LLMClient(
                model=cfg.hf_llm_model,
                api_key=cfg.api_key,
            )

    def _initialize_acqf(self, cfg):
        # Set the acquisition function for experiments
        try:
            super()._initialize_acqf(cfg)

        except ValueError:
            if cfg.acquisition_method == "llm_2step":
                self.acqf = self._llm_acqf
            elif cfg.acquisition_method == "llm_direct":
                self.acqf = self._llm_direct_acqf
            else:
                raise ValueError(
                    f"Unknown acquisition method: {cfg.acquisition_method}"
                )

    def _get_pool(self) -> pd.DataFrame:
        """
        Returns the pool of points to select from for the next batch of experiments.
        """
        pool_df = self.label_exp_df

        latest_exp_df = self.exp_df[
            self.exp_df.trial_index == self.exp_df.trial_index.max()
        ].copy()

        # Obtain labels for the latest experiment
        if self.cfg.uprox_prompt_type == "scalar":
            latest_exp_df = get_proxy_utilities_qa(
                to_label_df=latest_exp_df,
                context_df=self.context_df,
                env=self.env,
                llm_client=self.uprox_client,
                num_responses=self.cfg.num_llm_samples,
                include_goals=self.cfg.include_goals,
            )
        elif self.cfg.uprox_prompt_type == "pairwise":
            if self.pref_model is None:
                latest_exp_df, _ = get_pairwise_llm_proxy_utilities(
                    to_label_df=latest_exp_df,
                    context_df=self.context_df,
                    env=self.env,
                    llm_client=self.uprox_client,
                    include_goals=self.cfg.include_goals,
                    num_responses=self.cfg.num_llm_samples,
                    pref_model=None,
                    pref_model_input_type="y",
                    summarize_feedback=False,
                )
            else:
                # Use the latest pairwise GP model to label the latest experiment data points
                posterior = self.pref_model.posterior(
                    torch.tensor(latest_exp_df[self.env.y_names].values)
                )
                mean = posterior.mean.detach().squeeze().numpy()
                var = posterior.variance.detach().squeeze().numpy()
                latest_exp_df["p_accept_mean"] = mean
                latest_exp_df["p_accept_var"] = var

        elif self.cfg.uprox_prompt_type == "pairwise_direct":
            if self.proxy_model is None:
                latest_exp_df, _, _ = get_pairwise_proxy_and_pref_models(
                    to_label_df=latest_exp_df,
                    context_df=self.context_df,
                    env=self.env,
                    llm_client=self.uprox_client,
                    include_goals=self.cfg.include_goals,
                    num_responses=self.cfg.num_llm_samples,
                    pref_model=None,
                    pref_model_input_type="x",
                    summarize_feedback=False,
                )
            else:
                # Use the latest proxy model to label the latest experiment data points
                posterior = self.proxy_model.model.posterior(
                    torch.tensor(latest_exp_df[self.env.x_names].values)
                )
                posterior = assert_is_instance(posterior, GPyTorchPosterior)
                mean = posterior.mean.detach().squeeze().numpy()
                var = posterior.variance.detach().squeeze().numpy()
                latest_exp_df["p_accept_mean"] = mean
                latest_exp_df["p_accept_var"] = var

        pool_df = pd.concat([pool_df, latest_exp_df], ignore_index=True)

        return pool_df

    def _llm_acqf(self, seed: int, N: int) -> np.ndarray:
        """
        LLAMBO-style acquisition function that generates candidate hyperparameter configurations,
        via in-context LLM prompting.
        """

        if not self.cfg.use_prior_knowledge:
            prior_knowledge = ""
        else:
            prior_knowledge = self.env.get_prior_message(
                top_q=self.cfg.prior_knowledge_top_q,
                prior_type=self.cfg.prior_knowledge_type,
            )
            prior_knowledge = (
                "## Prior knowledge:\n"
                "You have obtained the following prior knowledge about the experiment:\n"
                + prior_knowledge
            )

        if self.label_exp_df.empty:
            experiment_data = ""
            ei_message = ""
        else:
            experiment_data = self.label_exp_df[
                self.env.x_names + self.env.y_names + ["p_accept_mean"]
            ].copy()
            experiment_data.rename({"p_accept_mean": "utility"}, axis=1, inplace=True)
            # Scale to [0, 1]
            X = torch.tensor(experiment_data[self.env.x_names].values)
            X = self.env.get_input_transform().transform(X)
            experiment_data[self.env.x_names] = X.detach().numpy()
            u_star = experiment_data["utility"].max()
            x_star = list(
                experiment_data[experiment_data.utility == u_star].iloc[0][
                    self.env.x_names
                ]
            )
            experiment_data = experiment_data[
                self.env.x_names + ["utility"]
            ].to_markdown(index=False)
            experiment_data = (
                "## Experiment data:\n"
                "Previous experiments have resulted in the following outcomes and estimated utilities:\n"
                + experiment_data
            )
            ei_message = f"Your candidates should maximize the expected improvement over the current best candidate x^* = {x_star} with utility u(x^*) = {u_star}."

        human_feedback = _get_human_feedback(
            self.context_df, self.env, include_goals=self.cfg.include_goals
        )

        prompt = CANDIDATE_SAMPLER.format(
            x_names=self.env.x_names,
            y_names=self.env.y_names,
            prior_knowledge=prior_knowledge,
            experiment_data=experiment_data,
            human_feedback=human_feedback,
            ei_message=ei_message,
            n_candidates=N,
            n=N - 1,
        )
        print("Generating candidates with the LLM...")
        print(prompt)
        response = None
        it = 0
        candidates = []
        while response is None and it < 3:
            response = asyncio.run(
                self.uprox_client.get_llm_response(prompt, kwargs={"max_tokens": 10000})
            )[0]
            print(response)
            candidate_dict = extract_json_from_text(response)
            if candidate_dict is None:
                print(response)
            if candidate_dict is not None and len(candidate_dict) >= N:
                candidates = []
                for x in candidate_dict.values():
                    candidates.append(np.array(x))
                candidates = candidates[:N]
            else:
                response = None
            it += 1
        if len(candidates) == 0:
            return self._random_acqf(seed, N)
        else:
            candidates = np.stack(candidates)
            candidates = torch.tensor(candidates)
            candidates = self.env.get_input_transform().untransform(candidates)
            candidates = candidates.detach().numpy()
        return candidates

    def _llm_direct_acqf(self, seed: int, N: int) -> np.ndarray:
        """
        LLAMBO-style acquisition function that generates candidate hyperparameter configurations,
        via in-context LLM prompting.
        """

        human_feedback = _get_human_feedback(
            self.context_df, self.env, include_goals=self.cfg.include_goals
        )

        if not self.cfg.use_prior_knowledge:
            prior_knowledge = ""
        else:
            prior_knowledge = self.env.get_prior_message(
                top_q=self.cfg.prior_knowledge_top_q,
                prior_type=self.cfg.prior_knowledge_type,
            )
            prior_knowledge = (
                "## Prior knowledge:\n"
                "You have obtained the following prior knowledge about the experiment:\n"
                + prior_knowledge
            )

        if self.exp_df.empty:
            experiment_data = ""
        else:
            experiment_data = self.exp_df.copy()
            # Scale to [0, 1]
            X = torch.tensor(experiment_data[self.env.x_names].values)
            X = self.env.get_input_transform().transform(X)
            experiment_data[self.env.x_names] = X.detach().numpy()
            experiment_data = experiment_data[
                ["arm_index"] + self.env.x_names + self.env.y_names
            ].to_markdown(index=False)
            experiment_data = (
                "## Experiment data:\n"
                "Previous experiments have resulted in the following outcomes:\n"
                + experiment_data
                + "\n"
            )

        prompt = FULL_CANDIDATE_SAMPLER.format(
            x_names=self.env.x_names,
            y_names=self.env.y_names,
            prior_knowledge=prior_knowledge,
            experiment_data=experiment_data,
            human_feedback=human_feedback,
            n_candidates=N,
            n=N - 1,
        )
        print("Generating candidates with the LLM...")
        print(prompt)
        response = None
        it = 0
        candidates = []
        while response is None and it < 3:
            response = asyncio.run(
                self.uprox_client.get_llm_response(prompt, kwargs={"max_tokens": 10000})
            )[0]
            candidate_dict = extract_json_from_text(response)
            if candidate_dict is None:
                print(response)
            if candidate_dict is not None and len(candidate_dict) >= N:
                candidates = []
                for x in candidate_dict.values():
                    candidates.append(np.array(x))
                candidates = candidates[:N]
            else:
                response = None
            it += 1
        if len(candidates) == 0:
            return self._random_acqf(seed, N)
        else:
            candidates = np.stack(candidates)
            candidates = torch.tensor(candidates)
            candidates = self.env.get_input_transform().untransform(candidates)
            candidates = candidates.detach().numpy()
        return candidates

    def update_context(self, trial_index: int, feedback: dict, arm_index_ls: list):
        """
        Updates the context dataset with new experimental data.

        Args:
            x: The inputs to the experiment with shape (bs_feedback, d)
            y: The outcomes of the experiment with shape (bs_feedback, k)
            feedback: The list of feedbacks received from the DM for each outcome.
        """
        new_context = pd.DataFrame({"trial_index": [trial_index]}, index=[0])
        new_context["feedback"] = None
        new_context.at[0, "feedback"] = feedback
        self.context_df = pd.concat([self.context_df, new_context]).reset_index(
            drop=True
        )

    def get_label_exp_df(
        self, to_label_df: pd.DataFrame, context_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[Union[PairwiseGP, SingleTaskGP]]]:
        """
        Labels the experimental dataset with the utility approximator.

        Args:
            label_df: The experimental dataset to be labeled
            context_df: The contextual dataset
        """
        if self.cfg.summarize_feedback and len(context_df) * self.cfg.bs_feedback >= 4:
            summarize_feedback = True
        else:
            summarize_feedback = False
        pref_model = None
        if self.cfg.uprox_prompt_type == "scalar":
            label_exp_df = get_proxy_utilities_qa(
                to_label_df=to_label_df,
                context_df=context_df,
                env=self.env,
                llm_client=self.uprox_client,
                num_responses=self.cfg.num_llm_samples,
                include_goals=self.cfg.include_goals,
                summarize_feedback=summarize_feedback,
            )
        elif self.cfg.uprox_prompt_type == "pairwise":
            label_exp_df, pref_model = get_pairwise_llm_proxy_utilities(
                to_label_df=to_label_df,
                context_df=context_df,
                env=self.env,
                llm_client=self.uprox_client,
                include_goals=self.cfg.include_goals,
                num_responses=self.cfg.num_llm_samples,
                pref_model=self.pref_model,
                pref_model_input_type="y",
                summarize_feedback=summarize_feedback,
            )
        else:
            raise ValueError("Invalid uprox_prompt_type.")
        return label_exp_df, pref_model

    def get_human_feedback(self, trial_index: int, arm_index_ls: list):
        # Generate questions with the LLM
        if self.cfg.get("uprox_prompt_type") == "scalar":
            prompt_type = "scalar"
        else:
            prompt_type = "pairwise"
        questions = get_questions(
            exp_df=self.exp_df,
            context_df=self.context_df,
            env=self.env,
            selected_arm_index_ls=arm_index_ls,
            n_questions=self.cfg.bs_feedback,
            llm_client=self.uprox_client,
            include_goals=self.cfg.include_goals,
            pre_select_data=(
                False if self.cfg.feedback_acquisition_method == "none" else True
            ),
            prompt_type=prompt_type,
        )
        print("Questions:")
        print(*questions, sep="\n")

        answers = get_human_answers(
            questions=questions,
            exp_df=self.exp_df,
            env=self.env,
            llm_client=self.hf_client,
        )
        print("Answers:")
        print(*answers, sep="\n")

        feedback = dict(zip(questions, answers))

        self.update_context(
            trial_index=trial_index, feedback=feedback, arm_index_ls=arm_index_ls
        )

        print("New Feedback:")
        print(feedback)

        print("Contextual Dataset:")
        print(self.context_df)

    def _run(self, trial_index: int, save_path: Optional[str] = None):
        """
        Runs the Bayesian optimization loop.

        Args:
            save_path: Optional path to save the results.
        """
        n_exp = self.bs_exp if trial_index > 1 else self.bs_exp_init

        if (
            self.cfg.use_prior_knowledge
            and self.cfg.prior_knowledge_inc_method == "llm_init"
            and trial_index == 1
        ) or (self.cfg.init_method == "llm" and trial_index == 1):
            x = self._llm_acqf(seed=self.cfg.seed + trial_index, N=n_exp)
        else:
            # Sample a batch of experimental points
            x = self.acqf(seed=self.cfg.seed + trial_index + 1, N=n_exp)

        if trial_index == 1:
            utilities = self.env.get_utility(x)
            print("Average utility of candidates:", utilities.mean())
            print("Best utility of candidates:", utilities.max())

        # Update the experimental dataset
        y = self.env.get_outcomes(x)
        self.latest_experiment = {"x": x, "y": y}
        self.update_exp_data(x=x, y=y, trial_index=trial_index)

        print("Experimental Dataset:")
        print(self.exp_df)

        # Sample a batch of outcomes for human evaluation
        arm_index_ls = self.feedback_acqf(
            n=self.cfg.bs_feedback * 2, pref_model=self.pref_model
        )

        # Obtain Human Feedback
        self.get_human_feedback(trial_index, arm_index_ls)

        if self.cfg.acquisition_method == "llm_direct":
            # Just compute the true utility for logging purposes
            label_exp_df = self.exp_df.copy()
            label_exp_df["p_accept_mean"] = None
            label_exp_df["true_utility"] = self.env.get_utility_from_y(
                label_exp_df[self.env.y_names].values
            ).flatten()
            self.label_exp_df = label_exp_df
            self.best_value = None

        elif (
            self.cfg.acquisition_method != "llm_direct"
            and self.cfg.uprox_prompt_type == "pairwise_direct"
        ):
            if self.cfg.pairwise_pref_model_input_type == "x":
                pref_model = (
                    self.proxy_model.model if self.proxy_model is not None else None
                )
                pref_model_input_type = "x"
            elif self.cfg.pairwise_pref_model_input_type == "y":
                pref_model = self.pref_model if self.pref_model is not None else None
                pref_model_input_type = "y"
            elif self.cfg.pairwise_pref_model_input_type == "none":
                pref_model = None
                pref_model_input_type = "none"
            else:
                raise ValueError("Invalid pairwise_pref_model_input_type.")
            # Fit the pariwise GP model as the proxy model
            label_exp_df, proxy_model, pref_model = get_pairwise_proxy_and_pref_models(
                to_label_df=self.exp_df,
                context_df=self.context_df,
                env=self.env,
                llm_client=self.uprox_client,
                include_goals=self.cfg.include_goals,
                num_responses=self.cfg.num_llm_samples,
                pref_model=pref_model,
                pref_model_input_type=pref_model_input_type,
                summarize_feedback=self.cfg.summarize_feedback,
            )
            self.proxy_model = SimpleGPProxyModel(
                input_names=self.env.x_names,
                target_col="",
                input_transform=self.env.get_input_transform(),
            )  # TO DO: should have its own class rather than cast it as a SimpleGPProxyModel
            self.proxy_model.model = proxy_model
            self.label_exp_df = label_exp_df
            self.best_value = label_exp_df["p_accept_mean"].max()
            self.pref_model = pref_model

        else:
            # Label the exp data with utility approximator
            label_exp_df, pref_model = self.get_label_exp_df(
                self.exp_df, self.context_df
            )
            self.pref_model = pref_model
            best_value = label_exp_df["p_accept_mean"].max()

            self.label_exp_df = label_exp_df
            self.best_value = best_value

            if self.cfg.acquisition_method != "llm_2step":
                # Fit the proxy GP model
                self.fit_proxy_model()

        print("Labeled exp data:")
        print(self.label_exp_df[self.env.y_names + ["true_utility", "p_accept_mean"]])
        print(
            "Spearman correlation: "
            + str(
                spearmanr(
                    self.label_exp_df["true_utility"],
                    self.label_exp_df["p_accept_mean"],
                )
            )
        )
        print("Best true value: " + str(label_exp_df["true_utility"].max()))

        result = {
            "trial_index": trial_index,
            "context_df": self.context_df.to_dict(),
            "label_exp_df": self.label_exp_df.to_dict(),
            "best_value": self.best_value,
            "prior_df": self.prior_df.to_dict(),
        }

        self.results[trial_index] = result

        if self.cfg.save_outputs:
            with open(f"{save_path}/results.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

    def run(self, save_path: Optional[str] = None):
        """
        Runs the Bayesian optimization loop.

        Args:
            save_path: Optional path to save the results.
        """
        # Obtain initial human goals
        self.get_human_feedback(trial_index=-1, arm_index_ls=[])
        super().run(save_path=save_path)
        return self.results


@hydra.main(config_path=f"./config", config_name="bo_loop")
def main(cfg: DictConfig):
    """
    Main Bayesian optimization loop for hyperparameter tuning with human feedback.
    Iteratively selects new points, evaluates them, collects feedback, and updates the model.

    Args:
        cfg: Hydra configuration containing BO parameters including acquisition method,
             number of iterations, utility weights, and save settings.
    """
    save_path = None
    if cfg.save_outputs:
        if cfg.save_name is None:
            save_name = "run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        else:
            save_name = cfg.save_name
        save_path = (
            f"{os.environ['HOME']}/local/data/language_bo/{cfg.save_dir}/{save_name}"
        )
        os.makedirs(save_path, exist_ok=True)
        print("Saving outputs to:", save_path)
        # Save the arguments
        with open(f"{save_path}/config.json", "w") as f:
            f.write(json.dumps(OmegaConf.to_container(cfg), indent=4))

    np.random.seed(cfg.seed)

    if cfg.environment.name == "dtlz2":
        env = DTLZ2Environment(cfg.environment)
    elif cfg.environment.name == "vehicle_safety":
        env = VehicleSafetyEnvironment(cfg.environment)
    elif cfg.environment.name == "carcab":
        env = CarCabEnvironment(cfg.environment)
    elif cfg.environment.name == "thermo":
        env = ThermoEnvironment(cfg.environment)
    else:
        raise ValueError(f"Unknown environment: {cfg.environment.name}")

    if cfg.feedback_type == "nl_qa":
        bo_optimizer = BOAgenticOptimizer(env, cfg)
    else:
        bo_optimizer = BOOptimizer(env, cfg)

    results = bo_optimizer.run(save_path=save_path)

    return results


if __name__ == "__main__":
    main()
