#!/usr/bin/env python3
"""
Iterative Utility Function Optimizer for Pareto-Optimal Solutions

This script uses an LLM
client to iteratively define and refine a utility function for evaluating
solutions. It runs a configurable number of iterations of:
1. Define utility function based on metrics
2. Parse and compute utility for all solutions
3. Find highest utility solution
4. Ask LLM how to adjust utility function
5. Repeat

Author: AI Assistant
"""
import pandas as pd
import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
import os
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

DEFAULT_FORMULA = "utility = sum(weight_i * score_i) for all metrics (score_i ∈ [0,1])"


@dataclass
class UtilityFunction:
    """Represents a utility function with weights and formula"""
    weights: Dict[str, float]
    formula: str
    description: str


class IterativeUtilityOptimizer:
    """Main class for iterative utility function optimization"""

    def __init__(self, csv_path: str, llm_client, policy_guidance: str = "",
                 metric_columns: Optional[List[str]] = None,
                 initial_prompt_template: str = "", refinement_prompt_template: str = "",
                 generation_seed: Optional[int] = None,
                 non_numeric_metrics: Optional[List[str]] = None,
                 numeric_metric_columns: Optional[List[str]] = None):
        """
        Initialize the optimizer

        Args:
            csv_path: Path to the CSV file containing solution data
            llm_client: Client object with a generate_response(prompt, ...) method
            policy_guidance: Custom policy guidance for the optimization
            metric_columns: Metrics to consider (columns must exist in CSV)
            initial_prompt_template: Template string for the initial prompt
            refinement_prompt_template: Template string for the refinement prompt
            generation_seed: Seed to pass to the LLM for deterministic decoding
            non_numeric_metrics: Metrics to exclude from the utility function
            numeric_metric_columns: Metrics to include in the utility function
        """
        if llm_client is None:
            raise ValueError("llm_client is required for the utility optimizer")
        if not metric_columns:
            raise ValueError("metric_columns is required for the utility optimizer")
        if not initial_prompt_template:
            raise ValueError("utility_initial_prompt is required for the utility optimizer")
        if not refinement_prompt_template:
            raise ValueError("utility_refinement_prompt is required for the utility optimizer")

        self.llm_client = llm_client
        self.csv_path = csv_path
        self.policy_guidance = policy_guidance or ""
        self.solutions_df = None
        # Full list of metrics used for prompts and reporting
        self.metric_columns = list(metric_columns)
        self.initial_prompt_template = initial_prompt_template
        self.refinement_prompt_template = refinement_prompt_template
        self.utility_history = []
        self.best_solutions_history = []
        self.generation_seed = generation_seed
        # Optional overrides for numeric vs non-numeric metrics
        self.user_non_numeric_metrics = set(non_numeric_metrics or [])
        self.user_numeric_metrics = set(numeric_metric_columns or [])
        # Derived at data-load time
        self.numeric_metric_columns: List[str] = []
        # Normalization stats (min, max) per numeric metric
        self.metric_min_max: Dict[str, Tuple[float, float]] = {}

        if os.path.exists(self.csv_path):
            self._load_data()
        else:
            logger.warning(
                f"CSV file {self.csv_path} not found. Skipping data loading.")

    def _load_data(self):
        """Load and prepare the solution data"""
        logger.info(f"Loading data from {self.csv_path}")

        try:
            self.solutions_df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.solutions_df)} solutions")

            # Ensure metric columns list matches CSV
            self.metric_columns = [m for m in self.metric_columns if m in self.solutions_df.columns]
            if not self.metric_columns:
                raise ValueError("No metric_columns found in CSV. Provide valid metric names.")
            # Determine numeric metric columns
            self.numeric_metric_columns = self._determine_numeric_metric_columns()
            if not self.numeric_metric_columns:
                logger.warning("No numeric metrics detected after configuration; utility computation may be trivial.")

            # Coerce numeric metrics only, and drop rows with missing numeric metrics
            for col in self.numeric_metric_columns:
                self.solutions_df[col] = pd.to_numeric(self.solutions_df[col], errors='coerce')

            before_drop_count = len(self.solutions_df)
            subset_cols = self.numeric_metric_columns
            if subset_cols:
                self.solutions_df = self.solutions_df.dropna(subset=subset_cols)
            after_drop_count = len(self.solutions_df)
            if subset_cols and after_drop_count < before_drop_count:
                logger.info(
                    f"Dropped {before_drop_count - after_drop_count} rows with missing values in numeric metrics {subset_cols}")

            #TODO: this is data-specific, should not be handled here
            if 'recall' in self.solutions_df.columns:
                original_count = len(self.solutions_df)
                self.solutions_df = self.solutions_df[self.solutions_df['recall'] > 0]
                filtered_count = len(self.solutions_df)
                logger.info(
                    f"Filtered out {original_count - filtered_count} rows where 'recall' <= 0 or NaN")

            # Compute normalization stats for numeric metrics
            self._compute_min_max_for_numeric_metrics()

            logger.info(
                f"Using {len(self.metric_columns)} metric columns: {self.metric_columns[:10]}...")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _render_template(self, template: str, **context) -> str:
        rendered = template
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, str(value))
        return rendered

    def _initial_prompt(self) -> str:
        return self._render_template(
            self.initial_prompt_template,
            policy_guidance=self.policy_guidance,
        )

    def _refinement_prompt(self, iteration: int, utility: UtilityFunction, best_solution: Dict[str, Any]) -> str:
        context = {
            "iteration": iteration,
            "weights": json.dumps(utility.weights, indent=2),
            "formula": utility.formula,
            "description": utility.description,
            "best_solution": json.dumps(best_solution, indent=2),
            "policy_guidance": self.policy_guidance,
        }
        return self._render_template(self.refinement_prompt_template, **context)

    def _call_llm(self, prompt: str) -> str:
        response = self.llm_client.generate_response(
            prompt,
            stop=["<END_JSON>"],
            seed=self.generation_seed
        )
        return response.strip()

    def _parse_utility_function(self, response: str, previous_utility: Optional[UtilityFunction] = None) -> UtilityFunction:
        """Parse utility function from LLM response"""
        try:
            cleaned_response = response.strip().strip('`')
            # Sanitize common invalid JSON escape from model outputs (e.g., "\\in" instead of the unicode character ∈)
            cleaned_response = cleaned_response.replace("\\in", "∈")
            # If the stop token worked, we should not see it. But if the client ignored stop,
            # defensively truncate at the first occurrence.
            end_marker = cleaned_response.find('<END_JSON>')
            if end_marker != -1:
                cleaned_response = cleaned_response[:end_marker]

            decoder = json.JSONDecoder()
            json_str = None
            for match in re.finditer(r'\{', cleaned_response):
                try:
                    obj, end = decoder.raw_decode(cleaned_response[match.start():])
                    json_candidate = cleaned_response[match.start():match.start()+end]
                    json_candidate = re.sub(r'//.*?\n', '\n', json_candidate)
                    json_candidate = re.sub(r'//.*?$', '', json_candidate, flags=re.MULTILINE)
                    # Ensure we can re-encode/decode after comment stripping
                    json.loads(json_candidate)
                    json_str = json_candidate
                    break
                except json.JSONDecodeError:
                    continue

            if not json_str:
                if previous_utility:
                    logger.warning("No JSON found in response; reusing previous utility function.")
                    return previous_utility
                raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            weights_data = data.get('weights')
            description = data.get('description')
            formula = data.get('formula')

            # Fallback: sometimes the decoder captures the inner weights object (if top-level JSON had an invalid escape)
            if weights_data is None and isinstance(data, dict):
                # Treat this dict as weights if keys overlap with metric columns
                if any(k in self.metric_columns for k in data.keys()):
                    logger.warning("Interpreting decoded object as weights payload (top-level JSON likely malformed).")
                    weights_data = data

            if weights_data is None:
                if previous_utility:
                    logger.warning("LLM response missing 'weights'; reusing previous weights.")
                    weights_data = previous_utility.weights
                else:
                    raise ValueError("Missing required fields in utility function")

            clean_weights: Dict[str, float] = {}
            has_new_weight = False
            for metric in self.metric_columns:
                source_weights = weights_data if isinstance(weights_data, dict) else {}
                if metric in source_weights:
                    try:
                        clean_weights[metric] = float(source_weights[metric])
                    except (TypeError, ValueError):
                        clean_weights[metric] = 0.0
                        logger.warning("Invalid weight for metric '%s'; defaulting to 0.0", metric)
                    else:
                        has_new_weight = True
                elif previous_utility and metric in previous_utility.weights:
                    clean_weights[metric] = previous_utility.weights[metric]
                else:
                    clean_weights[metric] = 0.0

            if not has_new_weight:
                logger.warning("LLM response did not reference known metrics; reusing previous weights.")
                if previous_utility:
                    clean_weights = previous_utility.weights.copy()
                else:
                    clean_weights = {metric: 0.0 for metric in self.metric_columns}

            if description is None:
                if previous_utility:
                    logger.warning("LLM response missing 'description'; reusing previous description.")
                    description = previous_utility.description
                else:
                    logger.warning("LLM response missing 'description'; using empty description.")
                    description = ""

            if formula is None:
                if previous_utility:
                    logger.warning("LLM response missing 'formula'; reusing previous formula.")
                    formula = previous_utility.formula
                else:
                    logger.warning("LLM response missing 'formula'; using default formula.")
                    formula = DEFAULT_FORMULA

            return UtilityFunction(
                weights=clean_weights,
                formula=formula,
                description=description
            )

        except Exception as e:
            logger.error(f"Error parsing utility function: {e}")
            logger.error(f"Response was: {response}")
            raise

    def _compute_utility(self, solution: pd.Series, utility_func: UtilityFunction) -> float:
        """Compute weighted sum over min-max normalized numeric metrics.

        Notes:
        - Metrics are normalized to [0, 1] using dataset min/max.
        - Always maximize utility: negative weights naturally penalize higher values.
        """
        utility = 0.0
        for metric in self.numeric_metric_columns:
            weight = utility_func.weights.get(metric, 0.0)
            if weight == 0:
                continue
            raw_value = solution.get(metric, 0.0)
            if pd.isna(raw_value):
                raw_value = 0.0
            mn, mx = self.metric_min_max.get(metric, (None, None))
            if mn is None or mx is None or mx == mn:
                norm = 0.0
            else:
                try:
                    norm = (float(raw_value) - float(mn)) / (float(mx) - float(mn))
                except Exception:
                    norm = 0.0
            if norm < 0.0:
                norm = 0.0
            elif norm > 1.0:
                norm = 1.0
            utility += float(weight) * norm
        return utility

    def _find_best_solution(self, utility_func: UtilityFunction) -> Tuple[int, Dict[str, Any], float]:
        best_utility = float('-inf')
        best_idx = 0
        best_solution = {}
        for idx, row in self.solutions_df.iterrows():
            utility = self._compute_utility(row, utility_func)
            if utility > best_utility:
                best_utility = utility
                best_idx = idx
                # Report all metrics (including non-numeric) for visibility
                best_solution = {metric: row[metric] for metric in self.metric_columns if metric in row.index}

        return best_idx, best_solution, best_utility

    def run_optimization(self, num_iterations: int = 25) -> Dict[str, Any]:
        logger.info(
            f"Starting iterative optimization with {num_iterations} iterations")

        results = {
            'iterations': [],
            'final_utility_function': None,
            'final_best_solution': None,
            'convergence_analysis': {}
        }

        logger.info("Defining initial utility function...")
        initial_prompt = self._initial_prompt()

        print(f"\n{'='*80}")
        print(f"INITIAL PROMPT FOR ITERATION 1")
        print(f"{'='*80}")
        print(initial_prompt)
        print(f"{'='*80}\n")

        initial_response = self._call_llm(initial_prompt)
        current_utility = self._parse_utility_function(initial_response)

        self.initial_prompt = initial_prompt
        self.initial_response = initial_response

        logger.info(f"Initial utility function: {current_utility.description}")

        for iteration in range(num_iterations):
            iter_num = iteration + 1
            logger.info(f"Starting iteration {iter_num}/{num_iterations}")

            best_idx, best_solution, best_utility = self._find_best_solution(
                current_utility)

            iteration_result = {
                'iteration': iter_num,
                'utility_function': {
                    'weights': current_utility.weights.copy(),
                    'formula': current_utility.formula,
                    'description': current_utility.description
                },
                'best_solution_idx': best_idx,
                'best_solution': best_solution.copy(),
                'best_utility': best_utility,
                'prompt_sent': self.initial_prompt if iteration == 0 else getattr(self, 'next_iteration_prompt', None),
                'llm_response': self.initial_response if iteration == 0 else getattr(self, 'next_iteration_response', None)
            }
            results['iterations'].append(iteration_result)

            logger.info(
                f"Iteration {iter_num}: Best utility = {best_utility:.4f}")

            self.utility_history.append(current_utility)
            self.best_solutions_history.append(
                (best_idx, best_solution, best_utility))

            if iteration < num_iterations - 1:
                logger.info("Refining utility function...")
                refinement_prompt = self._refinement_prompt(
                    iter_num + 1, current_utility, best_solution
                )

                print(f"\n{'='*80}")
                print(f"COMPLETE PROMPT FOR ITERATION {iter_num + 1}")
                print(f"{'='*80}")
                print(refinement_prompt)
                print(f"{'='*80}\n")

                refinement_response = self._call_llm(refinement_prompt)
                previous_utility = current_utility
                current_utility = self._parse_utility_function(
                    refinement_response, previous_utility=previous_utility)

                self.next_iteration_prompt = refinement_prompt
                self.next_iteration_response = refinement_response

                logger.info(
                    f"Refined utility function: {current_utility.description}")

                time.sleep(1)

        results['final_utility_function'] = {
            'weights': current_utility.weights,
            'formula': current_utility.formula,
            'description': current_utility.description
        }
        results['final_best_solution'] = best_solution
        results['convergence_analysis'] = self._analyze_convergence()
        # Add metric trajectories (track metrics of the best solution over iterations)
        try:
            metric_trajs: Dict[str, List[Any]] = {}
            for m in self.metric_columns:
                metric_trajs[m] = [
                    (it[1] or {}).get(m) if len(it) >= 2 else None
                    for it in self.best_solutions_history
                ]
            if isinstance(results.get('convergence_analysis'), dict):
                results['convergence_analysis']['metric_trajectories'] = metric_trajs
        except Exception:
            pass

        logger.info("Optimization completed!")
        return results

    def _analyze_convergence(self) -> Dict[str, Any]:
        if len(self.best_solutions_history) < 2:
            return {'status': 'insufficient_data'}

        utilities = [item[2] for item in self.best_solutions_history]
        best_solution_indices = [item[0]
                                 for item in self.best_solutions_history]

        last_5_indices = best_solution_indices[-5:] if len(
            best_solution_indices) >= 5 else best_solution_indices
        stabilized = len(set(last_5_indices)) == 1

        utility_improvement = utilities[-1] - \
            utilities[0] if len(utilities) > 1 else 0

        return {
            'stabilized': stabilized,
            'final_best_solution_idx': best_solution_indices[-1],
            'utility_improvement': utility_improvement,
            'utility_trajectory': utilities,
            'solution_trajectory': best_solution_indices
        }

    def _determine_numeric_metric_columns(self) -> List[str]:
        """Determine numeric metric columns using user overrides or simple auto-detection.

        Priority:
          1) If user provided numeric metrics, use their intersection with available columns.
          2) Else, exclude any user-specified non-numeric metrics from the full metric list.
          3) Else, auto-detect by attempting numeric coercion and keeping columns
             where at least one value parses to a number.
        """
        cols = [m for m in self.metric_columns if m in self.solutions_df.columns]
        if self.user_numeric_metrics:
            numeric_cols = [m for m in cols if m in self.user_numeric_metrics]
            return numeric_cols
        if self.user_non_numeric_metrics:
            numeric_cols = [m for m in cols if m not in self.user_non_numeric_metrics]
            return numeric_cols

        numeric_cols: List[str] = []
        for m in cols:
            try:
                coerced = pd.to_numeric(self.solutions_df[m], errors='coerce')
                # consider numeric if there exists at least one non-NaN numeric value
                if coerced.notna().any():
                    numeric_cols.append(m)
            except Exception:
                continue
        return numeric_cols

    def _compute_min_max_for_numeric_metrics(self) -> None:
        """Compute min and max per numeric metric for normalization."""
        self.metric_min_max = {}
        for m in self.numeric_metric_columns:
            try:
                series = pd.to_numeric(self.solutions_df[m], errors='coerce')
                mn = float(series.min(skipna=True)) if series.notna().any() else None
                mx = float(series.max(skipna=True)) if series.notna().any() else None
                self.metric_min_max[m] = (mn, mx)
            except Exception:
                self.metric_min_max[m] = (None, None)

    def save_results(self, results: Dict[str, Any], output_path: str = "optimization_results.json"):
        logger.info(f"Saving results to {output_path}")

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Results saved successfully")

    def save_best_solutions(self, results: Dict[str, Any], output_dir: str = "outputs"):
        import os
        from datetime import datetime

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for iteration_data in results['iterations']:
            iteration_num = iteration_data['iteration']
            best_solution = iteration_data['best_solution']
            best_solution_idx = iteration_data['best_solution_idx']
            best_utility = iteration_data['best_utility']
            utility_function = iteration_data['utility_function']

            filename = f"iteration_{iteration_num:02d}_best_solution_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

            solution_data = {
                "iteration": iteration_num,
                "best_solution_index": best_solution_idx,
                "best_utility": best_utility,
                "utility_function": utility_function,
                "best_solution_metrics": best_solution,
                "timestamp": timestamp
            }

            with open(filepath, 'w') as f:
                json.dump(solution_data, f, indent=2, default=str)

        logger.info(
            f"Saved {len(results['iterations'])} best solutions to {output_dir}")

        summary_filename = f"all_best_solutions_{timestamp}.json"
        summary_filepath = os.path.join(output_dir, summary_filename)

        all_solutions = {
            "experiment_timestamp": timestamp,
            "total_iterations": len(results['iterations']),
            "iterations": results['iterations']
        }

        with open(summary_filepath, 'w') as f:
            json.dump(all_solutions, f, indent=2, default=str)

        logger.info(
            f"Saved summary of all best solutions to {summary_filepath}")

    def print_summary(self, results: Dict[str, Any]):
        print("\n" + "="*80)
        print("ITERATIVE UTILITY OPTIMIZATION SUMMARY")
        print("="*80)

        final_utility = results['final_utility_function']
        final_solution = results['final_best_solution']
        convergence = results['convergence_analysis']

        print(f"\nFINAL UTILITY FUNCTION:")
        print(f"Description: {final_utility['description']}")
        print(f"Formula: {final_utility['formula']}")
        print(f"\nWeights:")
        for metric, weight in final_utility['weights'].items():
            if weight != 0:
                print(f"  {metric}: {weight}")

        print(f"\nFINAL BEST SOLUTION:")
        print(
            f"Solution Index: {convergence.get('final_best_solution_idx', 'N/A')}")
        print(f"Key Metrics:")
        for metric in ['conflicts', 'quints', 'quads', 'avg_max', 'lateness']:
            if metric in final_solution:
                print(f"  {metric}: {final_solution[metric]}")

        print(f"\nCONVERGENCE ANALYSIS:")
        print(f"Stabilized: {convergence.get('stabilized', 'N/A')}")
        print(
            f"Utility Improvement: {convergence.get('utility_improvement', 'N/A'):.4f}")

        print(f"\nTotal Iterations: {len(results['iterations'])}")
        print("="*80)

