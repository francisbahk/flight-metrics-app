import pandas as pd
import json
from datetime import datetime

from LISTEN import BatchDuelingBanditOptimizer 
import numpy as np
from typing import List, Set, Optional, Dict, Any

class SolutionBatchExperimentDueling:
    """
    Solution comparison using batch dueling bandits with EUBO.
    Shows batches of solutions, collects winner, and uses EUBO for intelligent selection.
    """
    
    def __init__(
        self,
        solutions_df: pd.DataFrame,
        metric_columns: List[str],
        llm_client,
        prompt_template=None,
        batch_size: int = 4,
        n_solutions_to_explore: int = None,
        acquisition: str = "eubo",
        LLM: bool = True,
        model_type: str = "gp",
        C: float = 1.0,
        random_seed: Optional[int] = None,
        non_numeric_metrics: Optional[List[str]] = None
    ):
        """
        Args:
            solutions_df: DataFrame containing solutions
            metric_columns: Columns to use as features
        llm_client: LLM client for preference collection
        prompt_template: ComparisonPromptTemplate-compatible instance
            batch_size: Number of solutions to show per batch
            n_solutions_to_explore: Total solutions to explore (None = all)
            acquisition: Acquisition function (default: "eubo")
            C: Regularization parameter for logistic regression
            random_seed: Random seed for reproducibility
        """
        self.solutions_df = solutions_df
        self.metric_columns = metric_columns
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.LLM = LLM
        # Initialize prompt template if not provided
        if prompt_template is None:
            from prompt import ComparisonPromptAdapter
            base_prompt = (
                "You are an expert evaluator. Review the options and choose the single best one "
                "based only on the provided metrics."
            )
            self.prompt_template = ComparisonPromptAdapter(
                base_prompt=base_prompt,
                reasoning=True,
                reasoning_history=False,
                metric_columns=metric_columns,
            )
        else:
            self.prompt_template = prompt_template
        
        # Extract numeric features only (prefer YAML-provided non-numeric list)
        provided_non_numeric: Set[str] = set(non_numeric_metrics or [])
        if provided_non_numeric:
            numeric_metric_columns: List[str] = [
                col for col in metric_columns
                if col in solutions_df.columns and col not in provided_non_numeric
            ]
        else:
            numeric_metric_columns: List[str] = []
            for col in metric_columns:
                try:
                    coerced = pd.to_numeric(solutions_df[col], errors='coerce')
                    if coerced.notna().any():
                        numeric_metric_columns.append(col)
                except Exception:
                    continue
        if not numeric_metric_columns:
            raise ValueError("No numeric metrics found among provided metric_columns for comparison experiment.")
        numeric_df = solutions_df[numeric_metric_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        self.numeric_metric_columns = numeric_metric_columns
        self.features = numeric_df.values
        self.n_solutions = len(solutions_df)
        self.n_solutions_to_explore = n_solutions_to_explore or self.n_solutions
        
        # Initialize dueling bandit optimizer
        all_options = list(range(self.n_solutions))
        self.optimizer = BatchDuelingBanditOptimizer(
            all_options=all_options,
            features=self.features,
            batch_size=batch_size,
            acquisition=acquisition,
            C=C,
            random_seed=random_seed, 
            model_type=model_type
        )
        # Store seed for deterministic LLM generations (if provided)
        self.random_seed = random_seed
        
        # History tracking
        self.comparison_history = []
    
    def collect_batch_preference(self, batch_indices: List[int]) -> Dict[str, Any]:
        """
        Collect preference from LLM for a batch of schedules.
        
        Args:
            batch_indices: Indices of schedules to compare
            
        Returns:
            Dictionary with winner index, choice letter, and reasoning
        """
        attempts = 0
        max_attempts = 5

        while attempts < max_attempts:
            try:
                # Get solution dictionaries
                solutions = []
                for idx in batch_indices:
                    sol = self.solutions_df.iloc[idx].to_dict()
                    solutions.append(sol)
                
                # Format prompt
                prompt = self.prompt_template.format(solutions)
                print(f"\n{'='*50}")
                print("Batch comparison prompt:")
                print(prompt)
                print(f"{'='*50}\n")
                
                # Call LLM
                if self.LLM :
                    response = self.llm_client.generate_response(
                        prompt,
                        max_new_tokens=4096,
                        seed=self.random_seed
                    )
                    print(f"LLM Response:\n{response}\n")
                    
                    # If the model explicitly outputs an invalid or non-answer (e.g., 'FINAL: N/A')
                    # record a null decision instead of forcing a choice.
                    import re as _re
                    if _re.search(r"FINAL\s*[:=]\s*(N/?A|NA|NONE|null)", response, _re.IGNORECASE):
                        self.prompt_template.add_to_history(solutions, None, response)
                        return {
                            'winner_idx': None,
                            'choice_letter': None,
                            'reasoning': response,
                            'batch_indices': batch_indices,
                            'solutions': solutions
                        }

                    # Parse response (restrict choices to current batch size)
                    choice_letter = self.prompt_template.parse_response(response, num_options=len(batch_indices))
                  # Convert letter to index
                else:
                    response, reasoning = self.llm_client.call_oracle(prompt, solutions )
                    print(f"Oracle Response:\n{response}\n", reasoning)    
                    import re as _re
                    if _re.search(r"FINAL\s*[:=]\s*(N/?A|NA|NONE|null)", response, _re.IGNORECASE):
                        self.prompt_template.add_to_history(solutions, None, response)
                        return {
                            'winner_idx': None,
                            'choice_letter': None,
                            'reasoning': response,
                            'batch_indices': batch_indices,
                            'solutions': solutions
                        }
                    choice_letter = self.prompt_template.parse_response(response, num_options=len(batch_indices))
                winner_idx_in_batch = ord(choice_letter) - ord('A')
                # Defensive guard in case parsing returned an unexpected label
                if winner_idx_in_batch < 0 or winner_idx_in_batch >= len(batch_indices):
                    print(f"[WARN] Parsed choice '{choice_letter}' out of range for batch size {len(batch_indices)}. Recording null decision.")
                    self.prompt_template.add_to_history(solutions, None, response)
                    return {
                        'winner_idx': None,
                        'choice_letter': None,
                        'reasoning': response,
                        'batch_indices': batch_indices,
                        'solutions': solutions
                    }
                winner_idx = batch_indices[winner_idx_in_batch]
                
                # Extract reasoning if present
                reasoning = None
                if "FINAL:" in response:
                    reasoning = response.split("FINAL:")[0].strip()
                
                # Add to history
                self.prompt_template.add_to_history(solutions, choice_letter, reasoning)
                
                return {
                    'winner_idx': winner_idx,
                    'choice_letter': choice_letter,
                    'reasoning': reasoning,
                    'batch_indices': batch_indices,
                    'solutions': solutions
                }
            except Exception as e:
                attempts += 1
                print(f"Error during preference collection (attempt {attempts}/{max_attempts}): {e}")
        
        # If all attempts fail, save history and end the run early
        print("Maximum retry attempts reached. Ending run early.")
        self.save_history({
            'error': f"Failed to collect preference after {max_attempts} attempts.",
            'batch_indices': batch_indices
        }, "early_termination_history.json")
        raise RuntimeError(f"Failed to collect preference after {max_attempts} attempts.")
    
    def run_experiment(
        self,
        n_batches: int,
        history_file: str,
        save_interval: int = 5
    ) -> Dict:
        """
        Run batch dueling bandit experiment with EUBO selection.
        
        Args:
            n_batches: Number of batches to run
            history_file: File to save history
            save_interval: Save history every N batches
            
        Returns:
            Results dictionary
        """
        print(f"Starting dueling bandit experiment with acquisition='{self.optimizer.acquisition}'")
        print(f"Batch size: {self.batch_size}")
        print(f"Total batches: {n_batches}")
        print(f"Total solutions: {self.n_solutions}")
        print(f"Acquisition: {self.optimizer.acquisition}")
        
        # Initialize history
        history = {
            'metadata': {
                'batch_size': self.batch_size,
                'n_batches': n_batches,
                'n_solutions': self.n_solutions,
                'metric_columns': self.metric_columns,
                'acquisition': self.optimizer.acquisition,
                'C': self.optimizer.C,
                'start_time': datetime.now().isoformat()
            },
            'batch_comparisons': [],
            'batch_summaries': [],
            'schedule_statistics': [],
            'model_utilities': []
        }
        
        # Run batches
        for batch_idx in range(n_batches):
            print(f"\n{'='*60}")
            print(f"BATCH {batch_idx + 1}/{n_batches}")
            print(f"{'='*60}")
            
            # Select next batch using selected acquisition
            batch_indices = self.optimizer.select_next_batch()
            print(f"Selected solutions (via {self.optimizer.acquisition}): {batch_indices}")
            print(f"Unshown solutions remaining: {len(self.optimizer.unshown_options)}")
            
            # Collect preference
            preference = self.collect_batch_preference(batch_indices)
            
            # Update optimizer with winner when available (skip if null)
            if preference.get('winner_idx') is not None:
                self.optimizer.update_with_winner(
                    batch_indices=batch_indices,
                    winner_idx=preference['winner_idx']
                )
            
            # Get current ranking
            ranking_info = self.optimizer.get_current_ranking(top_k=5)
            
            # Determine the index with the highest expected utility in the batch
            utilities = self.optimizer.model.posterior_mean_util(self.features)
            highest_utility_idx = max(batch_indices, key=lambda idx: utilities[idx])
            
            # Record comparison
            comparison = {
                'batch_num': batch_idx + 1,
                'batch_indices': batch_indices,
                'winner_idx': preference['winner_idx'],
                'choice_letter': preference['choice_letter'],
                'reasoning': preference['reasoning'],
                'highest_utility_idx': highest_utility_idx,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add solution details
            batch_solutions = {}
            for i, idx in enumerate(batch_indices):
                letter = chr(ord('A') + i)
                solution_data = self.solutions_df.iloc[idx].to_dict()
                batch_solutions[f'option_{letter}'] = {
                    'index': idx,
                    'metrics': solution_data,
                    'wins_at_time': int(self.optimizer.win_counts[idx]),
                    'appearances_at_time': int(self.optimizer.appearance_counts[idx])
                }
            comparison['batch_solutions'] = batch_solutions
            if preference.get('winner_idx') is not None:
                comparison['winner_solution'] = self.solutions_df.iloc[preference['winner_idx']].to_dict()
            else:
                comparison['winner_solution'] = None
            comparison['highest_utility_solution'] = self.solutions_df.iloc[highest_utility_idx].to_dict()
            
            history['batch_comparisons'].append(comparison)
            
            # Record batch summary
            summary = {
                'batch_num': batch_idx + 1,
                'top_5_indices': ranking_info['ranking'][:5],
                'top_5_utilities': ranking_info['utilities'][:5],
                'top_5_uncertainties': ranking_info['uncertainties'][:5],
                'top_5_win_rates': ranking_info['win_rates'][:5],
                'total_schedules_seen': len(self.optimizer.shown_options),
                'unshown_remaining': len(self.optimizer.unshown_options)
            }
            history['batch_summaries'].append(summary)
            
            # Store model utilities if available
            if self.optimizer.model.ready():
                utilities = self.optimizer.model.posterior_mean_util(self.features)
                history['model_utilities'].append({
                    'batch_num': batch_idx + 1,
                    'utilities': utilities.tolist()
                })
            
            print(f"\nCurrent top 5 solutions:")
            for i, (idx, util, wr) in enumerate(zip(
                ranking_info['ranking'][:5],
                ranking_info['utilities'][:5],
                ranking_info['win_rates'][:5]
            )):
                print(f"  {i+1}. Solution {idx}: "
                      f"Utility = {util:.3f}, "
                      f"Win rate = {wr:.2f}")
            
            # Save periodically
            if (batch_idx + 1) % save_interval == 0:
                self.save_history(history, history_file)
        
        # Final statistics
        history['metadata']['end_time'] = datetime.now().isoformat()
        
        # Add final solution statistics
        final_ranking = self.optimizer.get_current_ranking(top_k=self.n_solutions)
        for i, idx in enumerate(final_ranking['ranking']):
            history['schedule_statistics'].append({
                'solution_idx': idx,
                'final_rank': i + 1,
                'wins': int(self.optimizer.win_counts[idx]),
                'appearances': int(self.optimizer.appearance_counts[idx]),
                'win_rate': float(self.optimizer.win_counts[idx] / 
                                 max(1, self.optimizer.appearance_counts[idx])),
                'final_utility': final_ranking['utilities'][min(i, len(final_ranking['utilities'])-1)]
                if i < len(final_ranking['utilities']) else 0.0
            })
        
        # Save final history
        self.save_history(history, history_file)
        
        # Prepare results
        results = {
            'final_ranking': final_ranking['ranking'][:10],
            'final_utilities': final_ranking['utilities'][:10],
            'final_win_rates': final_ranking['win_rates'][:10],
            'total_batches': n_batches,
            'total_solutions_evaluated': len(self.optimizer.shown_options),
            'history_file': history_file
        }
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Top 10 solutions:")
        for i in range(min(10, len(final_ranking['ranking']))):
            idx = final_ranking['ranking'][i]
            print(f"  {i+1}. Solution {idx}: "
                  f"Utility = {final_ranking['utilities'][i]:.3f}, "
                  f"Win rate = {final_ranking['win_rates'][i]:.2f} "
                  f"({int(self.optimizer.win_counts[idx])}/{int(self.optimizer.appearance_counts[idx])} wins)")
        
        return results
    
    def save_history(self, history: Dict, filename: str):
        """Save history to JSON file."""
        json_file = filename.replace('.csv', '.json')
        with open(json_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        print(f"History saved to {json_file}")


def build_comparison_plot_data(history: Dict[str, Any], metric_columns: List[str]) -> Dict[str, Any]:
    """
    Build a compact plotting payload from a dueling comparison history.

    Returns a dict with keys:
      - iterations: [{iteration, best_solution_idx, best_solution}]
      - convergence_analysis: {stabilized, final_best_solution_idx, solution_trajectory, metric_trajectories}
      - final_best_solution_idx
      - final_best_solution
    """
    allowed_metric_keys: Set[str] = set(metric_columns or [])

    def _json_safe(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _json_safe(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_json_safe(val) for val in v]
        try:
            # ensure floats are JSON-safe
            if isinstance(v, float):
                import math
                return v if math.isfinite(v) else None
        except Exception:
            pass
        return v

    iterations: List[Dict[str, Any]] = []
    for bc in history.get('batch_comparisons', []) or []:
        winner = bc.get('winner_solution', {}) or bc.get('winner_schedule', {}) or {}
        # filter to metrics of interest and remap id if present
        filtered: Dict[str, Any] = {k: winner.get(k) for k in winner.keys() if k in allowed_metric_keys or k == 'Unnamed: 0'}
        if 'Unnamed: 0' in filtered:
            filtered['solution_id'] = filtered.pop('Unnamed: 0')
        iterations.append({
            'iteration': bc.get('batch_num'),
            'best_solution_idx': bc.get('winner_idx'),
            'best_solution': _json_safe(filtered),
        })

    solution_traj: List[Any] = [it.get('best_solution_idx') for it in iterations]
    metric_trajs: Dict[str, List[Any]] = {}
    for m in metric_columns or []:
        metric_trajs[m] = [
            (it.get('best_solution') or {}).get(m) for it in iterations
        ]

    def _stabilized(traj: List[Any]) -> bool:
        if not traj:
            return False
        if len(traj) < 3:
            return False
        return traj[-1] == traj[-2] == traj[-3] or len(set(traj)) == 1

    final_idx = solution_traj[-1] if solution_traj else None
    final_sol = (iterations[-1].get('best_solution') if iterations else None)

    convergence = {
        'stabilized': _stabilized(solution_traj),
        'final_best_solution_idx': final_idx,
        'solution_trajectory': solution_traj,
        'metric_trajectories': metric_trajs,
    }

    return {
        'iterations': iterations,
        'convergence_analysis': convergence,
        'final_best_solution_idx': final_idx,
        'final_best_solution': final_sol,
    }