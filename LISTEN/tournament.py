"""
Batch–Tournament LLM Selection (BTLS)

Purpose
-------
Select a single solution using an LLM as a preference oracle over randomly
sampled candidate solutions, organized in fixed-size batches with a final playoff.

Parameters
----------
batch_size B : int (default 50)
    Number of randomly sampled candidate solutions per preliminary batch.
    Uses Excel-style labels (A, B, ..., Z, AA, AB, ...) to support large batches.
iterations T : int
    Total number of LLM comparison rounds. Valid values are T = 1 or T >= 3.
    The case T = 2 is invalid (a playoff would have only one champion).

Interfaces (callables expected by the driver)
---------------------------------------------
sampler(n: int) -> List[Solution]
    Returns n i.i.d. random candidate solutions.
llm_choose(cands: List[Solution]) -> Solution
    Returns the single solution preferred by the LLM from the provided list.

Procedure
---------
If T = 1:
    • Sample B candidates with sampler(B).
    • Call llm_choose once on these B candidates and return the winner.

If T >= 3:
    • Let m := T - 1 be the number of preliminary batches.
    • For j = 1..m:
        – Sample B candidates with sampler(B).
        – Call llm_choose on that batch to obtain a batch champion c_j.
    • Final playoff: call llm_choose once on the m champions {c_1, …, c_m}
      and return the overall winner.

Complexity / Accounting
-----------------------
LLM calls:
    calls(T) = 1                 if T = 1
    calls(T) = T                 if T >= 3
Items shown to the LLM:
    shown(T) = B                 if T = 1
    shown(T) = m*B + m           if T >= 3, where m = T - 1
             = (T - 1) * (B + 1)

Notes
-----
• Non-determinism arises from both the sampler and the LLM; fix random seeds
  and deterministic LLM settings as needed for reproducibility.
• T=2 is invalid because a playoff with only one preliminary batch produces a
  single champion, making the final comparison degenerate.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
from datetime import datetime

import numpy as np
import pandas as pd

from prompt import ComparisonPromptAdapter


def _build_solution_plot_data(history: Dict[str, Any], metric_columns: List[str]) -> Dict[str, Any]:
    """Mirror comparison plot payloads using solution-centric keys."""
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    iterations: List[Dict[str, Any]] = []
    for comparison in history.get("batch_comparisons", []) or []:
        winner = comparison.get("winner_solution", {}) or {}
        filtered = {k: winner.get(k) for k in (metric_columns or []) if k in winner}
        iterations.append(
            {
                "iteration": comparison.get("batch_num"),
                "best_solution_idx": comparison.get("winner_idx"),
                "best_solution": _json_safe(filtered),
            }
        )

    solution_traj = [entry.get("best_solution_idx") for entry in iterations]
    metric_trajs: Dict[str, List[Any]] = {}
    for metric in metric_columns or []:
        metric_trajs[metric] = [
            (entry.get("best_solution") or {}).get(metric) for entry in iterations
        ]

    def _stabilized(traj: List[Any]) -> bool:
        if len(traj) < 3:
            return False
        return traj[-1] == traj[-2] == traj[-3] or len(set(traj)) == 1

    final_idx = solution_traj[-1] if solution_traj else None
    final_sol = iterations[-1].get("best_solution") if iterations else None

    return {
        "iterations": iterations,
        "convergence_analysis": {
            "stabilized": _stabilized(solution_traj),
            "final_best_solution_idx": final_idx,
            "solution_trajectory": solution_traj,
            "metric_trajectories": metric_trajs,
        },
        "final_best_solution_idx": final_idx,
        "final_best_solution": final_sol,
    }


def _excel_label_to_index(label: str) -> Optional[int]:
    """Convert Excel-style label (A, B, ..., Z, AA, AB, ...) to 0-based index.

    Returns None if the label cannot be parsed into alphabetic characters.
    """
    if label is None:
        return None
    s = "".join(ch for ch in str(label).strip().upper() if "A" <= ch <= "Z")
    if not s:
        return None
    value = 0
    for ch in s:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


class SolutionTournamentExperiment:
    """
    Generic tournament selection using an LLM as the preference oracle.

    Prompting mirrors the comparison algorithm via `ComparisonPromptAdapter`.
    Iterations follow the BTLS spec at the top of this file.
    """
    
    def __init__(
        self,
        metric_columns: List[str],
        llm_client,
        prompt_template: ComparisonPromptAdapter,
        *,
        solutions_df: pd.DataFrame,
        batch_size: int = 50,
        iterations: int = 3,
        random_seed: Optional[int] = None,
    ) -> None:
        assert iterations == 1 or iterations >= 3, "iterations T=2 is invalid"
        self.solutions_df = solutions_df.reset_index(drop=True)
        self.metric_columns = list(metric_columns or [])
        self.llm = llm_client
        self.prompt = prompt_template
        self.batch_size = int(batch_size)
        self.iterations = int(iterations)
        self.seed = random_seed

        if random_seed is not None:
            random.seed(int(random_seed))
            np.random.seed(int(random_seed))

        # For compact fallback decisions, precompute a numeric penalty view
        self._numeric_cols = [
            c for c in self.metric_columns
            if c in self.solutions_df.columns and pd.to_numeric(self.solutions_df[c], errors="coerce").notna().any()
        ]
        if self._numeric_cols:
            self._numeric_df = self.solutions_df[self._numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        else:
            self._numeric_df = pd.DataFrame(index=self.solutions_df.index)

    # ---------- Core helpers ----------
    def _sample_indices(self, n: int) -> List[int]:
        idxs = list(range(len(self.solutions_df)))
        random.shuffle(idxs)
        return idxs[:n]

    def _present_and_choose(self, candidate_indices: List[int]) -> Tuple[int, str, str, str]:
        items: List[Dict[str, Any]] = [self.solutions_df.iloc[i].to_dict() for i in candidate_indices]
        prompt = self.prompt.format(items)
        response = self.llm.generate_response(prompt)

        try:
            letter = self.prompt.parse_response(response)
        except Exception:
            letter = "A"

        winner: Optional[int] = None
        k = _excel_label_to_index(letter)
        if k is not None and 0 <= k < len(candidate_indices):
            winner = candidate_indices[int(k)]

        if winner is None:
            # Fallback: lowest sum across numeric metrics for the shown candidates
            if not self._numeric_cols:
                winner = candidate_indices[0]
            else:
                subset = self._numeric_df.iloc[candidate_indices]
                winner = int(subset.sum(axis=1).idxmin())

        return int(winner), letter, prompt, response

    # ---------- Public API ----------
    def run(self) -> Dict[str, Any]:
        """
        Execute tournament and return a compact results dict with history.
        History is shaped to work with `build_comparison_plot_data`.
        """
        history: Dict[str, Any] = {
            "metadata": {
                "batch_size": self.batch_size,
                "iterations": self.iterations,
                "n_solutions": len(self.solutions_df),
                "metric_columns": self.metric_columns,
                "start_time": datetime.now().isoformat(),
                "tournament_spec": "BTLS",
            },
            "batch_comparisons": [],  # reuse shape expected by plotting util
        }

        if self.iterations == 1:
            candidates = self._sample_indices(self.batch_size)
            winner_idx, letter, prompt, response = self._present_and_choose(candidates)
            print(f"\n{'='*80}")
            print(f"TOURNAMENT - Batch 1 (Single)")
            print(f"{'='*80}")
            print(f"PROMPT:\n{prompt}")
            print(f"\nRESPONSE:\n{response}")
            print(f"\nWINNER: {letter} (index {winner_idx})")
            print(f"{'='*80}\n")
            _winner_obj = self.solutions_df.iloc[winner_idx].to_dict()
            history["batch_comparisons"].append({
                "batch_num": 1,
                "stage": "single",
                "batch_indices": candidates,
                "winner_idx": winner_idx,
                "choice_letter": letter,
                "winner_solution": _winner_obj,
            })
            final_winner = winner_idx
        else:
            m = self.iterations - 1
            champions: List[int] = []
            for j in range(1, m + 1):
                candidates = self._sample_indices(self.batch_size)
                winner_idx, letter, prompt, response = self._present_and_choose(candidates)
                print(f"\n{'='*80}")
                print(f"TOURNAMENT - Batch {j} (Preliminary)")
                print(f"{'='*80}")
                print(f"PROMPT:\n{prompt}")
                print(f"\nRESPONSE:\n{response}")
                print(f"\nWINNER: {letter} (index {winner_idx})")
                print(f"{'='*80}\n")
                champions.append(winner_idx)
                _winner_obj = self.solutions_df.iloc[winner_idx].to_dict()
                history["batch_comparisons"].append({
                    "batch_num": j,
                    "stage": "prelim",
                    "batch_indices": candidates,
                    "winner_idx": winner_idx,
                    "choice_letter": letter,
                    "winner_solution": _winner_obj,
                })

            # Final playoff among champions
            final_candidates = champions
            winner_idx, letter, prompt, response = self._present_and_choose(final_candidates)
            print(f"\n{'='*80}")
            print(f"TOURNAMENT - Batch {m + 1} (Final Playoff)")
            print(f"{'='*80}")
            print(f"PROMPT:\n{prompt}")
            print(f"\nRESPONSE:\n{response}")
            print(f"\nWINNER: {letter} (index {winner_idx})")
            print(f"{'='*80}\n")
            _winner_obj = self.solutions_df.iloc[winner_idx].to_dict()
            history["batch_comparisons"].append({
                "batch_num": m + 1,
                "stage": "final",
                "batch_indices": final_candidates,
                "winner_idx": winner_idx,
                "choice_letter": letter,
                "winner_solution": _winner_obj,
            })
            final_winner = winner_idx

        history["metadata"]["end_time"] = datetime.now().isoformat()

        plot_data = _build_solution_plot_data(history, self.metric_columns)
        return {
            "final_winner_idx": int(final_winner),
            "final_winner_solution": self.solutions_df.iloc[final_winner].to_dict(),
            "final_winner": self.solutions_df.iloc[final_winner].to_dict(),
            "history": history,
            "plot_data": plot_data,
        }

__all__ = ["SolutionTournamentExperiment"]
