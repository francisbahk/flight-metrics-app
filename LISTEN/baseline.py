from __future__ import annotations
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


def _select_metrics(df: pd.DataFrame, metric_columns: List[str]) -> List[str]:
    numeric_cols: List[str] = []
    for col in metric_columns:
        if col not in df.columns:
            continue
        try:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().any():
                numeric_cols.append(col)
        except Exception:
            continue
    if not numeric_cols:
        raise ValueError("No numeric metrics available for baseline computation.")
    return numeric_cols


def _apply_metric_signs(df: pd.DataFrame, metric_columns: List[str], metric_signs: Optional[Dict[str, int]]) -> pd.DataFrame:
    if not metric_signs:
        return df[metric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    cols: List[str] = []
    signed = {}
    for col in metric_columns:
        if col not in df.columns:
            continue
        sign = int(metric_signs.get(col, -1))
        if sign == 0:
            # ignore in calculation
            continue
        val = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        # If metric is better when smaller, flip sign to make larger-is-better
        signed[col] = (-1.0 * val) if sign == -1 else val
        cols.append(col)
    if not cols:
        # Fall back to all provided metrics if all were excluded
        return df[metric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return pd.DataFrame(signed, index=df.index)


def _zscore_standardize(df_values: pd.DataFrame) -> pd.DataFrame:
    arr = df_values.to_numpy(dtype=float)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    # protect against zero std
    stds = np.where(stds == 0.0, 1.0, stds)
    z = (arr - means) / stds
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(z, index=df_values.index, columns=df_values.columns)


def _build_history_template(metric_columns: List[str]) -> Dict[str, Any]:
    return {
        "metadata": {
            "metric_columns": metric_columns,
            "start_time": None,
        },
        "batch_comparisons": [],
        "batch_summaries": [],
        "schedule_statistics": [],
        "model_utilities": [],
    }


def _history_append_winner(history: Dict[str, Any], batch_num: int, winner_idx: int, winner_row: Dict[str, Any]):
    comparison = {
        "batch_num": batch_num,
        "batch_indices": [winner_idx],
        "winner_idx": winner_idx,
        "choice_letter": "A",
        "reasoning": None,
        "highest_utility_idx": winner_idx,
        "winner_solution": winner_row,
        "timestamp": None,
    }
    history["batch_comparisons"].append(comparison)


def build_baseline_history(
    strategy: str,
    df: pd.DataFrame,
    metric_columns: List[str],
    metric_signs: Optional[Dict[str, int]],
    n_batches: int,
) -> Dict[str, Any]:
    """Build a pseudo-history compatible with build_comparison_plot_data.

    Strategies:
      - random: pick a random item each iteration
      - zscore-avg: flip per metric_signs, z-score per metric, average across metrics, rank
    """
    metric_columns_numeric = _select_metrics(df, metric_columns)
    history = _build_history_template(metric_columns_numeric)

    if strategy == "random":
        n = len(df)
        for t in range(1, n_batches + 1):
            idx = int(np.random.randint(0, n))
            winner_row = df.iloc[idx].to_dict()
            _history_append_winner(history, t, idx, winner_row)
        return history

    if strategy == "zscore-avg":
        # apply signs and standardize
        signed_df = _apply_metric_signs(df, metric_columns_numeric, metric_signs)
        z = _zscore_standardize(signed_df)
        avg = z.mean(axis=1)
        # bigger is better after sign flip and z-score
        ranking = avg.sort_values(ascending=False).index.tolist()
        # iterate through batches cycling top choices (or just repeat the top one)
        if len(ranking) == 0:
            # degenerate fallback
            for t in range(1, n_batches + 1):
                _history_append_winner(history, t, 0, df.iloc[0].to_dict())
            return history
        # choose the top element for all iterations (deterministic baseline)
        top_idx = int(ranking[0])
        winner_row = df.iloc[top_idx].to_dict()
        for t in range(1, n_batches + 1):
            _history_append_winner(history, t, top_idx, winner_row)
        return history

    raise ValueError(f"Unknown baseline strategy: {strategy}")


