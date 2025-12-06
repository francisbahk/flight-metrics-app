#!/usr/bin/env python3
"""
Difficulty Analysis Script

This script measures the difficulty of each dataset-mode combination by:
1. Sampling random weights from uniform(-1, 1) for each metric
2. Computing utility scores for all solutions using these random weights
3. Finding the top solution (highest utility)
4. Checking if this top solution is in the human solution set
5. Repeating 1000 times and returning the fraction of matches along with
   an uncertainty estimate (± 2 * standard error)

The idea is that if a dataset-mode is "easy", random weights should often pick 
solutions that humans also prefer. If it's "hard", random weights should rarely 
match human preferences.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

# Add parent directory to path to import plotting utilities
sys.path.append(str(Path(__file__).parent.parent))

try:
    from post_analysis.plotting import _find_scenario_yaml, _load_yaml, _numeric_min_max
except ImportError:
    # Fallback if we can't import from plotting
    def _find_scenario_yaml(scenario: str) -> Path:
        cfg_dir = Path(__file__).resolve().parents[1] / "configs"
        if scenario.endswith((".yaml", ".yml")):
            p = cfg_dir / scenario
            if p.exists():
                return p
        cand1 = cfg_dir / f"{scenario}.yaml"
        cand2 = cfg_dir / f"{scenario}.yml"
        if cand1.exists():
            return cand1
        if cand2.exists():
            return cand2
        raise FileNotFoundError(f"Scenario '{scenario}' not found under {cfg_dir}")

    def _load_yaml(path: Path) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _numeric_min_max(df: pd.DataFrame, metrics: list[str]) -> dict[str, tuple[float, float]]:
        min_max = {}
        for m in metrics:
            try:
                series = pd.to_numeric(df[m], errors='coerce')
            except Exception:
                continue
            if series.notna().any():
                mn = float(series.min(skipna=True))
                mx = float(series.max(skipna=True))
                if mx == mn:
                    min_max[m] = (0.0, 1.0)  # constant column
                else:
                    min_max[m] = (mn, mx)
        return min_max


def normalize_metric(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    """Normalize a metric series to [0, 1] range."""
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_utility_scores(
    df: pd.DataFrame, 
    weights: Dict[str, float], 
    min_max: Dict[str, Tuple[float, float]],
    metric_cols: List[str]
) -> np.ndarray:
    """Compute utility scores for all solutions given weights."""
    utilities = np.zeros(len(df))
    
    for metric in metric_cols:
        if metric in weights and metric in df.columns and metric in min_max:
            # Normalize the metric to [0, 1]
            normalized = normalize_metric(
                pd.to_numeric(df[metric], errors='coerce'), 
                min_max[metric][0], 
                min_max[metric][1]
            )
            # Apply weight and add to utility
            utilities += weights[metric] * normalized.fillna(0.5).values
    
    return utilities


def measure_difficulty(
    scenario: str, 
    mode: str, 
    n_samples: int = 1000,
    random_seed: Optional[int] = None
) -> tuple[float, float]:
    """
    Measure difficulty of a scenario-mode combination.
    
    Returns:
        A tuple of (mean_match_rate, standard_error).
        mean_match_rate is the fraction of random weight samples where the top
        solution matches the human solution set. The standard error reflects
        the uncertainty of this estimate assuming independent Bernoulli trials.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Load scenario configuration
    cfg_path = _find_scenario_yaml(scenario)
    cfg = _load_yaml(cfg_path)
    
    # Load data
    root = Path(__file__).resolve().parents[1]
    data_csv = (root / cfg['data_csv']).resolve()
    df = pd.read_csv(data_csv)
    
    # Get metric columns and their min-max values
    metric_cols = list(cfg.get('metric_columns') or [])
    non_numeric = set(cfg.get('non_numeric_metrics') or cfg.get('non_numerical_metrics') or [])
    numeric_metrics = [m for m in metric_cols if m in df.columns and m not in non_numeric]
    min_max = _numeric_min_max(df, numeric_metrics)
    
    # Get human solutions for this mode
    # Special case for exam/REGISTRAR: load filtered solutions from filtered_exam_data.csv
    if scenario.lower() == 'exam' and mode.upper() == 'REGISTRAR':
        filtered_csv = root / 'configs' / 'filtered_exam_data.csv'
        if filtered_csv.exists():
            df_filtered = pd.read_csv(filtered_csv)
            # First column contains solution IDs (after the unnamed index column)
            filtered_sol_ids = df_filtered.iloc[:, 0].dropna().astype(int).tolist()
            human_sol = set(filtered_sol_ids)
        else:
            print(f"Warning: filtered_exam_data.csv not found for {scenario}-{mode}")
            human_sol = set()
    else:
        modes = cfg.get('modes') or {}
        mode_def = modes.get(mode) or {}
        human_sol = set(mode_def.get('human_sol') or [])
    
    if not human_sol:
        print(f"Warning: No human solutions found for {scenario}-{mode}")
        return 0.0, 0.0
    
    # Get the number of metrics to sample weights for
    n_metrics = len(numeric_metrics)
    if n_metrics == 0:
        print(f"Warning: No numeric metrics found for {scenario}-{mode}")
        return 0.0, 0.0

    matches = 0

    for _ in range(n_samples):
        # Sample random weights from uniform(-1, 1)
        random_weights = {metric: np.random.uniform(-1, 1) for metric in numeric_metrics}
        
        # Compute utility scores
        utilities = compute_utility_scores(df, random_weights, min_max, numeric_metrics)
        
        # Find top solution (highest utility)
        top_idx = np.argmax(utilities)
        
        # Check if top solution is in human solutions
        if top_idx in human_sol:
            matches += 1

    mean = matches / n_samples
    if n_samples > 1:
        se = np.sqrt(mean * (1.0 - mean) / n_samples)
    else:
        se = 0.0

    return mean, se


def analyze_all_scenarios(
    scenarios: Optional[List[str]] = None,
    n_samples: int = 1000,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Analyze difficulty for all scenarios and modes.
    
    Args:
        scenarios: List of scenario names to analyze. If None, auto-discover all scenarios.
        n_samples: Number of random weight samples per scenario-mode
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: scenario, mode, difficulty (mean), difficulty_mean,
        difficulty_se, difficulty_lower, difficulty_upper, has_human_sol, n_human_sol
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Auto-discover scenarios if not provided
    if scenarios is None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        scenarios = []
        for yaml_file in config_dir.glob("*.yaml"):
            scenarios.append(yaml_file.stem)
        for yaml_file in config_dir.glob("*.yml"):
            scenarios.append(yaml_file.stem)
    
    results = []
    root = Path(__file__).resolve().parents[1]
    
    for scenario in scenarios:
        try:
            cfg_path = _find_scenario_yaml(scenario)
            cfg = _load_yaml(cfg_path)
            modes = cfg.get('modes') or {}
            
            for mode_name, mode_def in modes.items():
                # Special case for exam/REGISTRAR: load filtered solutions from filtered_exam_data.csv
                if scenario.lower() == 'exam' and mode_name.upper() == 'REGISTRAR':
                    filtered_csv = root / 'configs' / 'filtered_exam_data.csv'
                    if filtered_csv.exists():
                        df_filtered = pd.read_csv(filtered_csv)
                        filtered_sol_ids = df_filtered.iloc[:, 0].dropna().astype(int).tolist()
                        human_sol = filtered_sol_ids
                    else:
                        human_sol = []
                else:
                    human_sol = mode_def.get('human_sol') or []
                has_human_sol = len(human_sol) > 0
                
                if has_human_sol:
                    print(f"Analyzing {scenario}-{mode_name}...")
                    difficulty_mean, difficulty_se = measure_difficulty(scenario, mode_name, n_samples, None)
                else:
                    print(f"Skipping {scenario}-{mode_name} (no human solutions)")
                    difficulty_mean, difficulty_se = np.nan, np.nan

                if np.isnan(difficulty_mean):
                    lower = upper = np.nan
                else:
                    margin = 2.0 * difficulty_se if not np.isnan(difficulty_se) else np.nan
                    if np.isnan(margin):
                        lower = upper = np.nan
                    else:
                        lower = max(0.0, difficulty_mean - margin)
                        upper = min(1.0, difficulty_mean + margin)

                results.append({
                    'scenario': scenario,
                    'mode': mode_name,
                    'difficulty': difficulty_mean,
                    'difficulty_mean': difficulty_mean,
                    'difficulty_se': difficulty_se,
                    'difficulty_lower': lower,
                    'difficulty_upper': upper,
                    'has_human_sol': has_human_sol,
                    'n_human_sol': len(human_sol)
                })
                
        except Exception as e:
            print(f"Error processing {scenario}: {e}")
            continue
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Measure dataset-mode difficulty using random weight sampling")
    parser.add_argument("--scenario", type=str, help="Specific scenario to analyze")
    parser.add_argument("--mode", type=str, help="Specific mode to analyze")
    parser.add_argument("--scenarios", nargs="+", help="List of scenarios to analyze")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of random weight samples")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility", default=42)
    parser.add_argument("--output", type=str, help="Output CSV file path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    
    args = parser.parse_args()
    
    if args.scenario and args.mode:
        # Analyze single scenario-mode
        difficulty_mean, difficulty_se = measure_difficulty(
            args.scenario,
            args.mode,
            args.n_samples,
            args.random_seed,
        )
        margin = 2.0 * difficulty_se
        lower = max(0.0, difficulty_mean - margin)
        upper = min(1.0, difficulty_mean + margin)
        print(
            f"Difficulty for {args.scenario}-{args.mode}: "
            f"{difficulty_mean:.4f} ± {margin:.4f} (95% CI approx. [{lower:.4f}, {upper:.4f}])"
        )
        
        if args.output:
            df = pd.DataFrame([{
                'scenario': args.scenario,
                'mode': args.mode,
                'difficulty': difficulty_mean,
                'difficulty_mean': difficulty_mean,
                'difficulty_se': difficulty_se,
                'difficulty_lower': lower,
                'difficulty_upper': upper,
                'n_samples': args.n_samples,
            }])
            df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
    
    else:
        # Analyze all scenarios
        scenarios = args.scenarios
        df = analyze_all_scenarios(scenarios, args.n_samples, args.random_seed)
        
        print("\nDifficulty Analysis Results:")
        print("=" * 50)
        print(df.to_string(index=False, float_format='%.4f'))
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
        
        # Summary statistics
        valid_difficulties = df['difficulty'].dropna()
        if len(valid_difficulties) > 0:
            print(f"\nSummary Statistics:")
            print(f"  Mean difficulty: {valid_difficulties.mean():.4f}")
            print(f"  Std difficulty:  {valid_difficulties.std():.4f}")
            print(f"  Min difficulty:  {valid_difficulties.min():.4f}")
            print(f"  Max difficulty:  {valid_difficulties.max():.4f}")
            if 'difficulty_se' in df.columns:
                valid_se = df['difficulty_se'].dropna()
                if len(valid_se) > 0:
                    mean_se = valid_se.mean()
                    print(f"  Mean SE:         {mean_se:.4f} (2SE ≈ {2.0 * mean_se:.4f})")


if __name__ == "__main__":
    main()
