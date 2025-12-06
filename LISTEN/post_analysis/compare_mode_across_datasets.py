#!/usr/bin/env python3
"""Compare convergence for the same mode across multiple datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from plotting import (
    _group_convergence_stats_for,
    _render_convergence_plot_generic,
    load_human_rank_series,
)

PALETTE = [
    '#4477AA',
    '#EE7733',
    '#228833',
    '#AA3377',
    '#66CCEE',
    '#CCBB44',
    '#009988',
    '#CC3311',
    '#BBBBBB',
    '#EE6677',
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare the same mode across datasets.")
    ap.add_argument(
        '--pair',
        action='append',
        required=True,
        help='ScenarioA,ScenarioB,Mode (use commas, repeat for multiple pairs)',
    )
    ap.add_argument(
        '--algo',
        dest='algos',
        action='append',
        default=[],
        help='Algorithm/API pair like utility:gemini (repeatable).',
    )
    ap.add_argument('--max-iters', type=int, default=25, help='Iteration cap (default: 25)')
    ap.add_argument('--reps', type=int, default=10, help='Seeds per method (default: 10)')
    ap.add_argument('--include-base', action='store_true', help='Allow BASE mode runs')
    ap.add_argument('--include-baselines', action='store_true', help='Include baselines')
    ap.add_argument('--no-caption', action='store_true', help='Drop suptitle caption')
    ap.add_argument('--no-legend', action='store_true', help='Hide figure legend')
    return ap.parse_args()


def _parse_algo_pairs(pairs: Iterable[str]) -> Dict[str, str]:
    if not pairs:
        return {'utility': 'gemini', 'tournament': 'groq'}
    mapping: Dict[str, str] = {}
    for raw in pairs:
        if ':' not in raw:
            raise ValueError(f"Expected algo:api pair, got '{raw}'")
        algo, api = raw.split(':', 1)
        mapping[algo.strip()] = api.strip()
    return mapping


def _pretty(name: str) -> str:
    return name.replace('_', ' ').title()


def _method_prefix(algo: str) -> str:
    mapping = {
        'utility': 'LISTEN-U',
        'tournament': 'LISTEN-T',
        'comparison': 'comp',
        'baseline': 'baseline',
    }
    return mapping.get(algo, algo)


METHOD_NAME_OVERRIDES = {
    'utility': 'LISTEN-U',
    'util': 'LISTEN-U',
    'tournament': 'LISTEN-T',
}


def _load_dataset_frame(
    scenario: str,
    mode: str,
    algo_configs: Dict[str, str],
    max_iters: int,
    reps: int,
    include_base: bool,
    include_baselines: bool,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for algo, api in algo_configs.items():
        df = load_human_rank_series(
            scenario=scenario,
            mode=mode,
            api_model=api,
            max_iters=max_iters,
            reps=reps,
            include_base=include_base,
            include_baselines=include_baselines,
        )
        if df.empty:
            continue
        prefix = _method_prefix(algo)
        mask = df['method'].astype(str).str.startswith(prefix)
        if not mask.any():
            continue
        subset = df.loc[mask].copy()
        friendly = METHOD_NAME_OVERRIDES.get(str(algo).lower(), _pretty(algo))
        subset['dataset_label'] = _pretty(scenario)
        subset['method'] = friendly
        frames.append(subset)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined['method'] = combined['dataset_label'] + ' · ' + combined['method']
    return combined


def _slug(parts: Iterable[str]) -> str:
    return '__'.join([p.replace('/', '-').replace(' ', '_') for p in parts if p])


def compare_pair(
    scenario_a: str,
    scenario_b: str,
    mode: str,
    algo_configs: Dict[str, str],
    max_iters: int,
    reps: int,
    include_base: bool,
    include_baselines: bool,
    include_caption: bool,
    include_legend: bool,
) -> None:
    df_a = _load_dataset_frame(
        scenario=scenario_a,
        mode=mode,
        algo_configs=algo_configs,
        max_iters=max_iters,
        reps=reps,
        include_base=include_base,
        include_baselines=include_baselines,
    )
    df_b = _load_dataset_frame(
        scenario=scenario_b,
        mode=mode,
        algo_configs=algo_configs,
        max_iters=max_iters,
        reps=reps,
        include_base=include_base,
        include_baselines=include_baselines,
    )
    if df_a.empty and df_b.empty:
        print(f"No data for {scenario_a}/{scenario_b} mode={mode}.")
        return

    df_all = pd.concat([df for df in (df_a, df_b) if not df.empty], ignore_index=True)
    grouped = _group_convergence_stats_for(df_all, 'weighted_rank')

    title = (
        f"Human Rank · { _pretty(mode) } · { _pretty(scenario_a) } vs { _pretty(scenario_b) }"
    )
    fig, _ = _render_convergence_plot_generic(
        grouped=grouped,
        title=title,
        y_label='Normalized Average Rank (mean +/- 2SE)',
        scenario=None,
        mode=None,
        draw_ref=False,
        include_caption=include_caption,
        include_legend=include_legend,
    )

    output_dir = Path(__file__).parent / 'plots_compare'
    output_dir.mkdir(parents=True, exist_ok=True)
    algo_slug = _slug(sorted(algo_configs.keys())) or 'none'
    api_slug = _slug(sorted(set(algo_configs.values()))) or 'none'
    out_name = _slug(
        [scenario_a, scenario_b, mode, algo_slug, api_slug, 'datasets', 'HUMANRANK']
    ) + '.pdf'
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved dataset comparison to: {out_path}")


def main() -> None:
    args = parse_args()
    algo_configs = _parse_algo_pairs(args.algos)
    include_caption = not bool(args.no_caption)
    include_legend = not bool(args.no_legend)

    for raw in args.pair:
        parts = [p.strip() for p in raw.split(',')]
        if len(parts) != 3:
            raise ValueError(f"Expected ScenarioA,ScenarioB,Mode but got '{raw}'")
        scenario_a, scenario_b, mode = parts
        compare_pair(
            scenario_a=scenario_a,
            scenario_b=scenario_b,
            mode=mode,
            algo_configs=algo_configs,
            max_iters=args.max_iters,
            reps=args.reps,
            include_base=args.include_base,
            include_baselines=args.include_baselines,
            include_caption=include_caption,
            include_legend=include_legend,
        )


if __name__ == '__main__':
    main()
