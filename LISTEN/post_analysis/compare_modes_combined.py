#!/usr/bin/env python3
"""
Generate human-rank convergence comparisons for multiple modes.

By default all modes and algorithms are plotted together on a single figure,
so every trace is evaluated against that mode's own human ranking. A
side-by-side panel layout remains available via --layout panels.

Examples
--------
python post_analysis/compare_modes_combined.py \
    --scenario headphones --modes STUDENT STUDENT_HARD \
    --algo tournament:gemini --algo utility:gemini --reps 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Allow direct import of plotting helpers when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from plotting import (
    load_human_rank_series,
    _group_convergence_stats_for,
    _render_convergence_plot_generic,
)

PALETTE = [
    '#4477AA',  # Blue
    '#EE7733',  # Orange
    '#228833',  # Green
    '#AA3377',  # Purple
    '#66CCEE',  # Cyan
    '#CCBB44',  # Yellow
    '#009988',  # Teal
    '#CC3311',  # Dark Red
    '#BBBBBB',  # Gray
    '#EE6677',  # Red/Pink
]

LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h']

MODE_NAME_OVERRIDES = {
    ('headphones', 'STUDENT'): 'Headphones High Concordance',
    ('headphones', 'STUDENT_HARD'): 'Headphones Low Concordance',
}

METHOD_NAME_OVERRIDES = {
    'utility': 'LISTEN-U',
    'util': 'LISTEN-U',
    'tournament': 'LISTEN-T',
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compare modes using human weighted-rank convergence. Each mode is "
            "evaluated using its own human_sol ordering."
        )
    )
    ap.add_argument('--scenario', default='headphones', help='Scenario name (default: headphones)')
    ap.add_argument(
        '--modes',
        nargs='+',
        default=['STUDENT', 'STUDENT_HARD'],
        help='List of modes to compare (default: STUDENT STUDENT_HARD)',
    )
    ap.add_argument(
        '--algo',
        dest='algos',
        action='append',
        default=[],
        help=(
            "Algorithm/API pair in the form algo:api (e.g., tournament:gemini). "
            "Pass multiple times; defaults to tournament:gemini and utility:groq."
        ),
    )
    ap.add_argument('--max-iters', type=int, default=25, help='Max iterations filter (default: 25)')
    ap.add_argument('--reps', type=int, default=10, help='Number of seeds per method to include (default: 10)')
    ap.add_argument(
        '--include-base',
        action='store_true',
        help='Also include BASE-mode runs when loading results (default: off)',
    )
    ap.add_argument(
        '--include-baselines',
        action='store_true',
        help='Add baseline strategy runs to the comparison (default: off)',
    )
    ap.add_argument(
        '--layout',
        choices=['combined', 'panels'],
        default='combined',
        help='Plot layout: combined single plot (default) or per-mode panels',
    )
    ap.add_argument(
        '--no-caption',
        action='store_true',
        help='Omit figure captions/supertitles when rendering plots',
    )
    ap.add_argument(
        '--no-legend',
        action='store_true',
        help='Hide legends from the generated figures',
    )
    return ap.parse_args()


def _parse_algo_pairs(pairs: Iterable[str]) -> Dict[str, str]:
    if not pairs:
        pairs = ['tournament:gemini', 'utility:groq']
    mapping: Dict[str, str] = {}
    for raw in pairs:
        if ':' not in raw:
            raise ValueError(f"Expected algo:api pair, got '{raw}'")
        algo, api = raw.split(':', 1)
        algo = algo.strip()
        api = api.strip()
        if not algo or not api:
            raise ValueError(f"Invalid algo:api pair '{raw}'")
        mapping[algo] = api
    return mapping


def _pretty(label: str) -> str:
    return label.replace('_', ' ').title()


def _method_prefix(algo: str) -> str:
    mapping = {
        'utility': 'LISTEN-U',
        'tournament': 'LISTEN-T',
        'comparison': 'comp',
        'baseline': 'baseline',
    }
    return mapping.get(algo, algo)


def _mode_display(scenario: str, mode: str) -> str:
    scen_key = str(scenario).lower()
    mode_key = str(mode).upper()
    override = MODE_NAME_OVERRIDES.get((scen_key, mode_key))
    if override:
        return override
    return _pretty(mode)


def _friendly_method_name(algo: str) -> str:
    if not algo:
        return ''
    return METHOD_NAME_OVERRIDES.get(str(algo).lower(), _pretty(algo))


def _load_mode_data(
    scenario: str,
    mode: str,
    algo_configs: Dict[str, str],
    max_iters: int,
    reps: int,
    include_base: bool,
    include_baselines: bool,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    mode_label = _mode_display(scenario, mode)
    for algo, api_model in algo_configs.items():
        df = load_human_rank_series(
            scenario=scenario,
            mode=mode,
            api_model=api_model,
            max_iters=max_iters,
            reps=reps,
            include_base=include_base,
            include_baselines=include_baselines,
        )
        if df.empty:
            continue
        prefix = _method_prefix(algo)
        algo_mask = df['method'].astype(str).str.startswith(prefix)
        if not algo_mask.any():
            continue
        df = df.loc[algo_mask].copy()
        friendly_method = _friendly_method_name(algo)
        if mode_label:
            combined_label = f"{mode_label}:\n  {friendly_method}"
        else:
            combined_label = friendly_method
        df['method_combined'] = combined_label
        df['method_panel'] = combined_label
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_styles(methods: List[str]) -> Dict[str, Dict[str, str]]:
    styles: Dict[str, Dict[str, str]] = {}
    for idx, method in enumerate(methods):
        styles[method] = {
            'color': PALETTE[idx % len(PALETTE)],
            'linestyle': LINESTYLES[idx % len(LINESTYLES)],
            'marker': MARKERS[idx % len(MARKERS)],
        }
    return styles


def _plot_panel(ax, grouped: pd.DataFrame, styles: Dict[str, Dict[str, str]], mode_label: str, show_ylabel: bool) -> Tuple[float | None, float | None]:
    if grouped.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, alpha=0.7)
        ax.set_title(mode_label, fontsize=18, fontweight='bold', pad=12)
        ax.set_xlabel('Iteration', fontsize=18, fontweight='bold')
        if show_ylabel:
            ax.set_ylabel('Normalized Average Rank (mean +/- 2SE)', fontsize=18, fontweight='bold')
        ax.grid(False)
        ax.tick_params(axis='both', which='major', labelsize=15)
        return None, None

    global_min: float | None = None
    global_max: float | None = None

    for method, g in grouped.groupby('method'):
        style = styles.get(method, {
            'color': '#333333',
            'linestyle': '-',
            'marker': 'o',
        })
        g_sorted = g.sort_values('iteration')
        ax.errorbar(
            g_sorted['iteration'],
            g_sorted['mean'],
            yerr=g_sorted['two_se'],
            label=method,
            color=style['color'],
            linestyle=style['linestyle'],
            marker=style['marker'],
            linewidth=2.5,
            markersize=6,
            capsize=3,
            capthick=0.8,
            elinewidth=0.8,
            alpha=0.85,
            markeredgewidth=0.5,
            markeredgecolor='white',
        )
        lower = (g_sorted['mean'] - g_sorted['two_se']).min()
        upper = (g_sorted['mean'] + g_sorted['two_se']).max()
        if global_min is None or lower < global_min:
            global_min = lower
        if global_max is None or upper > global_max:
            global_max = upper

    ax.set_title(mode_label, fontsize=18, fontweight='bold', pad=12)
    ax.set_xlabel('Iteration', fontsize=18, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Normalized Average Rank (mean +/- 2SE)', fontsize=82, fontweight='bold')
    else:
        ax.set_ylabel('')

    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=10)
    return global_min, global_max


def _render_combined_plot(
    combined_df: pd.DataFrame,
    scenario: str,
    modes: List[str],
    include_caption: bool,
    include_legend: bool,
    algo_slug: str,
    api_slug: str,
) -> None:
    grouped = _group_convergence_stats_for(combined_df, 'weighted_rank')
    fig, _ = _render_convergence_plot_generic(
        grouped=grouped,
        title='',
        y_label='Normalized Average Rank (mean +/- 2SE)',
        scenario=None,
        mode=None,
        draw_ref=False,
        include_caption=False,
        include_legend=include_legend,
    )

    output_dir = Path(__file__).parent / 'plots_compare'
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_slug = '__'.join(modes)
    out_path = output_dir / (
        f"{scenario}__{mode_slug}__{algo_slug}__{api_slug}__combined__HUMANRANK.pdf"
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved combined comparison to: {out_path}")
    plt.show()


def _render_panel_plot(
    panels: List[Tuple[str, pd.DataFrame]],
    modes: List[str],
    scenario: str,
    include_caption: bool,
    include_legend: bool,
    algo_slug: str,
    api_slug: str,
) -> None:
    if not any(not df.empty for _, df in panels):
        print("No data found for the requested modes/algorithms.")
        return

    all_methods: List[str] = []
    for _, df in panels:
        if not df.empty:
            all_methods.extend(df['method_panel'].astype(str).unique().tolist())
    all_methods = list(dict.fromkeys(all_methods))
    if not all_methods:
        print("No matching runs found after filtering.")
        return

    styles = _build_styles(all_methods)

    fig, axes = plt.subplots(
        1,
        len(panels),
        sharey=True,
        figsize=(7 * max(1, len(panels)), 6),
        squeeze=False,
    )
    axes_list = axes[0]

    global_min: float | None = None
    global_max: float | None = None

    for idx, (mode, df) in enumerate(panels):
        ax = axes_list[idx]
        mode_label = _mode_display(scenario, mode)
        if df.empty:
            grouped = pd.DataFrame()
        else:
            df_panel = df.copy()
            df_panel['method'] = df_panel['method_panel']
            grouped = _group_convergence_stats_for(df_panel, 'weighted_rank')
        local_min, local_max = _plot_panel(
            ax,
            grouped,
            styles,
            f"{mode_label} (human)",
            show_ylabel=(idx == 0),
        )
        if local_min is not None:
            if global_min is None or local_min < global_min:
                global_min = local_min
        if local_max is not None:
            if global_max is None or local_max > global_max:
                global_max = local_max

    if global_min is not None and global_max is not None:
        if global_max == global_min:
            padding = 0.05 * max(1.0, abs(global_max))
        else:
            padding = 0.02 * (global_max - global_min)
        ymin = global_min - padding
        ymax = global_max + padding
        for ax in axes_list:
            ax.set_ylim(ymin, ymax)

    if include_legend:
        legend_handles = [
            Line2D(
                [],
                [],
                color=styles[m]['color'],
                linestyle=styles[m]['linestyle'],
                marker=styles[m]['marker'],
                linewidth=2.5,
                markersize=6,
                label=m,
            )
            for m in all_methods
        ]
        legend_labels = [h.get_label() for h in legend_handles]
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=max(1, len(all_methods)),
            frameon=True,
            framealpha=0.9,
            edgecolor='gray',
        )
    else:
        fig.legend(
            [],
            [],
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            handlelength=0,
            handletextpad=0,
            borderpad=0,
        )

    output_dir = Path(__file__).parent / 'plots_compare'
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_slug = '__'.join(modes)
    output_path = output_dir / (
        f"{scenario}__{mode_slug}__{algo_slug}__{api_slug}__side_by_side__HUMANRANK.pdf"
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved side-by-side comparison to: {output_path}")

    plt.show()


def main() -> None:
    args = parse_args()
    algo_configs = _parse_algo_pairs(args.algos)
    include_caption = not bool(args.no_caption)
    include_legend = not bool(args.no_legend)
    algo_slug = '__'.join(sorted(algo_configs.keys())) if algo_configs else 'none'
    api_slug = '__'.join(sorted(set(algo_configs.values()))) if algo_configs else 'none'

    panels: List[Tuple[str, pd.DataFrame]] = []
    combined_frames: List[pd.DataFrame] = []

    for mode in args.modes:
        df = _load_mode_data(
            scenario=args.scenario,
            mode=mode,
            algo_configs=algo_configs,
            max_iters=args.max_iters,
            reps=args.reps,
            include_base=args.include_base,
            include_baselines=args.include_baselines,
        )
        panels.append((mode, df))
        if not df.empty:
            df_combined = df.copy()
            df_combined['method'] = df_combined['method_combined']
            combined_frames.append(df_combined)

    if args.layout == 'panels':
        _render_panel_plot(
            panels,
            args.modes,
            args.scenario,
            include_caption=include_caption,
            include_legend=include_legend,
            algo_slug=algo_slug,
            api_slug=api_slug,
        )
        return

    if not combined_frames:
        print("No data found for the requested modes/algorithms.")
        return

    combined_df = pd.concat(combined_frames, ignore_index=True)
    _render_combined_plot(
        combined_df,
        args.scenario,
        args.modes,
        include_caption=include_caption,
        include_legend=include_legend,
        algo_slug=algo_slug,
        api_slug=api_slug,
    )


if __name__ == '__main__':
    main()
