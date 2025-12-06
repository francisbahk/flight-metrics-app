import os, glob, json, re
from collections import defaultdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    # Publication-ready defaults
    sns.set_theme(style="white", context="paper", font_scale=1.15)
except Exception:
    sns = None

try:
    import yaml
except ImportError:
    yaml = None

# Friendly dataset display names keyed by (scenario, mode)
DATASET_TITLE_MAP: dict[tuple[str, str], str] = {
    ("exam", "registrar"): "Exam Scheduling",
    ("flight02", "complicated"): "Flights 02",
    ("flight01", "complicated_structured"): "Flights 01",
    ("flight00", "complicated_structured"): "Flights 00",
    ("headphones", "student"): "Headphones",
}

# Scenario-level fallback display names when a mode-specific label is unavailable
SCENARIO_TITLE_MAP: dict[str, str] = {}


def _dataset_display_name(scenario: str | None, mode: str | None) -> str | None:
    """Return a polished display name for scenario/mode combinations."""
    if not scenario:
        return None
    scenario_key = scenario.strip().lower()
    mode_key = (mode or "").strip().lower()

    if scenario_key and mode_key:
        mapped = DATASET_TITLE_MAP.get((scenario_key, mode_key))
        if mapped:
            return mapped

    scenario_mapped = SCENARIO_TITLE_MAP.get(scenario_key)
    if scenario_mapped:
        return scenario_mapped

    cleaned = scenario.replace('_', ' ').strip()
    return cleaned.title() if cleaned else None


def _format_dataset_title(
    scenario: str | None,
    mode: str | None,
    *,
    extras: list[str] | None = None,
    prefix: str | None = None,
) -> str:
    base_label = _dataset_display_name(scenario, mode)
    if base_label:
        title = f"Performance on {base_label} Dataset"
    else:
        scen_text = (scenario or "Unknown").replace('_', ' ').strip() or "Unknown"
        mode_text = (mode or "Unknown").replace('_', ' ').strip() or "Unknown"
        title = f"Performance on {scen_text.title()} · {mode_text} Dataset"

    extras = [part for part in (extras or []) if part]
    if extras:
        title = f"{title} ({', '.join(extras)})"

    if prefix:
        title = f"{prefix} · {title}"

    return title


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
    if not yaml:
        raise RuntimeError("PyYAML not installed; pip install pyyaml")
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
                # avoid divide by zero; treat as degenerate range
                min_max[m] = (mn, mn + 1.0)
            else:
                min_max[m] = (mn, mx)
    return min_max


def _compute_reference_gt_stats(scenario: str, mode: str) -> dict[str, float]:
    """Compute min/avg/max of hidden ground-truth utility across all solutions in the dataset.

    Uses the same YAML-defined weights and dataset-level min-max normalization as optimization runs.
    """
    root = Path(__file__).resolve().parents[1]
    cfg = _load_yaml(_find_scenario_yaml(scenario))
    data_csv = (root / cfg['data_csv']).resolve()
    df_all = pd.read_csv(data_csv)

    metric_cols = list(cfg.get('metric_columns') or [])
    non_numeric = set(cfg.get('non_numeric_metrics') or cfg.get('non_numerical_metrics') or [])
    numeric_metrics = [m for m in metric_cols if m in df_all.columns and m not in non_numeric]
    min_max = _numeric_min_max(df_all, numeric_metrics)

    modes = cfg.get('modes') or {}
    mode_def = modes.get(mode) or {}
    weights = mode_def.get('weights') or {}
    if not weights:
        return {}

    gt_values: list[float] = []
    for _, row in df_all.iterrows():
        metric_values = {m: row[m] for m in weights.keys() if m in df_all.columns}
        if not metric_values:
            continue
        gt = _compute_gt_from_metrics(metric_values, weights, min_max)
        gt_values.append(gt)

    if not gt_values:
        return {}

    s = pd.Series(gt_values)
    return {
        'min': float(s.min(skipna=True)),
        'avg': float(s.mean(skipna=True)),
        'max': float(s.max(skipna=True)),
    }


def _compute_gt_from_metrics(metric_values: dict, weights: dict, min_max: dict) -> float | None:
    total = 0.0
    contributed = False
    for m, w in (weights or {}).items():
        if w == 0:
            continue
        if m not in metric_values or m not in min_max:
            continue
        raw = metric_values.get(m)
        try:
            val = float(raw)
        except Exception:
            continue
        mn, mx = min_max[m]
        if mx == mn:
            norm = 0.0
        else:
            norm = (val - mn) / (mx - mn)
        if norm < 0.0:
            norm = 0.0
        elif norm > 1.0:
            norm = 1.0
        total += float(w) * norm
        contributed = True
    # If no metrics contributed (e.g., null/empty winner), return None so it becomes NaN downstream
    return total if contributed else None


def _extract_metric_trajectories(opt_results: dict) -> list[dict]:
    """Return list of per-iteration dicts of metric values using trajectories if present."""
    trajs = (opt_results or {}).get('convergence_analysis', {}).get('metric_trajectories')
    if isinstance(trajs, dict) and trajs:
        # Determine max length among trajectories
        max_len = max(len(v) for v in trajs.values() if isinstance(v, list))
        iters: list[dict] = []
        for i in range(max_len):
            d = {}
            for m, seq in trajs.items():
                if isinstance(seq, list) and i < len(seq):
                    d[m] = seq[i]
            iters.append(d)
        return iters

    # Fallback to iterations' best_solution snapshots
    it_list = (opt_results or {}).get('iterations') or []
    out = []
    for it in it_list:
        sol = it.get('best_solution') or {}
        out.append(dict(sol))
    return out


def _extract_solution_indices(opt_results: dict) -> list[int]:
    """Return list of solution indices (one per iteration) if present.

    Prefers convergence_analysis.solution_trajectory; falls back to
    iterations[*].best_solution_idx. Returns empty list if not found.
    """
    if not isinstance(opt_results, dict):
        return []
    ca = (opt_results or {}).get('convergence_analysis') or {}
    traj = ca.get('solution_trajectory')
    if isinstance(traj, list) and traj:
        try:
            return [int(x) if x is not None else None for x in traj]  # type: ignore[list-item]
        except Exception:
            pass
    it_list = (opt_results or {}).get('iterations') or []
    out: list[int] = []
    for it in it_list:
        if isinstance(it, dict) and 'best_solution_idx' in it:
            try:
                out.append(int(it.get('best_solution_idx')))
            except Exception:
                continue
    return out


def _method_label(meta: dict) -> str:
    algo = meta.get('algo') or meta.get('algorithm') or 'unknown'
    mode = meta.get('mode')
    if algo == 'comparison':
        cs = meta.get('comparison_settings') or {}
        mt = cs.get('model_type')
        acq = cs.get('acq')
        hist = cs.get('use_history')
        bs = cs.get('batch_size')
        parts = ["comp"]
        if mt:
            parts.append(str(mt))
        if acq:
            parts.append(str(acq))
        if hist is not None:
            parts.append(f"hist{1 if hist else 0}")
        if bs is not None:
            parts.append(f"bs{bs}")
        label = "/".join(parts)
    elif algo == 'utility':
        label = 'utility'
    elif algo == 'baseline':
        bs = meta.get('baseline_settings') or {}
        strat = bs.get('strategy')
        return f"baseline/{strat}" if strat else 'baseline'
    else:
        label = str(algo)

    alias_map = {
        'utility': 'LISTEN-U',
        'util': 'LISTEN-U',
        'tournament': 'LISTEN-T',
    }

    if mode == 'BASE':
        label = f"{label}@BASE"

    for raw, friendly in alias_map.items():
        if label.startswith(raw):
            suffix = label[len(raw):]
            label = f"{friendly}{suffix}"
            break
    return label


def _iters_from_name(filename: str) -> int | None:
    try:
        m = re.search(r"__iters(\d+)__", filename)
        if m:
            return int(m.group(1))
    except Exception:
        return None
    return None


def load_gt_series(
    scenario: str,
    mode: str,
    model_name: str | None = None,
    api_model: str | None = None,
    max_iters: int | None = None,
    reps: int | None = None,
    include_baselines: bool = False,
    include_comparison: bool = False,
    include_base: bool = True,
) -> pd.DataFrame:
    """
    Load optimization outputs for a scenario and compute hidden ground-truth series per run.

    Returns a tidy DataFrame with columns:
      ['filename','iteration','method','gt_value','scenario','mode','api_model','model_name']

    If `reps` is provided, only the first `reps` distinct seeds per method are included.
    If `include_base` is False, BASE mode runs are excluded.
    Comparison algorithm runs are excluded unless include_comparison=True.
    """
    root = Path(__file__).resolve().parents[1]
    outdir = root / "outputs" / scenario
    if not outdir.exists():
        return pd.DataFrame()

    cfg = _load_yaml(_find_scenario_yaml(scenario))
    data_csv = (root / cfg['data_csv']).resolve()
    # Determine numeric metrics for min-max
    metric_cols = list(cfg.get('metric_columns') or [])
    non_numeric = set(cfg.get('non_numeric_metrics') or cfg.get('non_numerical_metrics') or [])
    df_all = pd.read_csv(data_csv)
    numeric_metrics = [m for m in metric_cols if m in df_all.columns and m not in non_numeric]
    min_max = _numeric_min_max(df_all, numeric_metrics)

    # Mode weights
    modes = cfg.get('modes') or {}
    mode_def = modes.get(mode) or {}
    weights = mode_def.get('weights') or {}

    rows = []
    reps = reps if isinstance(reps, int) and reps > 0 else None
    # If reps is requested, preselect the lowest numeric seeds per method
    allowed_by_method: dict[str, set[str]] | None = None
    if reps is not None:
        candidates: dict[str, dict[str, int | None]] = defaultdict(dict)
        for fp in sorted(glob.glob(str(outdir / "*.json"))):
            try:
                with open(fp, 'r') as f:
                    payload = json.load(f)
            except Exception:
                continue

            meta = payload.get('meta') or {}
            if not meta:
                base = os.path.basename(fp)
                parts = base.split("__")
                if len(parts) >= 4:
                    meta = {
                        'scenario': parts[0],
                        'algo': parts[1],
                        'mode': parts[2],
                        'api_model': parts[3].replace('api', ''),
                    }

            algo_name = meta.get('algo')
            is_baseline = (algo_name == 'baseline')
            is_comparison = (algo_name == 'comparison')
            if is_baseline and (not include_baselines):
                continue
            if is_comparison and (not include_comparison):
                continue

            run_mode = meta.get('mode')
            if not is_baseline:
                if include_base:
                    if not (run_mode == mode or run_mode == 'BASE'):
                        continue
                else:
                    if run_mode != mode:
                        continue

            if not is_baseline:
                if model_name and (meta.get('model_name') or '').split('/')[-1] != model_name.split('/')[-1]:
                    continue
                if api_model and meta.get('api_model') != api_model:
                    continue

            configured_iters = meta.get('max_iters')
            if configured_iters is None:
                configured_iters = _iters_from_name(os.path.basename(fp))
            if isinstance(max_iters, int) and max_iters > 0 and configured_iters is not None and configured_iters != max_iters:
                continue

            method = _method_label(meta)

            seed_value = (
                meta.get('seed')
                or meta.get('random_seed')
                or meta.get('seed_value')
                or payload.get('seed')
            )
            if seed_value is None:
                seed_match = re.search(r"__seed([^_]+)__", os.path.basename(fp))
                if seed_match:
                    seed_value = seed_match.group(1)
            seed_id = str(seed_value) if seed_value is not None else os.path.basename(fp)
            seed_num: int | None = None
            try:
                seed_num = int(seed_id)
            except Exception:
                seed_num = None

            if seed_id not in candidates[method]:
                candidates[method][seed_id] = seed_num

        # choose lowest numeric seeds; fill with lexicographic non-numeric if needed
        allowed_by_method = {}
        for method, seed_map in candidates.items():
            numeric = [(sid, sn) for sid, sn in seed_map.items() if sn is not None]
            numeric.sort(key=lambda t: t[1])
            selected = [sid for sid, _ in numeric[:reps]]
            if len(selected) < reps:
                non_numeric = sorted([sid for sid, sn in seed_map.items() if sn is None])
                selected.extend(non_numeric[: max(0, reps - len(selected))])
            allowed_by_method[method] = set(selected)

    for fp in sorted(glob.glob(str(outdir / "*.json"))):
        try:
            with open(fp, 'r') as f:
                payload = json.load(f)
        except Exception:
            continue

        meta = payload.get('meta') or {}
        if not meta:
            # legacy: try inferring from filename
            base = os.path.basename(fp)
            parts = base.split("__")
            if len(parts) >= 4:
                meta = {
                    'scenario': parts[0],
                    'algo': parts[1],
                    'mode': parts[2],
                    'api_model': parts[3].replace('api', ''),
                }

        # Check if this is a baseline first
        algo_name = meta.get('algo')
        is_baseline = (algo_name == 'baseline')
        is_comparison = (algo_name == 'comparison')
        
        # Exclude baselines by default unless explicitly included
        if is_baseline and (not include_baselines):
            continue
        
        # Exclude comparison algorithms unless explicitly included
        if is_comparison and (not include_comparison):
            continue
        
        # Filter by mode - but allow baselines through regardless of mode if include_baselines is True
        run_mode = meta.get('mode')
        if not is_baseline:
            # For non-baselines, include the requested mode and optionally BASE mode
            if include_base:
                if not (run_mode == mode or run_mode == 'BASE'):
                    continue
            else:
                if run_mode != mode:
                    continue
        # else: baselines are included regardless of their stored mode
        
        # Skip model filters for baselines (they have null api_model and model_name)
        if not is_baseline:
            if model_name and (meta.get('model_name') or '').split('/')[-1] != model_name.split('/')[-1]:
                continue
            if api_model and meta.get('api_model') != api_model:
                continue

        # Respect max_iters filter by matching the run configuration, not truncating
        configured_iters = meta.get('max_iters')
        if configured_iters is None:
            configured_iters = _iters_from_name(os.path.basename(fp))
        if isinstance(max_iters, int) and max_iters > 0 and configured_iters is not None and configured_iters != max_iters:
            continue

        opt = payload.get('optimization_results') or {}
        metric_iters = _extract_metric_trajectories(opt)
        method = _method_label(meta)

        seed_value = (
            meta.get('seed')
            or meta.get('random_seed')
            or meta.get('seed_value')
            or payload.get('seed')
        )
        if seed_value is None:
            seed_match = re.search(r"__seed([^_]+)__", os.path.basename(fp))
            if seed_match:
                seed_value = seed_match.group(1)
        seed_id = str(seed_value) if seed_value is not None else os.path.basename(fp)

        if allowed_by_method is not None:
            method_key = str(method)
            allowed = allowed_by_method.get(method_key)
            if allowed is not None and seed_id not in allowed:
                continue

        for idx, metrics in enumerate(metric_iters, start=1):
            gt = _compute_gt_from_metrics(metrics, weights, min_max)
            rows.append({
                'filename': os.path.basename(fp),
                'iteration': idx,
                'method': method,
                'scenario': meta.get('scenario') or scenario,
                'mode': meta.get('mode'),
                'api_model': meta.get('api_model'),
                'model_name': (meta.get('model_name') or ''),
                'gt_value': gt,
            })

    df = pd.DataFrame(rows)
    if not df.empty and 'gt_value' in df.columns:
        # Ensure numeric dtype; invalid/missing become NaN for robust aggregation
        try:
            df['gt_value'] = pd.to_numeric(df['gt_value'], errors='coerce')
        except Exception:
            pass
    return df


def _group_convergence_stats_for(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Aggregate df into mean and 2*SE by method and iteration.

    Expects columns: ['method', 'iteration', value_col]
    Returns columns: ['method', 'iteration', 'mean', 'two_se', 'std', 'count']
    """
    grouped = df.groupby(['method', 'iteration'])[value_col].agg(['mean', 'std', 'count']).reset_index()
    grouped['se'] = grouped['std'] / grouped['count'].clip(lower=1).pow(0.5)
    grouped['two_se'] = 2.0 * grouped['se']
    return grouped


def _group_convergence_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for GT plots."""
    if df.empty:
        return df
    return _group_convergence_stats_for(df, 'gt_value')


def _infer_scenario_mode_from_df(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Infer single scenario and mode from tidy df if present, else (None, None)."""
    scenario = None
    mode = None
    try:
        scenarios = [s for s in df.get('scenario', pd.Series(dtype=str)).dropna().unique() if s]
        modes = [m for m in df.get('mode', pd.Series(dtype=str)).dropna().unique() if m]
        scenario = scenarios[0] if scenarios else None
        mode = modes[0] if modes else None
    except Exception:
        pass
    return scenario, mode


def _render_convergence_plot_generic(
    grouped: pd.DataFrame,
    title: str,
    y_label: str,
    scenario: str | None = None,
    mode: str | None = None,
    draw_ref: bool = True,
    ref_label_prefix: str = 'GT',
    include_caption: bool = True,
    include_legend: bool = True,
):
    """Render convergence errorbar plot given aggregated stats.

    Returns (fig, ax).
    """
    # Figure size: add width when legend is shown (space on the right)
    base_size = (12, 7)
    legend_extra_width = 2
    width = base_size[0] + (legend_extra_width if include_legend else 0)
    fig, ax = plt.subplots(1, 1, figsize=(width, base_size[1]))

    # Colorblind-friendly palette with maximum distinction
    # Based on Paul Tol's bright qualitative scheme + high-contrast additions
    palette = [
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
    
    # Line styles for better distinction
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    # Marker styles for additional distinction
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h']
    
    colors: dict[str, str] = {}
    for i, (method, g) in enumerate(grouped.groupby('method')):
        colors[str(method)] = palette[i % len(palette)]
        g_sorted = g.sort_values('iteration')
        ax.errorbar(
            g_sorted['iteration'],
            g_sorted['mean'],
            yerr=g_sorted['two_se'],
            label=str(method),
            color=colors[str(method)],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            linewidth=2.5,
            markersize=6,
            capsize=3,
            capthick=0.8,
            elinewidth=0.8,
            alpha=0.85,
            markeredgewidth=0.5,
            markeredgecolor='white',  # White edge for better visibility
        )

    # Optional dataset-level reference lines (min/avg/max) - subtle
    if draw_ref:
        try:
            if scenario and mode:
                stats = _compute_reference_gt_stats(str(scenario), str(mode))
                if stats:
                    # ax.axhline(stats['min'], color='#27AE60', linestyle=':', linewidth=1.5, alpha=0.6, label=f'{ref_label_prefix} min', zorder=0)
                    # ax.axhline(stats['avg'], color='#7F8C8D', linestyle=':', linewidth=1.5, alpha=0.6, label=f'{ref_label_prefix} avg', zorder=0)
                    ax.axhline(stats['max'], color='#E74C3C', linestyle=':', linewidth=1.5, alpha=0.6, label=f'{ref_label_prefix} max', zorder=0)
        except Exception:
            pass

    # Styling for publication quality
    ax.set_xlabel('Iteration', fontsize=18, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=18, fontweight='bold')
    
    # Subtle grid for easier reading
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)  # Grid behind data
    
    if include_legend:
        # Legend - place outside plot area to avoid overlapping data
        num_methods = len(grouped['method'].unique())
        legend_kwargs = dict(
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            frameon=True,
            fancybox=False,
            shadow=False,
            fontsize=16,
            edgecolor='gray',
            framealpha=0.9,
        )
        if num_methods > 6:
            legend_kwargs['ncol'] = 1
        ax.legend(**legend_kwargs)
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # Add some padding to y-axis for better visibility
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    ax.set_ylim(ymin - 0.02 * y_range, ymax + 0.02 * y_range)
    
    caption_text = (title or '').strip()
    has_caption = bool(include_caption and caption_text)
    if has_caption:
        # fig.suptitle(caption_text, fontsize=20, fontweight='bold', y=0.99)
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    else:
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    return fig, ax


def _render_convergence_plot(
    grouped: pd.DataFrame,
    title: str,
    scenario: str | None,
    mode: str | None,
    ref_label_prefix: str = 'GT',
    include_caption: bool = True,
    include_legend: bool = True,
):
    return _render_convergence_plot_generic(
        grouped=grouped,
        title=title,
        y_label='Hidden Ground Truth (mean ± 2SE)',
        scenario=scenario,
        mode=mode,
        draw_ref=True,
        ref_label_prefix=ref_label_prefix,
        include_caption=include_caption,
        include_legend=include_legend,
    )


def plot_convergence(
    df: pd.DataFrame,
    title: str = "Convergence (hidden ground truth)",
    scenario: str | None = None,
    mode: str | None = None,
    include_caption: bool = True,
    include_legend: bool = True,
):
    if df.empty:
        print("No data to plot.")
        return

    grouped = _group_convergence_stats(df)
    scen = scenario
    md = mode
    if scen is None or md is None:
        scen, md = _infer_scenario_mode_from_df(df)
    effective_title = title
    if (not title or title == "Convergence (hidden ground truth)") and (scen or md):
        effective_title = _format_dataset_title(scen, md)
    fig, ax = _render_convergence_plot(
        grouped,
        effective_title,
        scen,
        md,
        ref_label_prefix='Hidden ground truth',
        include_caption=include_caption,
        include_legend=include_legend,
    )
    plt.show()


def load_human_rank_series(
    scenario: str,
    mode: str,
    model_name: str | None = None,
    api_model: str | None = None,
    max_iters: int | None = None,
    reps: int | None = None,
    include_baselines: bool = False,
    include_comparison: bool = False,
    include_base: bool = True,
) -> pd.DataFrame:
    """Load outputs and compute normalized human weighted-rank per iteration.

    Returns a tidy DataFrame with columns:
      ['filename','iteration','method','weighted_rank','scenario','mode','api_model','model_name']
    where 'weighted_rank' already reflects normalization by the number of human
    solutions provided for the mode. Comparison algorithm runs are excluded
    unless include_comparison=True.
    """
    root = Path(__file__).resolve().parents[1]
    outdir = root / "outputs" / scenario
    if not outdir.exists():
        return pd.DataFrame()

    cfg = _load_yaml(_find_scenario_yaml(scenario))
    data_csv = (root / cfg['data_csv']).resolve()
    df_all = pd.read_csv(data_csv)
    total_solutions = int(len(df_all))
    if total_solutions <= 0:
        return pd.DataFrame()

    # Special case for exam/REGISTRAR: load filtered solutions from filtered_exam_data.csv
    if scenario.lower() == 'exam' and mode.upper() == 'REGISTRAR':
        filtered_csv = root / 'configs' / 'filtered_exam_data.csv'
        if filtered_csv.exists():
            df_filtered = pd.read_csv(filtered_csv)
            # First column contains solution IDs (after the unnamed index column)
            # The actual solution IDs are in the first column (index 0 after reading)
            filtered_sol_ids = df_filtered.iloc[:, 0].dropna().astype(int).tolist()
            # All filtered solutions get rank 1
            rank_lookup: dict[int, int] = {}
            for sid in filtered_sol_ids:
                rank_lookup[int(sid)] = 1
            ranked_len = len(rank_lookup)
        else:
            # Fallback if CSV doesn't exist
            rank_lookup = {}
            ranked_len = 0
    else:
        # Fetch human ranking for the requested mode
        modes = cfg.get('modes') or {}
        mode_def = modes.get(mode) or {}
        human_sol = list(mode_def.get('human_sol') or [])
        if not human_sol:
            # No human ranking available
            return pd.DataFrame()
        # Map solution id -> 1-based rank
        rank_lookup: dict[int, int] = {}
        for i, sid in enumerate(human_sol, start=1):
            try:
                rank_lookup[int(sid)] = int(i)
            except Exception:
                continue
        ranked_len = int(len(rank_lookup))
    if ranked_len <= 0:
        return pd.DataFrame()
    normalize_by = float(total_solutions)
    # Average rank for unranked items: mean of [ranked_len+1, ..., total_solutions]
    # If ranked_len == total_solutions, this term is unused
    if ranked_len < total_solutions:
        unranked_avg = (float(ranked_len + 1) + float(total_solutions)) / 2.0
    else:
        unranked_avg = float(ranked_len)  # degenerate, all ranked
    unranked_norm = unranked_avg / normalize_by

    rows = []
    reps = reps if isinstance(reps, int) and reps > 0 else None
    # If reps is requested, preselect the lowest numeric seeds per method
    allowed_by_method: dict[str, set[str]] | None = None
    if reps is not None:
        candidates: dict[str, dict[str, int | None]] = defaultdict(dict)
        for fp in sorted(glob.glob(str(outdir / "*.json"))):
            try:
                with open(fp, 'r') as f:
                    payload = json.load(f)
            except Exception:
                continue

            meta = payload.get('meta') or {}
            if not meta:
                base = os.path.basename(fp)
                parts = base.split("__")
                if len(parts) >= 4:
                    meta = {
                        'scenario': parts[0],
                        'algo': parts[1],
                        'mode': parts[2],
                        'api_model': parts[3].replace('api', ''),
                    }

            algo_name = meta.get('algo')
            is_baseline = (algo_name == 'baseline')
            is_comparison = (algo_name == 'comparison')
            if is_baseline and (not include_baselines):
                continue
            if is_comparison and (not include_comparison):
                continue

            run_mode = meta.get('mode')
            if not is_baseline:
                if include_base:
                    if not (run_mode == mode or run_mode == 'BASE'):
                        continue
                else:
                    if run_mode != mode:
                        continue

            if not is_baseline:
                if model_name and (meta.get('model_name') or '').split('/')[-1] != model_name.split('/')[-1]:
                    continue
                if api_model and meta.get('api_model') != api_model:
                    continue

            configured_iters = meta.get('max_iters')
            if configured_iters is None:
                configured_iters = _iters_from_name(os.path.basename(fp))
            if isinstance(max_iters, int) and max_iters > 0 and configured_iters is not None and configured_iters != max_iters:
                continue

            method = _method_label(meta)

            seed_value = (
                meta.get('seed')
                or meta.get('random_seed')
                or meta.get('seed_value')
                or payload.get('seed')
            )
            if seed_value is None:
                seed_match = re.search(r"__seed([^_]+)__", os.path.basename(fp))
                if seed_match:
                    seed_value = seed_match.group(1)
            seed_id = str(seed_value) if seed_value is not None else os.path.basename(fp)
            seed_num: int | None = None
            try:
                seed_num = int(seed_id)
            except Exception:
                seed_num = None

            if seed_id not in candidates[method]:
                candidates[method][seed_id] = seed_num

        # choose lowest numeric seeds; fill with lexicographic non-numeric if needed
        allowed_by_method = {}
        for method, seed_map in candidates.items():
            numeric = [(sid, sn) for sid, sn in seed_map.items() if sn is not None]
            numeric.sort(key=lambda t: t[1])
            selected = [sid for sid, _ in numeric[:reps]]
            if len(selected) < reps:
                non_numeric = sorted([sid for sid, sn in seed_map.items() if sn is None])
                selected.extend(non_numeric[: max(0, reps - len(selected))])
            allowed_by_method[method] = set(selected)

    for fp in sorted(glob.glob(str(outdir / "*.json"))):
        try:
            with open(fp, 'r') as f:
                payload = json.load(f)
        except Exception:
            continue

        meta = payload.get('meta') or {}
        if not meta:
            base = os.path.basename(fp)
            parts = base.split("__")
            if len(parts) >= 4:
                meta = {
                    'scenario': parts[0],
                    'algo': parts[1],
                    'mode': parts[2],
                    'api_model': parts[3].replace('api', ''),
                }

        algo_name = meta.get('algo')
        is_baseline = (algo_name == 'baseline')
        is_comparison = (algo_name == 'comparison')
        if is_baseline and (not include_baselines):
            continue
        if is_comparison and (not include_comparison):
            continue

        run_mode = meta.get('mode')
        if not is_baseline:
            if include_base:
                if not (run_mode == mode or run_mode == 'BASE'):
                    continue
            else:
                if run_mode != mode:
                    continue

        if not is_baseline:
            if model_name and (meta.get('model_name') or '').split('/')[-1] != model_name.split('/')[-1]:
                continue
            if api_model and meta.get('api_model') != api_model:
                continue

        configured_iters = meta.get('max_iters')
        if configured_iters is None:
            configured_iters = _iters_from_name(os.path.basename(fp))
        if isinstance(max_iters, int) and max_iters > 0 and configured_iters is not None and configured_iters != max_iters:
            continue

        opt = payload.get('optimization_results') or {}
        sol_indices = _extract_solution_indices(opt)
        if not sol_indices:
            continue
        method = _method_label(meta)

        seed_value = (
            meta.get('seed')
            or meta.get('random_seed')
            or meta.get('seed_value')
            or payload.get('seed')
        )
        if seed_value is None:
            seed_match = re.search(r"__seed([^_]+)__", os.path.basename(fp))
            if seed_match:
                seed_value = seed_match.group(1)
        seed_id = str(seed_value) if seed_value is not None else os.path.basename(fp)

        if allowed_by_method is not None:
            method_key = str(method)
            allowed = allowed_by_method.get(method_key)
            if allowed is not None and seed_id not in allowed:
                continue

        for idx, sol_id in enumerate(sol_indices, start=1):
            if sol_id is None:
                rank_val = unranked_norm
            else:
                raw_rank = float(rank_lookup.get(int(sol_id), unranked_avg))
                rank_val = raw_rank / normalize_by
            rows.append({
                'filename': os.path.basename(fp),
                'iteration': idx,
                'method': method,
                'scenario': meta.get('scenario') or scenario,
                'mode': meta.get('mode'),
                'api_model': meta.get('api_model'),
                'model_name': (meta.get('model_name') or ''),
                'weighted_rank': rank_val,
            })

    df = pd.DataFrame(rows)
    if not df.empty and 'weighted_rank' in df.columns:
        try:
            df['weighted_rank'] = pd.to_numeric(df['weighted_rank'], errors='coerce')
        except Exception:
            pass
    return df


def plot_human_rank_convergence(
    df: pd.DataFrame,
    title: str,
    include_caption: bool = True,
    include_legend: bool = True,
):
    if df.empty:
        print("No data to plot.")
        return
    grouped = _group_convergence_stats_for(df, 'weighted_rank')
    scen, md = _infer_scenario_mode_from_df(df)
    effective_title = title or _format_dataset_title(scen, md, prefix='Human Rank')
    fig, ax = _render_convergence_plot_generic(
        grouped=grouped,
        title=effective_title,
        y_label='Normalized Average Rank (mean +/- 2SE)',
        scenario=scen,
        mode=md,
        draw_ref=False,
        include_caption=include_caption,
        include_legend=include_legend,
    )
    plt.show()


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Convergence plots using metric trajectories and YAML weights')
    ap.add_argument('--scenario', required=True, help='Scenario name (e.g., headphones, exam, cancer)')
    ap.add_argument('--mode', required=True, help='Mode name from scenario YAML (e.g., VALUE, PREMIUM)')
    ap.add_argument('--model-name', help='Filter by model name (exact or suffix after "/")')
    ap.add_argument('--api-model', help='Filter by API model/provider (e.g., groq, local)')
    ap.add_argument('--max-iters', type=int, help='Number of iterations')
    ap.add_argument('--reps', type=int, help='Maximum number of seeds per method to include')
    ap.add_argument('--include-baselines', action='store_true', help='Include baseline runs (random, zscore-avg) in plots')
    ap.add_argument('--include-comparison', action='store_true', help='Include comparison algorithm runs in plots')
    ap.add_argument('--exclude-base', action='store_true', help='Exclude BASE mode runs from plots')
    ap.add_argument('--human-rank', action='store_true', help='Plot human weighted rank instead of hidden ground truth')
    ap.add_argument('--no-caption', action='store_true', help='Omit the figure caption/supertitle when rendering plots')
    ap.add_argument('--no-legend', action='store_true', help='Hide the legend from the figure')
    ap.add_argument('--out-dir', help='Output directory to save plots (absolute or project-relative)')
    args = ap.parse_args()

    include_caption = not bool(args.no_caption)
    include_legend = not bool(args.no_legend)

    if args.human_rank:
        df = load_human_rank_series(
            args.scenario,
            args.mode,
            model_name=args.model_name,
            api_model=args.api_model,
            max_iters=args.max_iters,
            reps=args.reps,
            include_baselines=bool(args.include_baselines),
            include_comparison=bool(args.include_comparison),
            include_base=not bool(args.exclude_base),
        )
    else:
        df = load_gt_series(
            args.scenario,
            args.mode,
            model_name=args.model_name,
            api_model=args.api_model,
            max_iters=args.max_iters,
            reps=args.reps,
            include_baselines=bool(args.include_baselines),
            include_comparison=bool(args.include_comparison),
            include_base=not bool(args.exclude_base),
        )

    # title = f"{args.scenario} · {args.mode}"
    # if args.model_name:
    #     title += f" · {args.model_name.split('/')[-1]}"
    # if args.api_model:
    #     title += f" · api={args.api_model}"
    # if args.max_iters:
    #     title += f" · iters={args.max_iters}"
    # if args.reps:
    #     title += f" · reps={args.reps}"
    # if args.human_rank:
    #     title = "Human Rank · " + title

    subtitle_parts: list[str] = []
    # if args.mode:
    #     subtitle_parts.append(f"mode={args.mode}")
    # if args.model_name:
    #     subtitle_parts.append(args.model_name.split('/')[-1])
    # if args.api_model:
    #     subtitle_parts.append(f"api={args.api_model}")
    # if args.max_iters:
    #     subtitle_parts.append(f"iters={args.max_iters}")
    # if args.reps:
    #     subtitle_parts.append(f"reps={args.reps}")

    title = _format_dataset_title(
        args.scenario,
        args.mode,
        extras=subtitle_parts,
        prefix='Human Rank' if args.human_rank else None,
    )

    # Note: title shows requested reps only; actual included reps may vary by method

    if df.empty:
        print("No runs found that match filters.")
        return

    if args.human_rank:
        plot_human_rank_convergence(
            df,
            title,
            include_caption=include_caption,
            include_legend=include_legend,
        )
    else:
        plot_convergence(
            df,
            title,
            scenario=args.scenario,
            mode=args.mode,
            include_caption=include_caption,
            include_legend=include_legend,
        )
    fig = plt.gcf()

    # Determine output path
    if getattr(args, 'out_dir', None):
        base_dir = Path(args.out_dir)
        if not base_dir.is_absolute():
            # Resolve relative to project root
            base_dir = Path(__file__).resolve().parents[1] / base_dir
    else:
        base_dir = Path(__file__).resolve().parent / "plots"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_short = (args.model_name or "ALL").split("/")[-1]
    api_part = args.api_model or "ALLAPIs"
    suffix = "HUMANRANK" if args.human_rank else "GT"
    out_path = base_dir / f"{args.scenario}__{args.mode}__{api_part}__{model_short}__{suffix}.pdf"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # High-quality output for publication (DPI 300, tight layout)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved plot to {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
