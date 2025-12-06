from __future__ import annotations
import os, json, argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# Import provider clients lazily inside factory to avoid heavy globals
from Experiment import SolutionBatchExperimentDueling, build_comparison_plot_data
from perfect import SimpleUtilityPreferenceClient
from prompt import ComparisonPromptAdapter
from prompt_manager import PromptManager
from tournament import SolutionTournamentExperiment

from local_vllm_client import LocalVLLMPreferenceClient
from baseline import build_baseline_history

try:
    import yaml
except ImportError:
    yaml = None

# Minimal .env loader (no external dependency)
def _load_env_file():
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Do not override explicitly provided environment variables
            os.environ.setdefault(key, value)
    except Exception as e:
        print(f"[WARN] Failed to load .env: {e}")

def _load_global_defaults() -> dict:
    cfg_dir = Path(__file__).resolve().parent / "configs"
    cfg_path = cfg_dir / "config.yml"
    if cfg_path.exists():
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f) if yaml else {}
    else:
        raise ValueError(f"Config file {cfg_path} not found")

def _available_scenarios() -> list:
    cfg_dir = Path(__file__).resolve().parent / "configs"
    names = []
    for p in cfg_dir.glob("*.y*ml"):
        if p.name in ("config.yml", "config.yaml"):
            continue
        names.append(p.stem)
    return sorted(list(set(names)))


def _find_scenario_yaml(scenario: str) -> Path:
    cfg_dir = Path(__file__).resolve().parent / "configs"
    # Allow passing name with or without extension
    candidates = []
    if scenario.endswith((".yaml", ".yml")):
        candidates.append(cfg_dir / scenario)
    else:
        candidates.append(cfg_dir / f"{scenario}.yaml")
        candidates.append(cfg_dir / f"{scenario}.yml")

    for cand in candidates:
        if cand.exists():
            return cand

    avail = ", ".join(_available_scenarios())
    raise ValueError(f"Scenario '{scenario}' not found under configs/. Available: [{avail}]")


def _load_yaml(path: str) -> dict:
    if not yaml:
        raise RuntimeError("PyYAML not installed; `pip install pyyaml`.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Run LISTEN experiments. Required named arguments: "
            "--scenario and --algo. Provide --mode when the scenario does not "
            "define a default mode."
        )
    )
    required = ap.add_argument_group("required named arguments")
    required.add_argument("--scenario", required=True)
    required.add_argument("--algo", choices=["comparison", "utility", "tournament", "baseline"], required=True)
    optional = ap.add_argument_group("optional arguments")
    optional.add_argument(
        "--max-iters",
        type=int,
        help="Total number of optimization iterations shared by both algorithms",
    )
    optional.add_argument("--api-model")
    optional.add_argument("--model-name")
    optional.add_argument(
        "--mode",
        help="Scenario mode; required when the scenario config lacks default_mode",
    )
    # comparison optimizer knobs (single replication only)
    optional.add_argument("--model-type", default="logistic", choices=['logistic','gp'])
    optional.add_argument("--acq", default="eubo", choices=['eubo','random'])
    optional.add_argument("--use-history", action="store_true")
    optional.add_argument("--comparison-batch-size", type=int, help="Batch size for comparison algorithm")

    # tournament optimizer knobs
    optional.add_argument("--tournament-batch-size", type=int, help="Batch size for tournament algorithm")

    # shared knobs for comparison and tournament
    optional.add_argument("--reasoning", action="store_true", help="Ask LLM to provide reasoning before choosing (comparison & tournament only)")

    # baseline knobs (no LLM)
    optional.add_argument("--baseline-strategy", choices=["random", "zscore-avg"], default="random")

    # utility optimizer knobs
    optional.add_argument("--utility-outdir")
    # reproducibility
    optional.add_argument("--seed", type=int, default=42)
    optional.add_argument("--use-utility-sim", action="store_true")
    # replications and retries
    optional.add_argument("--reps", type=int, default=1, help="Number of replications per configuration")
    optional.add_argument("--retries", type=int, default=3, help="Retries per replication on failure")
    args = ap.parse_args()

    max_iters = getattr(args, "max_iters", None)
    args.n_batches = max_iters
    args.utility_iters = max_iters

    return args


def load_config(scenario: str) -> dict:
    cfg = _load_global_defaults()
    scenario_path = _find_scenario_yaml(scenario)
    override = _load_yaml(str(scenario_path))
    cfg.update({k: v for k, v in override.items() if k not in ("model_configs",)})
    if "model_configs" in override:
        cfg["model_configs"] = {**cfg["model_configs"], **override["model_configs"]}
    for req in ["metric_columns", "data_csv", "tag"]:
        if req not in cfg:
            raise ValueError(f"Scenario config missing '{req}'")
    if "modes" not in cfg and "utility_prompt_text" not in cfg:
        raise ValueError("Provide either 'modes' with 'default_mode' or 'utility_prompt_text' in scenario config")
    if "prompts" not in cfg:
        raise ValueError("Scenario config must define a 'prompts' section with all required templates.")
    # Optional: metric_signs mapping for baseline ranking (-1 smaller-better, +1 larger-better, 0 ignore)
    if "metric_signs" in cfg and not isinstance(cfg.get("metric_signs"), dict):
        raise ValueError("If provided, metric_signs must be a mapping of metric -> {-1,0,1}")
    return cfg


def apply_overrides(cfg: dict, args):
    # Dueling single-run fields
    if args.model_type: cfg["model_type"] = args.model_type
    if args.acq: cfg["acq"] = args.acq
    cfg["use_history"] = bool(args.use_history)
    if args.comparison_batch_size is not None: cfg["comparison_batch_size"] = args.comparison_batch_size
    if args.tournament_batch_size is not None: cfg["tournament_batch_size"] = args.tournament_batch_size
    if args.n_batches is not None: cfg["n_batches"] = args.n_batches
    if args.api_model: cfg["api_model"] = args.api_model
    if args.model_name:
        am = cfg["api_model"]
        if am in cfg["model_configs"]:
            cfg["model_configs"][am]["model_name"] = args.model_name
    if args.use_utility_sim: cfg["use_utility_sim"] = True
    if args.seed is not None: cfg["seed"] = args.seed
    # Shared knobs for comparison and tournament
    if hasattr(args, 'reasoning'): cfg["reasoning"] = bool(args.reasoning)


def resolve_mode(cfg: dict, mode_cli: str | None):
    modes = cfg.get("modes")
    selected = None
    if modes:
        if mode_cli:
            selected = mode_cli
        else:
            selected = cfg.get("mode") or cfg.get("default_mode")
        if not selected:
            raise ValueError("Mode not specified; pass --mode or set default_mode in config")
        if selected not in modes:
            raise ValueError(f"Mode '{selected}' not found in scenario modes: {list(modes.keys())}")
        mode_def = modes[selected] or {}
        # rename this to policy guidance or whatever
        # set prompt
        prompt = mode_def.get("prompt") or mode_def.get("utility_prompt_text")
        if prompt:
            cfg["utility_prompt_text"] = prompt
        # set weights for utility sim
        cfg["mode_weights"] = mode_def.get("weights")
        cfg["mode"] = selected
    else:
        cfg["mode"] = mode_cli or cfg.get("mode")


def single_combo(cfg: dict):
    return dict(
        model_type=cfg.get("model_type", "logistic"),
        run_idx=0,
        prompt_idx=0,
        use_history=cfg.get("use_history", False),
        acq_func=cfg.get("acq", "eubo"),
    )


def build_output_filename(
    scenario: str,
    algo: str,
    mode_name: str,
    api_model: str,
    model_name_short: str,
    seed: int | None,
    run_stamp: str,
    max_iters: int,
    comparison_settings: dict | None = None,
):
    parts = [
        f"{scenario}__{algo}__{mode_name}",
        f"api{api_model}",
        f"llm{model_name_short}",
        f"iters{max_iters}",
        f"seed{seed if seed is not None else 'NA'}",
    ]
    if comparison_settings:
        mt = comparison_settings.get("model_type")
        acq = comparison_settings.get("acq")
        bs = comparison_settings.get("batch_size")
        hist = comparison_settings.get("use_history")
        if mt is not None:
            parts.append(f"model{mt}")
        if acq is not None:
            parts.append(f"acq{acq}")
        if bs is not None:
            parts.append(f"bs{bs}")
        if hist is not None:
            parts.append(f"hist{1 if hist else 0}")
    return "__".join(parts) + ".json"


def create_client(api_model: str, model_configs: dict):
    """Unified factory for both comparison and reasoning clients."""
    config = dict(model_configs[api_model])

    if api_model == "local":
        # Merge overrides and build LocalVLLMPreferenceClient
        local_config = dict(config)

        available_models = local_config.get("available_models") or {}
        default_model_name = local_config.get("default_model")
        requested_model_name = local_config.get("model_name")
        selected_model_name = requested_model_name or default_model_name

        if available_models:
            if not selected_model_name:
                raise ValueError(
                    "Local strategy configured with available_models but no model_name/default_model set"
                )
            if selected_model_name not in available_models:
                raise ValueError(
                    f"Local model '{selected_model_name}' not found. Options: {list(available_models.keys())}"
                )
            selected_overrides = dict(available_models[selected_model_name] or {})
            local_config.update(selected_overrides)
            local_config["model_name"] = selected_model_name

        # Clean helper keys so they don't leak into kwargs
        local_config.pop("available_models", None)
        local_config.pop("default_model", None)

        # Expand any paths in model_id
        if "model_id" in local_config and local_config["model_id"]:
            local_config["model_id"] = os.path.expandvars(os.path.expanduser(local_config["model_id"]))

        strategy = local_config.get("strategy", "vllm")

        if strategy != "vllm":
            raise ValueError(
                f"Unsupported local strategy '{strategy}'. Expected 'vllm'."
            )

        model_id = local_config.get("model_id")
        if not model_id:
            raise ValueError("Local vLLM strategy requires 'model_id' in configuration")

        sampling_kwargs = dict(local_config.get("sampling_kwargs") or {})
        # Map max_output_tokens -> max_tokens for underlying client
        max_out = (
            local_config.get("max_output_tokens")
            if local_config.get("max_output_tokens") is not None
            else local_config.get("max_tokens")
        )
        for key in ("temperature", "top_p", "stop", "seed"):
            if local_config.get(key) is not None:
                sampling_kwargs[key] = local_config[key]
        if max_out is not None:
            sampling_kwargs["max_tokens"] = max_out
        sampling_kwargs = {k: v for k, v in sampling_kwargs.items() if v is not None}

        client_kwargs = {
            "model_id": model_id,
            "tensor_parallel_size": local_config.get("tensor_parallel_size", 1),
            "max_model_len": local_config.get("max_model_len", local_config.get("max_len", 8192)),
            "dtype": local_config.get("dtype", "auto"),
            "default_temperature": local_config.get("temperature", 0.2),
            "default_top_p": local_config.get("top_p", 0.95),
            "default_max_new_tokens": max_out if max_out is not None else 8192,
            "default_seed": local_config.get("seed"),
            "default_stop_sequences": local_config.get("stop"),
        }
        client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}

        llm_kwargs = local_config.get("llm_kwargs") or {}

        return LocalVLLMPreferenceClient(
            **client_kwargs,
            sampling_kwargs=sampling_kwargs,
            llm_kwargs=llm_kwargs,
        )

    if api_model == "groq":
        from groq_client import FreeLLMPreferenceClient as GroqClient

        api_key_env = config.get("api_key_env", "GROQ_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing GROQ API key in environment variable '{api_key_env}'")
        max_out = (
            config.get("max_output_tokens")
            if config.get("max_output_tokens") is not None
            else config.get("max_tokens")
        )
        client = GroqClient(
            api_key=api_key,
            model_name=config.get("model_name", "llama-3.3-70b-versatile"),
            max_tokens=int(max_out) if max_out is not None else 8192,
            rate_limit_delay=config.get("rate_limit_delay", 0.1),
        )
        # Apply optional sampling defaults
        if config.get("temperature") is not None:
            try:
                client.default_temperature = float(config.get("temperature"))
            except Exception:
                pass
        if config.get("top_p") is not None:
            try:
                client.default_top_p = float(config.get("top_p"))
            except Exception:
                pass
        if max_out is not None:
            try:
                client.default_max_new_tokens = int(max_out)
            except Exception:
                pass
        return client

    if api_model == "openai":
        from openai_client import FreeLLMPreferenceClient as OpenAIClient

        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing OPENAI API key in environment variable '{api_key_env}'")
        max_out = (
            config.get("max_output_tokens")
            if config.get("max_output_tokens") is not None
            else config.get("max_tokens")
        )
        client = OpenAIClient(
            api_key=api_key,
            model_name=config.get("model_name", "gpt-4o"),
            max_tokens=int(max_out) if max_out is not None else 8192,
            rate_limit_delay=config.get("rate_limit_delay", 0.1),
        )
        # Apply optional sampling defaults
        if config.get("temperature") is not None:
            try:
                client.default_temperature = float(config.get("temperature"))
            except Exception:
                pass
        if config.get("top_p") is not None:
            try:
                client.default_top_p = float(config.get("top_p"))
            except Exception:
                pass
        if max_out is not None:
            try:
                client.default_max_new_tokens = int(max_out)
            except Exception:
                pass
        return client

    if api_model == "gemini":
        from gemini import FreeLLMPreferenceClient as GeminiClient

        api_key_env = config.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        max_out = (
            config.get("max_output_tokens")
            if config.get("max_output_tokens") is not None
            else config.get("max_tokens")
        )
        client = GeminiClient(
            api_key=api_key,
            model_name=config.get("model_name", "gemini-2.5-flash"),
            max_tokens=int(max_out) if max_out is not None else 8192,
            rate_limit_delay=config.get("rate_limit_delay", 0.1),
        )
        # Apply optional sampling defaults
        if config.get("temperature") is not None:
            try:
                client.default_temperature = float(config.get("temperature"))
            except Exception:
                pass
        if config.get("top_p") is not None:
            try:
                client.default_top_p = float(config.get("top_p"))
            except Exception:
                pass
        if max_out is not None:
            try:
                client.default_max_new_tokens = int(max_out)
            except Exception:
                pass
        return client

    raise ValueError(f"Unsupported api_model: {api_model}")


def run_comparison(args, config, run_id, run_stamp, log_dir, model_name_short):
    data_path = Path(__file__).resolve().parent / config["data_csv"]
    df = pd.read_csv(data_path).dropna(subset=config["metric_columns"])

    if config["use_utility_sim"]:
        client = SimpleUtilityPreferenceClient(weights=config.get("mode_weights") or {})
    else:
        client = create_client(config["api_model"], config["model_configs"])

    combo = single_combo(config)
    # If acquisition is random, disable model usage to avoid training/printing logistic/GP
    acq_is_random = (config.get('acq') == 'random')

    try:
        import random as _random
        import numpy as _np
        if config.get('seed') is not None:
            _random.seed(int(config['seed']))
            _np.random.seed(int(config['seed']))
    except Exception:
        pass

    base = (
        f"{args.scenario}__comparison__{config.get('mode', 'MODE')}__{model_name_short}__seed{config.get('seed', 'NA')}__"
        f"model{combo['model_type']}__acq{combo['acq_func']}__run{combo['run_idx']}__prompt{combo['prompt_idx']}__hist{combo['use_history']}"
    )
    history_filename = log_dir / f"{run_id}_{base}.csv"

    print("\n" + "=" * 64)
    print(f"RUN_ID: {run_id}")
    print(f"API Model: {config['api_model']} ({config['model_configs'].get(config['api_model'], {}).get('model_name', 'default')})")
    print(f"Scenario: {args.scenario} | Mode: {config.get('mode')} | Algo: comparison")
    model_label = combo['model_type'] if not acq_is_random else 'none'
    print(f"Starting: model={model_label}, acq={combo['acq_func']}, run={combo['run_idx']}, prompt={combo['prompt_idx']}, history={combo['use_history']}")
    print(f"History file: {history_filename}")
    print("=" * 64 + "\n")

    prompt_manager = PromptManager(config)
    base_prompt = prompt_manager.get_comparison_base(policy_guidance=config.get("utility_prompt_text", ""))
    prompt_template = ComparisonPromptAdapter(
        base_prompt=base_prompt,
        reasoning_history=combo['use_history'],
        metric_columns=config["metric_columns"],
        reasoning=config['reasoning'],
    )

    comparison_batch_size = config.get("comparison_batch_size", config.get("batch_size", 2))

    experiment = SolutionBatchExperimentDueling(
        solutions_df=df,
        metric_columns=config["metric_columns"],
        batch_size=comparison_batch_size,
        llm_client=client,
        prompt_template=prompt_template,
        LLM = not config["use_utility_sim"],
        model_type = combo['model_type'],
        acquisition=combo['acq_func'],
        random_seed=config.get('seed'),
        non_numeric_metrics=(config.get("non_numeric_metrics") or config.get("non_numerical_metrics"))
    )
    experiment.run_experiment(
        n_batches=config["n_batches"],
        history_file=str(history_filename),
        save_interval=5,
    )

    try:
        hist_json = str(history_filename).replace('.csv', '.json')
        with open(hist_json, 'r') as _f:
            _hist = json.load(_f)

        # Build compact plotting data via Experiment helper
        plot_data = build_comparison_plot_data(_hist, config.get('metric_columns', []))

        # Unified meta and filename
        outdir_path = Path(__file__).resolve().parent / "outputs" / args.scenario
        outdir_path.mkdir(parents=True, exist_ok=True)
        mode_name = config.get('mode') or 'MODE'
        max_iters = int(config.get("n_batches", 25))
        meta = {
            "scenario": args.scenario,
            "algo": "comparison",
            "mode": mode_name,
            "api_model": config.get("api_model"),
            "model_name": config.get("model_configs", {}).get(config.get("api_model"), {}).get("model_name"),
            "seed": config.get("seed"),
            "max_iters": max_iters,
            "batch_size": comparison_batch_size,
            "comparison_settings": {
                "model_type": combo.get("model_type"),
                "acq": combo.get("acq_func"),
                "batch_size": comparison_batch_size,
                "use_history": combo.get("use_history"),
            },
            "run_id": run_id,
            "run_stamp_utc": run_stamp,
        }

        out_name = build_output_filename(
            scenario=args.scenario,
            algo="comparison",
            mode_name=mode_name,
            api_model=config.get("api_model"),
            model_name_short=model_name_short,
            seed=config.get("seed"),
            run_stamp=run_stamp,
            max_iters=max_iters,
            comparison_settings={
                "model_type": combo.get("model_type"),
                "acq": combo.get("acq_func"),
                "batch_size": comparison_batch_size,
                "use_history": combo.get("use_history"),
            },
        )

        payload = {"meta": meta, 'optimization_results': plot_data}
        (outdir_path / out_name).write_text(json.dumps(payload, indent=2, allow_nan=False))
    except Exception as _e:
        print(f"Warning: failed to write plotting JSON: {_e}")


def run_utility(args, config, run_id, model_name_short, run_stamp, log_dir):
    from utility.iterative_utility_optimizer import IterativeUtilityOptimizer

    csv_path = str((Path(__file__).resolve().parent / config["data_csv"]).resolve())
    outdir_path = Path(__file__).resolve().parent / "outputs" / args.scenario
    outdir_path.mkdir(parents=True, exist_ok=True)
    print(f"[UTILITY] csv={csv_path} outdir={outdir_path}")

    reasoning_client = create_client(config["api_model"], config["model_configs"])
    prompt_manager = PromptManager(config)

    optimizer = IterativeUtilityOptimizer(
        csv_path=csv_path,
        llm_client=reasoning_client,
        policy_guidance=config.get("utility_prompt_text", ""),
        metric_columns=config.get("metric_columns"),
        initial_prompt_template=prompt_manager.get_utility_base(),
        refinement_prompt_template=prompt_manager.get_utility_refinement(),
        generation_seed=config.get("seed"),
        non_numeric_metrics=(config.get("non_numeric_metrics") or config.get("non_numerical_metrics")),
        numeric_metric_columns=config.get("numeric_metric_columns")
    )

    seed = config.get('seed')
    if seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(int(seed))
        _np.random.seed(int(seed))

    iterations = args.utility_iters or config.get("n_batches", 25)
    results = optimizer.run_optimization(num_iterations=iterations)

    # Unified meta and filename
    mode_name = config.get("mode") or "MODE"
    max_iters = int(iterations)
    meta = {
        "scenario": args.scenario,
        "algo": "utility",
        "mode": mode_name,
        "api_model": config.get("api_model"),
        "model_name": config.get("model_configs", {}).get(config.get("api_model"), {}).get("model_name"),
        "seed": seed,
        "max_iters": max_iters,
        "batch_size": None,  # utility optimizer doesn't use batch_size
        "run_id": run_id,
        "run_stamp_utc": run_stamp,
    }

    out_name = build_output_filename(
        scenario=args.scenario,
        algo="utility",
        mode_name=mode_name,
        api_model=config.get("api_model"),
        model_name_short=model_name_short,
        seed=seed,
        run_stamp=run_stamp,
        max_iters=max_iters,
        comparison_settings=None,
    )

    payload = {"meta": meta, "optimization_results": results}
    (outdir_path / out_name).write_text(json.dumps(payload, indent=2, default=str))

    optimizer.print_summary(results)
    return


def run_tournament(args, config, run_id, run_stamp, log_dir, model_name_short):
    data_path = Path(__file__).resolve().parent / config["data_csv"]
    df = pd.read_csv(data_path).dropna(subset=config["metric_columns"])  # solutions_df

    # Client: reuse same factory as comparison
    if config["use_utility_sim"]:
        client = SimpleUtilityPreferenceClient(weights=config.get("mode_weights") or {})
    else:
        client = create_client(config["api_model"], config["model_configs"])

    # Prompt: reuse comparison base prompt
    prompt_manager = PromptManager(config)
    base_prompt = prompt_manager.get_comparison_base(policy_guidance=config.get("utility_prompt_text", ""))
    prompt_template = ComparisonPromptAdapter(
        base_prompt=base_prompt,
        reasoning_history=config.get("use_history", False),
        metric_columns=config["metric_columns"],
        reasoning=config['reasoning'],
    )

    # Iterations for tournament: use n_batches for consistency
    iterations = int(config.get("n_batches", 25))
    batch_size = int(config.get("tournament_batch_size", config.get("batch_size", 50)))

    experiment = SolutionTournamentExperiment(
        solutions_df=df,
        metric_columns=config["metric_columns"],
        llm_client=client,
        prompt_template=prompt_template,
        batch_size=batch_size,
        iterations=iterations,
        random_seed=config.get('seed'),
    )
    results = experiment.run()

    # Unified meta and filename
    outdir_path = Path(__file__).resolve().parent / "outputs" / args.scenario
    outdir_path.mkdir(parents=True, exist_ok=True)
    mode_name = config.get('mode') or 'MODE'
    max_iters = iterations
    meta = {
        "scenario": args.scenario,
        "algo": "tournament",
        "mode": mode_name,
        "api_model": config.get("api_model"),
        "model_name": config.get("model_configs", {}).get(config.get("api_model"), {}).get("model_name"),
        "seed": config.get("seed"),
        "max_iters": max_iters,
        "batch_size": batch_size,
        "run_id": run_id,
        "run_stamp_utc": run_stamp,
    }

    out_name = build_output_filename(
        scenario=args.scenario,
        algo="tournament",
        mode_name=mode_name,
        api_model=config.get("api_model"),
        model_name_short=model_name_short,
        seed=config.get("seed"),
        run_stamp=run_stamp,
        max_iters=max_iters,
        comparison_settings=None,
    )

    payload = {"meta": meta, "optimization_results": results.get("plot_data")}
    (outdir_path / out_name).write_text(json.dumps(payload, indent=2, default=str))

    # Brief console summary
    print("\n" + "=" * 64)
    print(f"RUN_ID: {run_id}")
    print(f"API Model: {config['api_model']} ({config['model_configs'].get(config['api_model'], {}).get('model_name', 'default')})")
    print(f"Scenario: {args.scenario} | Mode: {mode_name} | Algo: tournament")
    print(f"Final winner idx: {results.get('final_winner_idx')}")
    print("=" * 64 + "\n")



def run_baseline(args, config, run_id, run_stamp, log_dir, model_name_short):
    data_path = Path(__file__).resolve().parent / config["data_csv"]
    df = pd.read_csv(data_path).dropna(subset=config["metric_columns"])  # solutions_df

    try:
        import random as _random
        import numpy as _np
        if config.get('seed') is not None:
            _random.seed(int(config['seed']))
            _np.random.seed(int(config['seed']))
    except Exception:
        pass

    strategy = getattr(args, "baseline_strategy", "random")
    n_batches = int(config.get("n_batches", 25))
    metric_signs = config.get("metric_signs")

    history = build_baseline_history(
        strategy=strategy,
        df=df,
        metric_columns=config["metric_columns"],
        metric_signs=metric_signs,
        n_batches=n_batches,
    )

    plot_data = build_comparison_plot_data(history, config.get('metric_columns', []))

    # Unified meta and filename
    outdir_path = Path(__file__).resolve().parent / "outputs" / args.scenario
    outdir_path.mkdir(parents=True, exist_ok=True)
    scenario_mode = config.get('mode')
    mode_name = strategy
    max_iters = n_batches
    meta = {
        "scenario": args.scenario,
        "algo": "baseline",
        "mode": scenario_mode,
        "api_model": None,
        "model_name": None,
        "seed": config.get("seed"),
        "max_iters": max_iters,
        "batch_size": None,
        "baseline_settings": {
            "strategy": strategy,
        },
        "run_id": run_id,
        "run_stamp_utc": run_stamp,
    }

    out_name = build_output_filename(
        scenario=args.scenario,
        algo="baseline",
        mode_name=mode_name,
        api_model="baseline",
        model_name_short="baseline",
        seed=config.get("seed"),
        run_stamp=run_stamp,
        max_iters=max_iters,
        comparison_settings=None,
    )

    payload = {"meta": meta, "optimization_results": plot_data}
    (outdir_path / out_name).write_text(json.dumps(payload, indent=2, default=str))

    # Brief console summary
    print("\n" + "=" * 64)
    print(f"RUN_ID: {run_id}")
    mode_display = scenario_mode if scenario_mode else "(none)"
    print(f"Scenario: {args.scenario} | Scenario mode: {mode_display} | Algo: baseline | Strategy: {strategy}")
    print(f"Final best idx: {plot_data.get('final_best_solution_idx')}")
    print("=" * 64 + "\n")

def write_manifest(log_dir, run_id, config):
    manifest = {
        "run_id": run_id,
        "config": config,
        "stamp_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    }
    (log_dir / f"{run_id}_run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    # Load .env first so env vars are available
    _load_env_file()
    args = parse_args()
    config = load_config(args.scenario)
    apply_overrides(config, args)
    resolve_mode(config, args.mode)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slurm_job = os.environ.get("SLURM_JOB_ID")
    current_dir = Path(__file__).resolve().parent
    log_dir = current_dir / config["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    model_name_short = config["model_configs"].get(config["api_model"], {}).get("model_name", config["api_model"])
    model_name_short = model_name_short.split("/")[-1] if "/" in model_name_short else model_name_short
    run_id_base = f"{args.scenario}_{config['api_model']}-{model_name_short}_{run_stamp}"
    if slurm_job:
        run_id_base += f"_job{slurm_job}"

    write_manifest(log_dir, run_id_base, config)

    base_seed = int(config.get("seed", 42))
    reps = int(getattr(args, "reps", 1) or 1)
    retries = int(getattr(args, "retries", 0) or 0)

    for rep_idx in range(reps):
        success = False
        for attempt in range(retries + 1):
            rep_seed = base_seed + rep_idx + attempt
            cfg = dict(config)
            cfg["seed"] = rep_seed
            run_id = f"{run_id_base}_rep{rep_idx}_seed{rep_seed}"
            try:
                if args.algo == "utility":
                    run_utility(args, cfg, run_id, model_name_short, run_stamp, log_dir)
                elif args.algo == "tournament":
                    run_tournament(args, cfg, run_id, run_stamp, log_dir, model_name_short)
                elif args.algo == "baseline":
                    run_baseline(args, cfg, run_id, run_stamp, log_dir, model_name_short)
                else:
                    run_comparison(args, cfg, run_id, run_stamp, log_dir, model_name_short)
                success = True
                break
            except Exception as e:
                print(f"[WARN] Rep {rep_idx} attempt {attempt} failed: {e}. Retrying with seed {rep_seed + 1}...")
                continue
        if not success:
            print(f"[ERROR] Rep {rep_idx} exhausted retries; skipping.")

