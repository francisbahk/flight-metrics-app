## Post-Analysis Guide

This directory collects utilities for inspecting LISTEN experiment outputs. The most common follow-ups are generating convergence plots and running the difficulty analysis sweep. Below is a quick reference for both workflows.

## Requirements
- Activate the `listen` conda environment used for experiments (`module load mamba && mamba activate listen`).
- Ensure experiment outputs are written under `outputs/` with the expected naming convention (`<scenario>__<algo>__...`).

## Generating Plots
### Quick-start scripts
- `post_analysis/generate_selected_plots_groq.sh` renders the curated Lllama set (flight00/01 structured, flight02, exam, headphones).
- `post_analysis/generate_selected_plots_gemini.sh` does the same for Gemini runs.
- `post_analysis/generate_selected_plots_groq_ablations.sh` focuses on Lllama ablation comparisons.
- `post_analysis/generate_selected_plots_gemini_ablations.sh` mirrors the ablation plots for Gemini.

Run any script from the repo root:

```bash
bash post_analysis/generate_selected_plots_groq.sh
```

Each script calls `python -m post_analysis.plotting` with preset flags (`--include-baselines`, `--human-rank`, `--exclude-base`, etc.). Adjust or duplicate the script if you need a different scenario/mode combination.

### Custom plotting
To tweak arguments manually, call the module directly. For example:

```bash
python -m post_analysis.plotting \
  --scenario flight00 \
  --mode Complicated_structured \
  --api-model groq \
  --max-iters 25 \
  --reps 50 \
  --include-baselines \
  --human-rank
```

Outputs default to `post_analysis/plots/` (with `_compare` or `_ablations` subdirectories when relevant). Remove `--human-rank` or add `--exclude-baselines` as needed.

## Difficulty Analysis
- The canonical entry point is `post_analysis/run_difficulty_analysis.sh`, which iterates over structured flight scenarios, headphones modes, and exam registrar runs.
- Each invocation of `difficulty_analysis.py` writes a CSV row to `post_analysis/difficulty_results_<timestamp>.csv`.

Run the sweep from the repo root:

```bash
bash post_analysis/run_difficulty_analysis.sh
```

To target a single scenario/mode pair interactively:

```bash
python post_analysis/difficulty_analysis.py \
  --scenario flight01 \
  --mode Complicated_structured \
  --n-samples 1000
```

The `--n-samples` flag controls how many synthetic completions to draw when estimating difficulty. Increase it for smoother estimates at the cost of runtime.

## Tips
- All scripts assume results already exist; re-run the corresponding experiment launcher if plots are empty or the difficulty script cannot find data.
- Plotting scripts can be parallelized by splitting the scenario list if needed.
- Keep track of generated CSV and PDF files under `post_analysis/plots*/` and archive them alongside reports when sharing results.
