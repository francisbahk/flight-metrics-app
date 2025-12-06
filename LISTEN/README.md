## LISTEN — Configure and Run

This guide shows how to configure and run the LISTEN experiments.

### 1) Create the environment

```bash
conda env create -f environment.yml
conda activate listen
```

### 2) Configure API Keys

Experiments are run against API providers (Groq for Llama models, and Gemini). You will need to provide API keys.

Providers and required environment variables:
- **groq**: `GROQ_API_KEY`
- **gemini**: `GEMINI_API_KEY`

The recommended way to provide these is to create a `.env` file in the project root:
```bash
GROQ_API_KEY="your_groq_key"
GEMINI_API_KEY="your_gemini_key"
```

The main script will automatically load these variables.

### 3) Configure Scenario and Model Defaults

The `configs/` folder contains two kinds of files:
- `configs/config.yml` holds global defaults—log directory, default API model, and provider-specific model settings (including local vLLM variants under `model_configs.local`).
- Scenario YAML files (e.g. `configs/exam.yaml`, `configs/flight00.yml`, `configs/headphones.yaml`) define dataset paths, metric semantics, and per-mode prompts used by the algorithms.

Review or customize these files before launching runs if you need different defaults or prompt text.

### 4) Run Experiments

#### Using Launcher Scripts (Recommended)

The `scripts/` directory contains bash launchers for running the main experiment sets. These are the recommended way to run experiments, as they ensure consistent settings.

- `scripts/run_listen_selected_llama.sh`: Runs main algorithms for all scenarios using Llama (via Groq).
- `scripts/run_listen_selected_gemini.sh`: Runs main algorithms for all scenarios using Gemini.
- `scripts/run_listen_base.sh`: Runs main algorithms for `BASE` mode across all scenarios and both APIs.
- `scripts/run_listen_headphones_student_hard.sh`: Runs main algorithms for the `headphones/STUDENT_HARD` case.
- `scripts/run_listen_baseline.sh`: Runs baseline strategies (`random`, `zscore-avg`) for all scenarios.

To run an entire set of jobs locally:
```bash
bash scripts/run_listen_selected_llama.sh
```

These scripts can also be used with a job scheduler (like SLURM) by passing a task ID, which will execute a single job from the matrix. For example, to run the first job:
```bash
bash scripts/run_listen_selected_llama.sh 0
```

#### Running Manually with `main.py`

For debugging or custom runs, you can invoke `main.py` directly.

Key flags:
```bash
python -m main \
  --scenario {exam|headphones|flight00|flight01|flight02} \
  --algo {utility|tournament|comparison} \
  --api-model {groq|gemini|local} \
  [--mode MODE] [--model-name MODEL_NAME] \
  [--reps REPS] [--max-iters ITERS] [--seed SEED]
```

- **Algorithms**: `utility` and `tournament` are the primary algorithms used in the paper. `comparison` is an experimental variant that has not been extensively tested yet.
- **API models**: using Groq (Llama) or Gemini is recommended to match the paper results.
- **Local models**: set `--api-model local` with a compatible vLLM deployment and configure `model_configs.local` in `configs/config.yml`.

Example:
```bash
python -m main \
  --scenario exam --algo utility \
  --api-model groq --model-name llama-3.3-70b-versatile \
  --reps 10 --max-iters 25
```

### 5) Outputs

- Run manifest and histories: written under `logs/`.
- Results for plotting/inspection: `outputs/<scenario>/...json`.

### 6) Troubleshooting

- “Missing <PROVIDER> API key”: set it in `.env` (recommended) or export the variable listed above.


