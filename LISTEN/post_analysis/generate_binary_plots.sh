#!/bin/bash
# Generate convergence plots for binary scenarios/modes across key providers.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MAX_ITERS=${MAX_ITERS:-25}
REPS=${REPS:-50}

SCENARIO_MODE_PAIRS=(
  "flight00_binary:Complicated_structured"
  "flight01_binary:Complicated_structured"
  "headphones_binary:STUDENT_HARD"
)

APIS=(
  gemini
  groq
)

run_plot_pair() {
  local args=("$@")
  python -m post_analysis.plotting "${args[@]}"
  python -m post_analysis.plotting "${args[@]}" --human-rank
}

for entry in "${SCENARIO_MODE_PAIRS[@]}"; do
  IFS=":" read -r scenario mode <<<"${entry}"
  for api in "${APIS[@]}"; do
    echo "Generating plots for ${scenario}/${mode} via ${api} (iters=${MAX_ITERS}, reps=${REPS})"
    base_args=(
      --scenario "${scenario}"
      --mode "${mode}"
      --api-model "${api}"
      --max-iters "${MAX_ITERS}"
      --reps "${REPS}"
      --include-baselines
      --exclude-base
    )
    run_plot_pair "${base_args[@]}" "$@"
  done
  echo
  echo "------------------------------------------------------------"
  echo
  sleep 1
done
