#!/bin/bash
# Batch runner for compare_modes_combined.py across key scenarios and providers.
set -euo pipefail

MAX_ITERS=${MAX_ITERS:-25}
REPS=${REPS:-50}

SCENARIO_MODE_PAIRS=(
  "headphones:STUDENT:STUDENT_HARD"
  "flight00:Complicated_structured:Complicated_structured_no_hard"
  "flight01:Complicated_structured:Complicated_structured_no_hard"
)

APIS=(
  gemini
  groq
)

ALGOS=(
  utility
  tournament
)

for entry in "${SCENARIO_MODE_PAIRS[@]}"; do
  IFS=":" read -r scenario mode_a mode_b <<<"${entry}"
  for api in "${APIS[@]}"; do
    algo_args=()
    for algo in "${ALGOS[@]}"; do
      algo_args+=(--algo "${algo}:${api}")
    done

    echo "Comparing ${scenario} modes ${mode_a} vs ${mode_b} with API=${api} (max_iters=${MAX_ITERS}, reps=${REPS})"
    python -m post_analysis.compare_modes_combined \
      --scenario "${scenario}" \
      --modes "${mode_a}" "${mode_b}" \
      --max-iters "${MAX_ITERS}" \
      --reps "${REPS}" \
      "${algo_args[@]}" \
      "$@"
  done

  echo
  echo "------------------------------------------------------------"
  echo
  sleep 1
done
