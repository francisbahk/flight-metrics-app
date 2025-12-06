#!/bin/bash

set -euo pipefail

REPS=${REPS:-20}
MAX_ITERS=${MAX_ITERS:-25}
BASE_SEED=${BASE_SEED:-42}
MODEL_NAME_GEMINI_DEFAULT=${MODEL_NAME_GEMINI_DEFAULT:-gemini-2.5-flash-lite}

SCENARIOS=(
  exam
  flight00
  flight01
  flight02
  headphones
)

MODES=(
  REGISTRAR
  Complicated_structured
  Complicated_structured
  Complicated
  STUDENT
)

ALGOS=(
  utility
  tournament
)

MODEL_NAME=${MODEL_NAME_GEMINI:-$MODEL_NAME_GEMINI_DEFAULT}

run_job() {
  local scenario=$1
  local mode=$2
  local algo=$3

  echo "Running: scenario=$scenario mode=$mode algo=$algo api=gemini model=$MODEL_NAME reps=$REPS iters=$MAX_ITERS"

  python -m main \
    --reps "$REPS" --max-iters "$MAX_ITERS" \
    --api-model gemini --model-name "$MODEL_NAME" --reasoning \
    --scenario "$scenario" --mode "$mode" \
    --algo "$algo" \
    --seed "$BASE_SEED"
}

if [[ $# -gt 0 ]]; then
  TASK_ID=$1
  if ! [[ $TASK_ID =~ ^[0-9]+$ ]]; then
    echo "Task ID must be an integer" >&2
    exit 1
  fi

  total_jobs=$((${#SCENARIOS[@]} * ${#ALGOS[@]}))
  if (( TASK_ID < 0 || TASK_ID >= total_jobs )); then
    echo "Task ID $TASK_ID out of range for $total_jobs jobs" >&2
    exit 1
  fi

  scenario_index=$((TASK_ID / ${#ALGOS[@]}))
  algo_index=$((TASK_ID % ${#ALGOS[@]}))

  run_job "${SCENARIOS[$scenario_index]}" "${MODES[$scenario_index]}" "${ALGOS[$algo_index]}"
else
  for idx in "${!SCENARIOS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      run_job "${SCENARIOS[$idx]}" "${MODES[$idx]}" "$algo"
    done
  done
fi

