#!/bin/bash

set -euo pipefail

SCENARIOS=(
  exam
  exam
  headphones
  headphones
  flight00
  flight00
  flight01
  flight01
  flight02
  flight02
)

STRATEGIES=(
  random
  zscore-avg
  random
  zscore-avg
  random
  zscore-avg
  random
  zscore-avg
  random
  zscore-avg
)

MAX_ITERS=${MAX_ITERS:-25}
REPS_RANDOM=${REPS_RANDOM:-20}
BASE_SEED=${BASE_SEED:-42}

run_job() {
  local scenario=$1
  local strategy=$2

  local cmd=(python -m main \
    --scenario "$scenario" \
    --algo baseline \
    --baseline-strategy "$strategy" \
    --max-iters "$MAX_ITERS" \
    --api-model baseline \
    --seed "$BASE_SEED")

  if [[ "$strategy" == "random" ]]; then
    cmd+=(--reps "$REPS_RANDOM")
  fi

  echo "Running baseline job: ${cmd[*]}"
  "${cmd[@]}"
}

total_jobs=${#SCENARIOS[@]}

if [[ $# -gt 0 ]]; then
  TASK_ID=$1
  if ! [[ $TASK_ID =~ ^[0-9]+$ ]]; then
    echo "Task ID must be an integer" >&2
    exit 1
  fi

  if (( TASK_ID < 0 || TASK_ID >= total_jobs )); then
    echo "Task ID $TASK_ID out of range for $total_jobs jobs" >&2
    exit 1
  fi

  run_job "${SCENARIOS[$TASK_ID]}" "${STRATEGIES[$TASK_ID]}"
else
  for idx in "${!SCENARIOS[@]}"; do
    run_job "${SCENARIOS[$idx]}" "${STRATEGIES[$idx]}"
  done
fi

