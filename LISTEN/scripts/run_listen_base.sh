#!/bin/bash

set -euo pipefail

REPS=${REPS:-50}
MAX_ITERS=${MAX_ITERS:-25}
BASE_SEED=${BASE_SEED:-42}
MODEL_NAME_GROQ_DEFAULT=${MODEL_NAME_GROQ_DEFAULT:-llama-3.3-70b-versatile}
MODEL_NAME_GEMINI_DEFAULT=${MODEL_NAME_GEMINI_DEFAULT:-gemini-2.5-flash-lite}

SCENARIOS=(
  exam
  flight00
  flight01
  flight02
  headphones
)

MODE=BASE

APIS=(
  groq
  gemini
)

ALGOS=(
  utility
  tournament
)

run_job() {
  local scenario=$1
  local api=$2
  local algo=$3

  local model_name
  if [[ "$api" == "gemini" ]]; then
    model_name=${MODEL_NAME_GEMINI:-$MODEL_NAME_GEMINI_DEFAULT}
  else
    model_name=${MODEL_NAME_GROQ:-$MODEL_NAME_GROQ_DEFAULT}
  fi

  echo "Running: scenario=$scenario mode=$MODE algo=$algo api=$api model=$model_name reps=$REPS iters=$MAX_ITERS seed=$BASE_SEED"

  python -m main \
    --reps "$REPS" --max-iters "$MAX_ITERS" \
    --api-model "$api" --model-name "$model_name" --reasoning \
    --scenario "$scenario" --mode "$MODE" \
    --algo "$algo" \
    --seed "$BASE_SEED"
}

total_jobs=$((${#SCENARIOS[@]} * ${#APIS[@]} * ${#ALGOS[@]}))

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

  scenario_index=$((TASK_ID / (${#APIS[@]} * ${#ALGOS[@]})))
  remainder=$((TASK_ID % (${#APIS[@]} * ${#ALGOS[@]})))
  api_index=$((remainder / ${#ALGOS[@]}))
  algo_index=$((remainder % ${#ALGOS[@]}))

  run_job "${SCENARIOS[$scenario_index]}" "${APIS[$api_index]}" "${ALGOS[$algo_index]}"
else
  for scenario in "${SCENARIOS[@]}"; do
    for api in "${APIS[@]}"; do
      for algo in "${ALGOS[@]}"; do
        run_job "$scenario" "$api" "$algo"
      done
    done
  done
fi

