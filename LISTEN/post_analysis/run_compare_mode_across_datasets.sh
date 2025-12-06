#!/bin/bash
# Compare identical modes across paired datasets.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MAX_ITERS=${MAX_ITERS:-25}
REPS=${REPS:-50}

PAIR_SPECS=(
  "flight00,flight00_binary,Complicated_structured"
  "flight01,flight01_binary,Complicated_structured"
  "headphones,headphones_binary,STUDENT_HARD"
)

APIS=(
  gemini
  groq
)

ALGOS=(
  utility
  tournament
)

for spec in "${PAIR_SPECS[@]}"; do
  for api in "${APIS[@]}"; do
    algo_args=()
    for algo in "${ALGOS[@]}"; do
      algo_args+=(--algo "${algo}:${api}")
    done
    echo "Comparing datasets ${spec} using API=${api}"
    python -m post_analysis.compare_mode_across_datasets \
      --pair "${spec}" \
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
