#!/bin/bash
# Generate convergence plots for selected scenarios (Groq only)

set -euo pipefail

# Change to repository root (directory that contains post_analysis/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Headphones STUDENT - Groq
python -m post_analysis.plotting \
    --scenario headphones \
    --mode STUDENT \
    --api-model groq \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --exclude-base

python -m post_analysis.plotting \
    --scenario headphones \
    --mode STUDENT \
    --api-model groq \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --human-rank \
    --exclude-base

# Exam REGISTRAR - Groq
python -m post_analysis.plotting \
    --scenario exam \
    --mode REGISTRAR \
    --api-model groq \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --exclude-base

python -m post_analysis.plotting \
    --scenario exam \
    --mode REGISTRAR \
    --api-model groq \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --human-rank \
    --exclude-base

# Flight Frazier Complicated - Groq
python -m post_analysis.plotting \
    --scenario flight02 \
    --mode Complicated \
    --api-model groq \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --exclude-base

python -m post_analysis.plotting \
    --scenario flight02 \
    --mode Complicated \
    --api-model groq \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --human-rank \
    --exclude-base

# Flight00 Complicated - Groq
python -m post_analysis.plotting \
    --scenario flight00 \
    --mode Complicated_structured \
    --api-model groq \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --exclude-base

python -m post_analysis.plotting \
    --scenario flight00 \
    --mode Complicated_structured \
    --api-model groq \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --human-rank \
    --exclude-base

# Flight01 Complicated - Groq
python -m post_analysis.plotting \
    --scenario flight01 \
    --mode Complicated_structured \
    --api-model groq \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --exclude-base

python -m post_analysis.plotting \
    --scenario flight01 \
    --mode Complicated_structured \
    --api-model groq \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --human-rank \
    --exclude-base
