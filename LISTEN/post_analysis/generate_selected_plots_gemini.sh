#!/bin/bash
# Generate convergence plots for selected scenarios (Gemini only)

set -euo pipefail

# Change to repository root (directory that contains post_analysis/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# echo "Generating selected plots for headphones/STUDENT, exam/REGISTRAR, and flight02/Complicated using Gemini..."

# Headphones STUDENT - Gemini
python -m post_analysis.plotting \
    --scenario headphones \
    --mode STUDENT \
    --api-model gemini \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --exclude-base

python -m post_analysis.plotting \
    --scenario headphones \
    --mode STUDENT \
    --api-model gemini \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --human-rank \
    --exclude-base

# Exam REGISTRAR - Gemini
python -m post_analysis.plotting \
    --scenario exam \
    --mode REGISTRAR \
    --api-model gemini \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --exclude-base

python -m post_analysis.plotting \
    --scenario exam \
    --mode REGISTRAR \
    --api-model gemini \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --human-rank \
    --exclude-base

# Flight Frazier Complicated - Gemini
python -m post_analysis.plotting \
    --scenario flight02 \
    --mode Complicated \
    --api-model gemini \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --exclude-base

python -m post_analysis.plotting \
    --scenario flight02 \
    --mode Complicated \
    --api-model gemini \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --human-rank \
    --exclude-base

# Flight00 Complicated - Gemini
python -m post_analysis.plotting \
    --scenario flight00 \
    --mode Complicated_structured \
    --api-model gemini \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --exclude-base

python -m post_analysis.plotting \
    --scenario flight00 \
    --mode Complicated_structured \
    --api-model gemini \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --human-rank \
    --exclude-base

# Flight01 Complicated - Gemini
python -m post_analysis.plotting \
    --scenario flight01 \
    --mode Complicated_structured \
    --api-model gemini \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --exclude-base

python -m post_analysis.plotting \
    --scenario flight01 \
    --mode Complicated_structured \
    --api-model gemini \
    --max-iters 25 \
    --reps 50 \
    --include-baselines \
    --human-rank \
    --exclude-base
