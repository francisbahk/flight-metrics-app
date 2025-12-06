#!/bin/bash
# Generate convergence plots for ablation comparisons (Gemini)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR_REL="post_analysis/plots_ablations"
OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR_REL}"
mkdir -p "${OUTPUT_DIR}"

run_plot() {
    local scenario="$1"
    local mode="$2"
    local api="$3"
    shift 3
    local extra_args=("$@")

    python -m post_analysis.plotting \
        --scenario "${scenario}" \
        --mode "${mode}" \
        --api-model "${api}" \
        --max-iters 25 \
        --reps 50 \
        --out-dir "${OUTPUT_DIR_REL}" \
        "${extra_args[@]}"

    local suffix="GT"
    for arg in "${extra_args[@]}"; do
        if [[ "${arg}" == "--human-rank" ]]; then
            suffix="HUMANRANK"
            break
        fi
    done

    local out_path="${OUTPUT_DIR}/${scenario}__${mode}__${api}__ALL__${suffix}.pdf"

    if [[ -f "${out_path}" ]]; then
        echo "Saved plot to ${out_path}"
    else
        echo "Warning: expected plot ${out_path} not found" >&2
    fi
}

# Headphones STUDENT - Gemini
run_plot "headphones" "STUDENT" "gemini"
run_plot "headphones" "STUDENT" "gemini" --human-rank

# Exam REGISTRAR - Gemini
run_plot "exam" "REGISTRAR" "gemini"
run_plot "exam" "REGISTRAR" "gemini" --human-rank

# Flight Frazier Complicated - Gemini
run_plot "flight02" "Complicated" "gemini"
run_plot "flight02" "Complicated" "gemini" --human-rank

# Flight00 Complicated - Gemini
run_plot "flight00" "Complicated_structured" "gemini"
run_plot "flight00" "Complicated_structured" "gemini" --human-rank

# Flight01 Complicated - Gemini
run_plot "flight01" "Complicated_structured" "gemini"
run_plot "flight01" "Complicated_structured" "gemini" --human-rank
