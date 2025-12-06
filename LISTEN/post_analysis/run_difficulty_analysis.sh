#!/bin/bash

# Difficulty Analysis Runner Script
# This script runs difficulty analysis for all scenarios and modes with human solutions

echo "Starting difficulty analysis for all scenarios..."
echo "================================================"

# Set parameters
N_SAMPLES=1000
OUTPUT_FILE="post_analysis/difficulty_results_$(date +%Y%m%d_%H%M%S).csv"

echo "Running individual scenario analyses for reference..."
echo "====================================================="

echo "1. flight00-Complicated:"
python post_analysis/difficulty_analysis.py --scenario flight00 --mode Complicated_structured --n-samples $N_SAMPLES

echo "1. flight00-Complicated binary:"
python post_analysis/difficulty_analysis.py --scenario flight00_binary --mode Complicated_structured --n-samples $N_SAMPLES


echo "2. flight01-Complicated:"
python post_analysis/difficulty_analysis.py --scenario flight01 --mode Complicated_structured --n-samples $N_SAMPLES

echo "2. flight01-Complicated binary:"
python post_analysis/difficulty_analysis.py --scenario flight01_binary --mode Complicated_structured --n-samples $N_SAMPLES


echo "1. flight00-Complicated no hard:"
python post_analysis/difficulty_analysis.py --scenario flight00 --mode Complicated_structured_no_hard --n-samples $N_SAMPLES

echo "2. flight01-Complicated no hard:"
python post_analysis/difficulty_analysis.py --scenario flight01 --mode Complicated_structured_no_hard --n-samples $N_SAMPLES


echo "3. flight02-Complicated:"
python post_analysis/difficulty_analysis.py --scenario flight02 --mode Complicated --n-samples $N_SAMPLES

echo "4. headphones-STUDENT:"
python post_analysis/difficulty_analysis.py --scenario headphones --mode STUDENT --n-samples $N_SAMPLES

echo "4. headphones-STUDENT_HARD:"
python post_analysis/difficulty_analysis.py --scenario headphones --mode STUDENT_HARD --n-samples $N_SAMPLES

echo "4. headphones-STUDENT_HARD binary:"
python post_analysis/difficulty_analysis.py --scenario headphones_binary --mode STUDENT_HARD --n-samples $N_SAMPLES

echo "5. exam-REGISTRAR:"
python post_analysis/difficulty_analysis.py --scenario exam --mode REGISTRAR --n-samples $N_SAMPLES

echo ""
echo "All analyses complete!"
echo "Summary results are in: $OUTPUT_FILE"
