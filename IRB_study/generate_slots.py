#!/usr/bin/env python3
"""
Generate a markdown template with n seed prompt slots.

Usage:
    python generate_slots.py <n>

Example:
    python generate_slots.py 90   # creates seed_prompts.md with 90 slots

Each slot in the output file has fields for a Prolific ID and prompt text.
Fill them in, then run load_seeds.py to load into the database.
"""
import sys
import os
from datetime import datetime


def generate_slots(n: int, output_path: str = None) -> str:
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_prompts.md")

    lines = [
        "# Seed Prompts for Cross-Validation",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Slots: **{n}** prompts × 10 re-rankings each = **{n * 10}** total re-rankings",
        "",
        "**Instructions:**",
        "1. Fill in each slot below with the Prolific ID and prompt text of a participant",
        "   whose search you want to use as a cross-validation seed.",
        "2. Leave a slot's Prolific ID blank to skip it.",
        "3. Run `load_seeds.py` to load filled slots into the database.",
        "",
        "---",
        "",
    ]

    for i in range(1, n + 1):
        lines += [
            f"## Slot {i}",
            "",
            "**Prolific ID:**",
            "",
            "**Prompt Text:** *(for your reference — the app will pull the exact text from the DB)*",
            "",
            "---",
            "",
        ]

    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)

    print(f"Created {output_path} with {n} slots ({n * 10} total re-rankings).")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_slots.py <n>")
        print("Example: python generate_slots.py 90")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid integer.")
        sys.exit(1)

    if n <= 0:
        print("Error: n must be a positive integer.")
        sys.exit(1)

    generate_slots(n)
