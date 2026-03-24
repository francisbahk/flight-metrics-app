#!/usr/bin/env python3
"""
Load filled-in seed prompts from seed_prompts.md into the database.

Usage:
    python load_seeds.py                      # uses seed_prompts.md in this folder
    python load_seeds.py path/to/file.md      # uses a specific file

For each slot in the markdown:
  - Reads the Prolific ID
  - Looks up that participant in the DB to get their prompt text and flights
  - Inserts a SeedPrompt row (skips if already loaded)
"""
import sys
import os
import re

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db import SessionLocal, Participant, load_seed_prompt


def parse_seed_prompts_md(filepath: str) -> list:
    """Parse the filled-in markdown and return list of (slot_number, prolific_id)."""
    with open(filepath) as f:
        content = f.read()

    # Split on "## Slot N" headers; first chunk is the file header
    sections = re.split(r"^## Slot (\d+)", content, flags=re.MULTILINE)
    # sections = [header, "1", slot1_body, "2", slot2_body, ...]

    results = []
    i = 1
    while i + 1 < len(sections):
        slot_num = int(sections[i])
        body = sections[i + 1]

        # Extract the line after "**Prolific ID:**"
        pid_match = re.search(r"\*\*Prolific ID:\*\*\s*\n+(.*)", body)
        prolific_id = pid_match.group(1).strip() if pid_match else ""

        # Skip blank or placeholder slots
        if prolific_id and not prolific_id.startswith("<!--") and prolific_id != "":
            results.append((slot_num, prolific_id))

        i += 2

    return results


def load_seeds(filepath: str):
    slots = parse_seed_prompts_md(filepath)
    print(f"Found {len(slots)} filled slot(s) in {filepath}\n")

    db = SessionLocal()
    loaded = 0
    skipped = 0
    errors = []

    try:
        for slot_num, prolific_id in slots:
            participant = db.query(Participant).filter_by(prolific_id=prolific_id).first()

            if not participant:
                errors.append(f"Slot {slot_num}: Prolific ID '{prolific_id}' not found in DB.")
                continue

            if not participant.all_flights_json:
                errors.append(f"Slot {slot_num}: No flights stored for '{prolific_id}'.")
                continue

            if not participant.prompt:
                errors.append(f"Slot {slot_num}: No prompt text stored for '{prolific_id}'.")
                continue

            inserted = load_seed_prompt(
                slot_number=slot_num,
                prolific_id=prolific_id,
                prompt_text=participant.prompt,
                flights_json=participant.all_flights_json,
            )

            if inserted:
                loaded += 1
                preview = participant.prompt[:70].replace("\n", " ")
                print(f"  [OK] Slot {slot_num}: {prolific_id} — \"{preview}...\"")
            else:
                skipped += 1
                print(f"  [SKIP] Slot {slot_num}: {prolific_id} already in DB.")
    finally:
        db.close()

    print(f"\nDone. Loaded: {loaded}  Skipped: {skipped}  Errors: {len(errors)}")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_prompts.md")
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(filepath):
        print(f"Error: file not found: {filepath}")
        sys.exit(1)

    load_seeds(filepath)
