#!/usr/bin/env python3
"""
Generate Stroop task datasets (Conditional and Serial) from OpenNeuro ds005237
Adapted from Prompt_Generation_Serial_Conditional.ipynb
"""

import os
import pandas as pd
import glob

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to the cloned ds005237 dataset
# You need to clone it first: git clone https://github.com/OpenNeuroDatasets/ds005237.git
BASE_PATH = "./ds005237"

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def read_tsv(path):
    """Read TSV file and return as list of dictionaries."""
    df = pd.read_csv(path, sep='\t')
    return df.to_dict(orient="records")

# ==============================================================================
# CONDITIONAL ASSOCIATIVE LEARNING (Participants 11-136)
# ==============================================================================

def make_trial_prompt_conditional(row):
    """Generate a story-like prompt for conditional task."""

    rt = row.get("response_time", "n/a")
    acc = row.get("accuracy_binarized", "n/a")
    stim_text = row.get("stimulus_text", "unknown")
    stim_color = row.get("stimulus_color", "unknown")
    trial_type = row.get("trial_type", "unknown")
    points = 1 if acc == 1.0 else 0

    return (
        f"You see the word '{stim_text}' written in '{stim_color}' color."
        f"You respond after {rt} seconds."
        f"Your response is <<{'correct' if acc == 1.0 else 'incorrect'}>> and you get {points} point{'s' if points != 1 else ''}."
    )

def make_prompts_from_events_conditional(records, file):
    """Combine records + trial into a storytelling prompt for conditional task."""

    basename = os.path.basename(file)
    participant_id = basename.split("_")[0].replace('sub-', '')

    stroop_intro = (
        "You are about to perform a test of attention and cognitive control. "
        "On each trial, a color word appears on the screen in colored ink. "
        "Your goal is to name the ink color, not the word itself. "
        "Sometimes they match (congruent), other times they differ (incongruent). "
        "Respond as quickly and accurately as possible. The task begins now."
        "The nine responses available are con1, con2, con3, inc1, inc2, inc3, inc4, inc5, in6"
        "After your response, you will receive feedback: 1 point for a correct response, or 0 points for an incorrect response."
        "The correct response for one stimulus does not inform you about the correct response for another stimulus."
    )

    trial_prompts = [make_trial_prompt_conditional(row) for row in records]

    full_text = stroop_intro + "\n\n" + " ".join(trial_prompts)

    return {
        'text': full_text,
        'experiment': file,
        'participant': participant_id,
        'meta-data': {
            'task-label': 'Conditional Associative Learning',
            'prompt-style': 'context-cue + stimulus + feedback',
        }
    }

def generate_conditional_dataset(base_path):
    """Generate conditional associative learning dataset (subjects 11-136)."""

    print("="*80)
    print("GENERATING CONDITIONAL ASSOCIATIVE LEARNING DATASET")
    print("="*80)

    all_subjects = sorted(os.listdir(base_path))
    subjects = all_subjects[11:136]  # Subjects 11-136

    print(f"Processing {len(subjects)} subjects (indices 11-136)...")

    all_records = []

    for i, sub in enumerate(subjects, start=11):
        sub_path = os.path.join(base_path, sub, 'func')
        if not os.path.exists(sub_path):
            print(f"  ⚠ Skipping {sub}: func directory not found")
            continue

        # Find stroop event files
        files = sorted(glob.glob(os.path.join(sub_path, f"{sub}_task-stroop*run-01_events.tsv")))
        selected = [f for f in files if 'stroopPA_run-01_events.tsv' in f or 'stroopAP_run-01_events.tsv' in f]

        if not selected:
            print(f"  ⚠ Skipping {sub}: no stroop event files found")
            continue

        combined_text = []

        for select in selected:
            recs = read_tsv(select)
            story_dict = make_prompts_from_events_conditional(recs, select)
            combined_text.append(story_dict['text'])

        if combined_text:
            all_records.append({
                'text': " ".join(combined_text),
                'experiment': 'stroop_task[Conditional-Associative-Learning]',
                'participant': sub.replace("sub-", "")
            })
            print(f"  ✓ Processed {sub} ({len(combined_text)} file(s))")

    # Create DataFrame
    dataset_df = pd.DataFrame(all_records)

    # Save as CSV
    csv_path = "stroop_dataset_conditional.csv"
    dataset_df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved: {csv_path} ({len(dataset_df)} records)")

    # Save as JSONL
    jsonl_path = "stroop_dataset_conditional.jsonl"
    dataset_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"✓ JSONL saved: {jsonl_path}")

    return dataset_df

# ==============================================================================
# SERIAL REACTION TIME (Participants 136+)
# ==============================================================================

def make_trial_prompt_serial(row):
    """Generate a story-like prompt for serial task."""

    rt = row.get("response_time", "n/a")
    acc = row.get("accuracy_binarized", "n/a")
    stim_text = row.get("stimulus_text", "unknown")
    stim_color = row.get("stimulus_color", "unknown")
    trial_type = row.get("trial_type", "unknown")

    return (
        f"The instruction is to name the ink color as quickly and accurately as possible. "
        f"The word shown is '{stim_text}' in '{stim_color}' ink which is a '{trial_type}' event. "
        f"You respond after {rt} seconds. "
        f"Your response is <<{'correct' if acc == 1.0 else 'incorrect'}>>. "
    )

def make_prompts_from_events_serial(records, file):
    """Combine records + trial into a storytelling prompt for serial task."""

    basename = os.path.basename(file)
    participant_id = basename.split("_")[0].replace('sub-', '')

    stroop_intro = (
        "You are about to perform a test of attention and cognitive control. "
        "On each trial, a color word appears on the screen in colored ink. "
        "Your goal is to name the ink color, not the word itself. "
        "Sometimes they match (congruent), other times they differ (incongruent). "
        "Respond as quickly and accurately as possible. The task begins now. "
    )

    trial_prompts = [make_trial_prompt_serial(row) for row in records]

    full_text = stroop_intro + "\n\n" + " ".join(trial_prompts)

    return {
        'text': full_text,
        'experiment': file,
        'participant': participant_id,
        'meta-data': {
            'task-label': 'Serial Reaction Time',
            'prompt-style': 'instruction + action + feedback',
        }
    }

def generate_serial_dataset(base_path):
    """Generate serial reaction time dataset (subjects 136+)."""

    print("\n" + "="*80)
    print("GENERATING SERIAL REACTION TIME DATASET")
    print("="*80)

    all_subjects = sorted(os.listdir(base_path))
    subjects = all_subjects[136:]  # Subjects 136+

    print(f"Processing {len(subjects)} subjects (indices 136+)...")

    all_records = []

    for sub in subjects:
        sub_path = os.path.join(base_path, sub, 'func')
        if not os.path.exists(sub_path):
            print(f"  ⚠ Skipping {sub}: func directory not found")
            continue

        # Find stroop event files
        files = sorted(glob.glob(os.path.join(sub_path, f"{sub}_task-stroop*run-01_events.tsv")))
        selected = [f for f in files if 'stroopPA_run-01_events.tsv' in f or 'stroopAP_run-01_events.tsv' in f]

        if not selected:
            print(f"  ⚠ Skipping {sub}: no stroop event files found")
            continue

        combined_text = []

        for select in selected:
            recs = read_tsv(select)
            story_dict = make_prompts_from_events_serial(recs, select)
            combined_text.append(story_dict['text'])

        if combined_text:
            all_records.append({
                'text': " ".join(combined_text),
                'experiment': 'stroop_task[Serial Reaction Timing]',
                'participant': sub.replace("sub-", "")
            })
            print(f"  ✓ Processed {sub} ({len(combined_text)} file(s))")

    # Create DataFrame
    dataset_df = pd.DataFrame(all_records)

    # Save as CSV
    csv_path = "stroop_dataset_serial.csv"
    dataset_df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved: {csv_path} ({len(dataset_df)} records)")

    # Save as JSONL
    jsonl_path = "stroop_dataset_serial.jsonl"
    dataset_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"✓ JSONL saved: {jsonl_path}")

    return dataset_df

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    # Check if dataset exists
    if not os.path.exists(BASE_PATH):
        print("="*80)
        print("ERROR: OpenNeuro dataset ds005237 not found!")
        print("="*80)
        print("\nPlease clone the dataset first:")
        print("  git clone https://github.com/OpenNeuroDatasets/ds005237.git")
        print("\nOr update BASE_PATH in this script to point to the dataset location.")
        print("="*80)
        exit(1)

    print("\n" + "="*80)
    print("STROOP DATASET GENERATION")
    print("="*80)
    print(f"Base path: {BASE_PATH}")

    # Generate both datasets
    conditional_df = generate_conditional_dataset(BASE_PATH)
    serial_df = generate_serial_dataset(BASE_PATH)

    # Summary
    print("\n" + "="*80)
    print("✅ GENERATION COMPLETE")
    print("="*80)
    print(f"Conditional dataset: {len(conditional_df)} participants")
    print(f"Serial dataset: {len(serial_df)} participants")
    print(f"Total: {len(conditional_df) + len(serial_df)} participants")
    print("\nFiles created:")
    print("  - stroop_dataset_conditional.csv")
    print("  - stroop_dataset_conditional.jsonl")
    print("  - stroop_dataset_serial.csv")
    print("  - stroop_dataset_serial.jsonl")
    print("="*80)
