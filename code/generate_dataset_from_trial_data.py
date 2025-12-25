#!/usr/bin/env python3
"""
Generate ENIGMA-style dataset from the trial_data CSV files
(Uses only the first table - trial-by-trial event data)
"""

import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. MAPPING DICTIONARY (FULLY POPULATED)
# ==============================================================================

def create_full_code_meaning():
    """Generates a robust code_to_meaning dictionary covering all ranges."""

    base_mappings = {
        110: {"BaseDescription": "Smallest concentric ring (CS−, 0.00 continuum)", "Shock": "No", "RDoC_Domain": "Negative Valence → Acute Threat Inhibition"},
        120: {"BaseDescription": "Small concentric ring (GS1, 0.25 continuum)", "Shock": "No", "RDoC_Domain": "Negative Valence → Generalization (Low Threat)"},
        130: {"BaseDescription": "Intermediate concentric ring (GS2, 0.50 continuum)", "Shock": "No", "RDoC_Domain": "Negative Valence → Generalization (Ambiguous Threat)"},
        140: {"BaseDescription": "Large concentric ring (GS3, 0.75 continuum)", "Shock": "No", "RDoC_Domain": "Negative Valence → Generalization (High Threat)"},
        150: {"BaseDescription": "Largest concentric ring (CS+ unpaired, 1.00 continuum)", "Shock": "No", "RDoC_Domain": "Negative Valence → Fear Acquisition (Unpaired)"},
        250: {"BaseDescription": "Largest concentric ring (CS+ paired, 1.00 continuum)", "Shock": "Yes", "RDoC_Domain": "Negative Valence → Fear Acquisition (Paired)"},
    }

    code_to_meaning = {}

    # Map Ring Stimuli (110-155, 250-255)
    for base_code, data in base_mappings.items():
        if base_code < 250:
            for i in range(6):  # Covering variants 0 through 5
                code = base_code + i
                if code > 155: break
                new_data = data.copy()
                if i > 0:
                    new_data["BaseDescription"] += f" with red{i} overlay"
                code_to_meaning[code] = new_data

        elif base_code == 250:
            for i in range(6):  # Covering variants 0 through 5 (250-255)
                code = base_code + i
                new_data = data.copy()
                if i > 0:
                    new_data["BaseDescription"] = new_data["BaseDescription"].replace("paired", f"paired with red{i} overlay")
                code_to_meaning[code] = new_data

    # Map Control Stimuli (40-45, 51-55)
    for code in range(40, 46):
        desc = "Neutral visual control stimulus (non-ring)" if code == 40 else f"Visual control stimulus (VC{code-40})"
        code_to_meaning[code] = {"BaseDescription": desc, "Shock": "No", "RDoC_Domain": "Cognitive Systems → Perception (Baseline Control)"}

    for code in range(51, 56):
        code_to_meaning[code] = {"BaseDescription": f"Checkerboard pattern {code-50}", "Shock": "No", "RDoC_Domain": "Cognitive Systems → Attention / Visual Tracking"}

    return code_to_meaning

code_to_meaning = create_full_code_meaning()


# ==============================================================================
# 2. CORE FUNCTIONS
# ==============================================================================

def read_csv_properly(csv_file):
    """Try different methods to read the CSV correctly."""
    try:
        df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
        print(f"  ✓ Read {os.path.basename(csv_file)}: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"  ✗ Error reading {csv_file}: {e}")
        return None


def generate_sg_trial_parts(row, code_to_meaning):
    """
    Generates the text parts for either the Stimulus (Picture) or the Response event.
    """

    trial = row.get('Trial', 'N/A')
    event_type = row.get('Event Type', '')
    code_raw = row.get('Code', '')
    rt = row.get('Time') or row.get('TTime')

    # --- Picture events (Stimulus/Cues) ---
    if event_type == 'Picture' and str(code_raw).strip() and str(code_raw).isdigit():
        code_int = int(code_raw)
        stimulus_details = code_to_meaning.get(code_int)
        if stimulus_details:
            description = stimulus_details.get("BaseDescription", f"Unknown stimulus")
            return f"You see {description} of code {code_int}"

    # --- Response events (Risk Rating 0, 1, 2) ---
    elif event_type == 'Response' and code_raw in ['0', '1', '2']:
        if code_raw == '2':
            rdoc_domain = code_to_meaning.get(250, {}).get("RDoC_Domain", "UNKNOWN RDoC DOMAIN")
            rt_text = ""
            if rt is not None and str(rt).strip() and str(rt).strip().replace('.', '', 1).isdigit():
                rt_text = f"{float(rt):.3f} secs."

            return (
                f"press <<X>> in {rt_text} which is linked with the {rdoc_domain} domain."
            )
        else:
            return f"respond with <<no response>>."

    return None


def extract_block_name(csv_file_name):
    """Extracts a readable block name from the CSV file name."""
    name = os.path.basename(csv_file_name)
    if 'preACQ' in name:
        return "Pre-Acquisition Baseline"
    if 'ACQ' in name:
        return "Acquisition (Learning)"
    if 'Test_1' in name:
        return "Generalization Run 1 (Early)"
    if 'Test_2' in name:
        return "Generalization Run 2 (Late)"
    return "Unknown Block"


# ==============================================================================
# 3. ASSEMBLE BLOCKS BY PHASE
# ==============================================================================

def assemble_sg_blocks_by_phase(csv_files, participant_id):
    """
    Combines all trial events into the N-Back block format, grouped by CSV file.
    """

    full_text_lines = []

    # N-BACK STYLE INTRO
    intro = (
        "You will view a stream of images on the screen. "
        "You will go through different blocks which represent each phase of the log cycle. "
        "The task requires you to monitor stimulus properties and respond when prompted by the red crosshair."
    )
    full_text_lines.append(intro)

    for csv_file in sorted(csv_files):
        df = read_csv_properly(csv_file)

        if df is None:
            continue

        # 1. Start a new block for each CSV file
        block_name = extract_block_name(csv_file)
        full_text_lines.append(f"\n\nBLOCK: {block_name}")

        # 2. Process all trials within this block
        last_stimulus_prompt_part = None

        for _, row in df.iterrows():
            prompt_part = generate_sg_trial_parts(row, code_to_meaning)

            if not prompt_part:
                continue

            event_type = row.get('Event Type')

            if event_type == 'Picture':
                last_stimulus_prompt_part = prompt_part

            elif event_type == 'Response' and row.get('Code') in ['0', '1', '2'] and last_stimulus_prompt_part:
                # Combine Picture and subsequent Response into one line
                trial = row.get('Trial', 'N/A')
                final_line = f"Trial {trial}: {last_stimulus_prompt_part} and {prompt_part}"
                full_text_lines.append(final_line)
                last_stimulus_prompt_part = None  # Reset for next trial

    # Combine all lines
    full_text = "\n\n".join(full_text_lines)

    return {
        'text': full_text,
        'experiment': 'Go/No-Go - Sequential Fear Generalization (N-Back Style)',
        'participant': participant_id
    }


# ==============================================================================
# 4. PROCESS ALL PARTICIPANTS
# ==============================================================================

def process_all_sg_subjects(base_dir='.'):
    """
    Processes all participants by looking for processed/*_trial_data.csv files.
    """

    all_data = []

    # Find all participant directories (numeric folder names starting with digit)
    participant_dirs = []
    try:
        for entry in os.scandir(base_dir):
            if entry.is_dir() and entry.name[0].isdigit():
                participant_dirs.append(entry.path)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The base directory '{base_dir}' was not found.")
        return pd.DataFrame(all_data)

    for participant_dir in sorted(participant_dirs):
        participant_id = os.path.basename(participant_dir)

        print(f"\n{'='*80}")
        print(f"PROCESSING PARTICIPANT: {participant_id}")
        print(f"{'='*80}")

        # Look for trial_data CSV files in the processed subdirectory
        processed_dir = os.path.join(participant_dir, 'processed')

        if not os.path.exists(processed_dir):
            print(f"  ℹ No 'processed' folder found for {participant_id}")
            continue

        csv_files = []
        try:
            csv_files = [
                os.path.join(processed_dir, f)
                for f in os.listdir(processed_dir)
                if f.endswith('_trial_data.csv')
            ]
        except FileNotFoundError:
            continue

        if not csv_files:
            print(f"  ✗ No trial_data CSV files found for {participant_id}")
            continue

        print(f"  Found {len(csv_files)} trial_data.csv file(s)")

        prompt_data = assemble_sg_blocks_by_phase(csv_files, participant_id)
        all_data.append(prompt_data)
        print(f"\n  ✓ {participant_id}: Generated {len(prompt_data['text'])} characters")

    df = pd.DataFrame(all_data)
    df = df[['text', 'experiment', 'participant']]
    return df


# ==============================================================================
# 5. TRAIN/TEST SPLIT
# ==============================================================================

def create_train_test_split(df, test_size=0.25, random_seed=42):
    """
    Split the dataset into train and test sets.
    """

    if len(df) == 0:
        print("No data to split!")
        return None, None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True
    )

    print(f"\n{'='*80}")
    print(f"TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    print(f"Total records: {len(df)}")
    print(f"Train set: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, test_df


# ==============================================================================
# 6. SAVE TO JSONL
# ==============================================================================

def save_to_jsonl(df, output_path):
    """Save DataFrame to JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # Ensure participant is a string
            record = {
                'text': row['text'],
                'experiment': row['experiment'],
                'participant': str(row['participant'])
            }
            f.write(json.dumps(record) + '\n')
    print(f"  ✓ Saved: {output_path}")


# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":

    print("="*80)
    print("GENERATING SEQUENTIAL FEAR GENERALIZATION DATASET")
    print("FROM TRIAL_DATA CSV FILES")
    print("="*80)

    # Process all participants
    df = process_all_sg_subjects('.')

    if len(df) == 0:
        print("\n✗ No data generated. Exiting.")
        exit(1)

    # Save full dataset
    output_csv = 'sequential_fear_generalization_dataset_from_trial_data.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"✓ SAVED FULL DATASET TO: {output_csv}")
    print(f"{'='*80}")
    print(f"Total participants processed: {len(df)}")

    # Convert to JSONL
    output_jsonl = 'sequential_fear_generalization_dataset_from_trial_data.jsonl'
    save_to_jsonl(df, output_jsonl)

    # Create train/test split
    train_df, test_df = create_train_test_split(df, test_size=0.25, random_seed=42)

    # Save train and test sets
    if train_df is not None and test_df is not None:
        os.makedirs('splits', exist_ok=True)

        train_jsonl = 'splits/train.jsonl'
        test_jsonl = 'splits/test.jsonl'

        save_to_jsonl(train_df, train_jsonl)
        save_to_jsonl(test_df, test_jsonl)

        print(f"\n{'='*80}")
        print(f"✓ TRAIN/TEST SPLIT SAVED")
        print(f"{'='*80}")
        print(f"Train: {train_jsonl}")
        print(f"Test: {test_jsonl}")

    # Show example
    if not df.empty:
        print(f"\n{'='*80}")
        print(f"EXAMPLE OUTPUT (First participant)")
        print(f"{'='*80}")
        sample_text = df['text'].iloc[0].split('\n\n')
        print('\n'.join(sample_text[0:5]))
        print("...")
