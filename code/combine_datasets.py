#!/usr/bin/env python3
"""
Combine multiple task datasets (Fear Generalization + Stroop) into train/test splits
Following the exact logic from ENIGMA_data_prompt.ipynb
"""

import json
import os
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Input JSONL files to combine
INPUT_FILES = [
    'sequential_fear_generalization_dataset_from_trial_data.jsonl',
    'stroop_dataset_conditional.jsonl',
    'stroop_dataset_serial.jsonl',
]

# Output directory for combined splits
OUTPUT_DIR = 'combined_splits'

# Split parameters
TEST_SIZE_RATIO = 0.25
RANDOM_SEED = 42

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def main():
    print("="*80)
    print("COMBINING MULTIPLE TASK DATASETS")
    print("="*80)

    final_train_data = []
    final_test_data = []

    print(f"\nProcessing {len(INPUT_FILES)} dataset file(s)...")

    for filename in INPUT_FILES:
        print(f"\n{'='*80}")
        print(f"Processing: {filename}")
        print(f"{'='*80}")

        # Check if file exists
        if not os.path.exists(filename):
            print(f"⚠ Warning: File not found: {filename}. Skipping.")
            continue

        # Load data from current file
        current_file_data = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        current_file_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
        except Exception as e:
            print(f"✗ Error reading {filename}: {e}")
            continue

        if not current_file_data:
            print(f"⚠ File {filename} is empty or contains no valid data. Skipping.")
            continue

        print(f"  Loaded {len(current_file_data)} records")

        # Perform train_test_split on THIS file's data only
        train_subset, test_subset = train_test_split(
            current_file_data,
            test_size=TEST_SIZE_RATIO,
            random_state=RANDOM_SEED,
            shuffle=True
        )

        print(f"  Train subset: {len(train_subset)} records ({len(train_subset)/len(current_file_data)*100:.1f}%)")
        print(f"  Test subset: {len(test_subset)} records ({len(test_subset)/len(current_file_data)*100:.1f}%)")

        # Add to master lists
        final_train_data.extend(train_subset)
        final_test_data.extend(test_subset)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save combined train set
    train_path = os.path.join(OUTPUT_DIR, 'combined_train.jsonl')
    print(f"\n{'='*80}")
    print(f"Saving combined train set...")
    print(f"{'='*80}")
    with open(train_path, 'w', encoding='utf-8') as f:
        for record in final_train_data:
            # Ensure participant is string
            if 'participant' in record:
                record['participant'] = str(record['participant'])
            json.dump(record, f)
            f.write('\n')
    print(f"  ✓ Saved: {train_path}")
    print(f"  Records: {len(final_train_data)}")

    # Save combined test set
    test_path = os.path.join(OUTPUT_DIR, 'combined_test.jsonl')
    print(f"\nSaving combined test set...")
    with open(test_path, 'w', encoding='utf-8') as f:
        for record in final_test_data:
            # Ensure participant is string
            if 'participant' in record:
                record['participant'] = str(record['participant'])
            json.dump(record, f)
            f.write('\n')
    print(f"  ✓ Saved: {test_path}")
    print(f"  Records: {len(final_test_data)}")

    # Final statistics
    total_records = len(final_train_data) + len(final_test_data)
    print(f"\n{'='*80}")
    print(f"✅ FINAL COMBINED STATISTICS")
    print(f"{'='*80}")
    print(f"Total Combined Records: {total_records}")
    print(f"Final Train Set Size: {len(final_train_data)}")
    print(f"Final Test Set Size: {len(final_test_data)}")
    print(f"Final Test Ratio: {len(final_test_data) / total_records * 100:.1f}%")
    print(f"{'='*80}\n")

    # Show sample from each dataset
    print(f"\n{'='*80}")
    print(f"SAMPLE RECORDS (First from each experiment type)")
    print(f"{'='*80}")

    experiments_seen = set()
    for record in final_train_data:
        exp_type = record.get('experiment', 'Unknown')
        if exp_type not in experiments_seen:
            experiments_seen.add(exp_type)
            print(f"\nExperiment: {exp_type}")
            print(f"Participant: {record.get('participant', 'N/A')}")
            print(f"Text preview: {record.get('text', '')[:200]}...")
            print("-" * 80)

if __name__ == "__main__":
    main()
