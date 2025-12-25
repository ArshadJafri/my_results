#!/usr/bin/env python3
"""
Script to split log files into two separate tables:
- Table 1: Trial-by-trial event data
- Table 2: Summary statistics table
"""

import pandas as pd
import os
import sys

def split_log_file(log_file_path):
    """
    Split a log file into two tables and save as separate CSV files.

    Args:
        log_file_path: Path to the .log file

    Returns:
        tuple: (table1_df, table2_df) DataFrames
    """

    print(f"Processing: {log_file_path}")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the split point (where second table starts)
    split_index = None
    for i, line in enumerate(lines):
        # Look for the second table header starting with "Event Type"
        if i > 100 and line.strip().startswith("Event Type\tCode\tType"):
            split_index = i
            print(f"  Found split at line {i}")
            break

    if split_index is None:
        print("  Warning: Could not find second table. File may have different format.")
        split_index = len(lines)  # Use entire file as table 1

    # Extract Table 1 (Trial data)
    table1_lines = lines[3:split_index]  # Skip first 3 header lines

    # Find the header line for table 1 (line 3)
    table1_header = lines[3].strip().split('\t')

    # Parse table 1 data
    table1_data = []
    for line in table1_lines[2:]:  # Skip header and blank line
        line = line.strip()
        if line:  # Skip empty lines
            row = line.split('\t')
            table1_data.append(row)

    # Create DataFrame for table 1
    df_table1 = pd.DataFrame(table1_data, columns=table1_header)

    # Extract Table 2 (Summary statistics) if it exists
    df_table2 = None
    if split_index < len(lines):
        table2_lines = lines[split_index:]

        # Find the header
        table2_header = table2_lines[0].strip().split('\t')

        # Parse table 2 data
        table2_data = []
        for line in table2_lines[2:]:  # Skip header and blank line
            line = line.strip()
            if line and not line.startswith("The following"):  # Skip empty and footer lines
                row = line.split('\t')
                table2_data.append(row)

        # Create DataFrame for table 2
        if table2_data:
            df_table2 = pd.DataFrame(table2_data, columns=table2_header)

    return df_table1, df_table2


def process_single_file(log_file, output_dir=None):
    """Process a single log file and save the split tables."""

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(log_file)

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename
    base_name = os.path.splitext(os.path.basename(log_file))[0]

    # Split the file
    df_table1, df_table2 = split_log_file(log_file)

    # Save table 1
    output_file1 = os.path.join(output_dir, f"{base_name}_trial_data.csv")
    df_table1.to_csv(output_file1, index=False)
    print(f"  ✓ Saved Table 1: {output_file1} ({len(df_table1)} rows)")

    # Save table 2 if it exists
    if df_table2 is not None and len(df_table2) > 0:
        output_file2 = os.path.join(output_dir, f"{base_name}_summary.csv")
        df_table2.to_csv(output_file2, index=False)
        print(f"  ✓ Saved Table 2: {output_file2} ({len(df_table2)} rows)")
    else:
        print(f"  ℹ No Table 2 found in this file")

    return df_table1, df_table2


def process_participant_folder(participant_dir, output_dir=None):
    """Process all log files in a participant folder."""

    print(f"\n{'='*80}")
    print(f"Processing participant: {os.path.basename(participant_dir)}")
    print(f"{'='*80}")

    # Find all .log files
    log_files = [f for f in os.listdir(participant_dir) if f.endswith('.log')]

    if not log_files:
        print(f"No .log files found in {participant_dir}")
        return

    print(f"Found {len(log_files)} log file(s)\n")

    # Process each file
    for log_file in sorted(log_files):
        log_path = os.path.join(participant_dir, log_file)

        # Set output directory
        if output_dir is None:
            out_dir = os.path.join(participant_dir, "processed")
        else:
            out_dir = output_dir

        try:
            process_single_file(log_path, out_dir)
        except Exception as e:
            print(f"  ✗ Error processing {log_file}: {e}")

    print()


def process_all_participants(base_dir='.'):
    """Process all participant folders in the base directory."""

    # Find all participant directories (numeric folder names)
    participant_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d[0].isdigit()
    ]

    print(f"\n{'#'*80}")
    print(f"Found {len(participant_dirs)} participant directories")
    print(f"{'#'*80}")

    for participant_dir in sorted(participant_dirs):
        process_participant_folder(participant_dir)

    print(f"\n{'#'*80}")
    print(f"✓ PROCESSING COMPLETE")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Split log files into separate tables')
    parser.add_argument('path', nargs='?', default='.',
                       help='Path to participant folder, log file, or base directory (default: current directory)')
    parser.add_argument('--output', '-o',
                       help='Output directory for processed files')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Process all participant folders in the directory')

    args = parser.parse_args()

    # Determine what to process
    if os.path.isfile(args.path):
        # Single file
        print("Processing single file...")
        process_single_file(args.path, args.output)
    elif os.path.isdir(args.path):
        if args.all:
            # All participants in base directory
            process_all_participants(args.path)
        else:
            # Single participant folder
            process_participant_folder(args.path, args.output)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)
