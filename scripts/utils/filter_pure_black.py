#!/usr/bin/env python3
"""
Filter out pure black videos from dataset

This script reads a blacklist of video paths from a JSONL file,
filters out matching videos from the main dataset JSON file,
and writes the filtered data to a new JSON file.
"""

import json
import os
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_jsonl_paths(blacklist_file):
    """Read paths from each line of a JSONL file"""
    blacklist_paths = set()
    with open(blacklist_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            if 'path' in entry:
                blacklist_paths.add(entry['path'])

    logger.info(f"Loaded {len(blacklist_paths)} blacklisted paths")
    return blacklist_paths

def normalize_path(path):
    """Normalize paths to handle potential inconsistencies in representation"""
    return os.path.normpath(path)

def filter_dataset(dataset_file, blacklist_paths, output_file):
    """Filter out entries with paths in the blacklist"""
    # Load the dataset
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    original_count = len(dataset)
    logger.info(f"Loaded dataset with {original_count} entries")

    # Normalize blacklist paths
    normalized_blacklist = {normalize_path(path) for path in blacklist_paths}

    # Filter the dataset
    filtered_dataset = []
    blacklisted_count = 0

    for entry in tqdm(dataset, desc="Filtering dataset"):
        if 'path' in entry:
            normalized_entry_path = normalize_path(entry['path'])
            if normalized_entry_path not in normalized_blacklist:
                filtered_dataset.append(entry)
            else:
                blacklisted_count += 1
        else:
            # Keep entries without a path
            filtered_dataset.append(entry)

    # Write the filtered dataset
    with open(output_file, 'w') as f:
        json.dump(filtered_dataset, f, indent=2)

    logger.info(f"Removed {blacklisted_count} blacklisted entries")
    logger.info(f"Wrote {len(filtered_dataset)} entries to {output_file}")

    return len(filtered_dataset)

def main():
    # File paths
    base_dir = "/cv/bjzhu/datasets/moviidb_v0.2"

    blacklist_file = os.path.join(base_dir, "pure_black_list.jsonl")
    dataset_file = os.path.join(base_dir, "wanx_dataset_meta_extend_with_meta_2_5.json")
    output_file = os.path.join(base_dir, "wanx_dataset_meta_extend_with_meta_2_5_filtered.json")

    # Read the black list
    logger.info(f"Reading blacklist from {blacklist_file}")
    blacklist_paths = read_jsonl_paths(blacklist_file)

    # Filter the dataset
    logger.info(f"Filtering dataset {dataset_file}")
    filtered_count = filter_dataset(dataset_file, blacklist_paths, output_file)

    # Print summary
    logger.info(f"Filtering complete. {filtered_count} entries remain in the filtered dataset.")


if __name__ == "__main__":
    main()
