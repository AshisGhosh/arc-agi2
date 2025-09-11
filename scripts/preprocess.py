#!/usr/bin/env python3
"""preprocess ARC JSON files into HRM training format"""

import os
import sys
import json
import random
import torch
import argparse
from pathlib import Path
from uuid import uuid4
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_arc_json, pad_grid_center


def main():
    parser = argparse.ArgumentParser(description="preprocess ARC data for HRM training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="validation split ratio"
    )
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)

    # source directories
    arc_agi2_dir = Path("ARC-AGI-2/data")
    arc_agi1_dir = Path("ARC-AGI/data")

    # output directories - split by agi1 vs agi2
    output_dir = Path("data")
    agi2_dir = output_dir / "agi2"
    agi1_dir = output_dir / "agi1"

    # create output directories
    for split in ["train", "val"]:
        os.makedirs(agi2_dir / split, exist_ok=True)
        os.makedirs(agi1_dir / split, exist_ok=True)

    # process agi2 dataset
    print("processing ARC-AGI-2 dataset...")
    process_dataset(arc_agi2_dir, agi2_dir, seed, args.val_ratio, "agi2")

    # process agi1 dataset
    print("processing ARC-AGI dataset...")
    process_dataset(arc_agi1_dir, agi1_dir, seed, args.val_ratio, "agi1")

    print("preprocessing complete!")


def process_dataset(
    src_dir: Path, output_dir: Path, seed: int, val_ratio: float, dataset_name: str
):
    """process a single dataset (agi1 or agi2)"""

    # find all json files
    json_files = list(src_dir.glob("**/*.json"))
    print(f"found {len(json_files)} json files in {dataset_name}")

    if not json_files:
        print(f"warning: no json files found in {src_dir}")
        return

    # create task ID mapping
    task_to_id = {}
    for i, filepath in enumerate(json_files):
        task_to_id[filepath.stem] = i

    # save task ID mapping
    task_id_map = {
        "num_tasks": len(task_to_id),
        "task_to_id": task_to_id,
        "dataset": dataset_name,
    }

    task_map_path = output_dir / "task_id_map.json"
    with open(task_map_path, "w") as f:
        json.dump(task_id_map, f, indent=2)

    print(f"saved task ID mapping for {dataset_name}: {len(task_to_id)} tasks")

    # shuffle JSON files first, then split by task (not by example)
    rng = random.Random(seed)
    rng.shuffle(json_files)
    n_val = int(len(json_files) * val_ratio)
    val_set = set(f.stem for f in json_files[:n_val])

    train_count = 0
    val_count = 0

    for filepath in json_files:
        task_id = task_to_id[filepath.stem]
        split = "val" if filepath.stem in val_set else "train"

        data = load_arc_json(str(filepath))
        train_examples = data["train"]
        test_examples = data["test"] if "test" in data else []
        all_examples = train_examples + test_examples  # combine for max samples

        if len(all_examples) >= 3:
            from itertools import combinations

            for combo in combinations(range(len(all_examples)), 3):
                support_indices = combo[:2]
                query_index = combo[2]

                # build support pairs
                support_pairs = []
                for j in support_indices:
                    inp = np.array(all_examples[j]["input"], dtype=np.int64)
                    out = np.array(all_examples[j]["output"], dtype=np.int64)
                    inp_padded = pad_grid_center(inp, target_size=30)
                    out_padded = pad_grid_center(out, target_size=30)
                    support_pairs.append(
                        {
                            "inp": torch.from_numpy(inp_padded.ravel()),
                            "out": torch.from_numpy(out_padded.ravel()),
                        }
                    )

                # build query pair
                query_inp = np.array(all_examples[query_index]["input"], dtype=np.int64)
                query_out = np.array(
                    all_examples[query_index]["output"], dtype=np.int64
                )
                query_inp_flat = torch.from_numpy(
                    pad_grid_center(query_inp, target_size=30).ravel()
                )
                query_out_flat = torch.from_numpy(
                    pad_grid_center(query_out, target_size=30).ravel()
                )

                # track sources
                support_sources = [
                    "train" if j < len(train_examples) else "test"
                    for j in support_indices
                ]
                query_source = "train" if query_index < len(train_examples) else "test"

                # create sample
                sample = {
                    "task_id": task_id,
                    "support_pairs": support_pairs,
                    "query_inp": query_inp_flat,
                    "query_out": query_out_flat,
                    "support_sources": support_sources,
                    "query_source": query_source,
                }

                # save with dataset prefix
                filename = f"{split}_{filepath.stem}_combo_{support_indices[0]:02d}_{support_indices[1]:02d}_{query_index:02d}_{uuid4().hex}.pt"
                torch.save(sample, os.path.join(output_dir, split, filename))

                if split == "train":
                    train_count += 1
                else:
                    val_count += 1
        else:
            print(
                f"warning: task {filepath.stem} has {len(all_examples)} examples, need at least 3"
            )

    print(f"{dataset_name} - train: {train_count}, val: {val_count}")


if __name__ == "__main__":
    main()
