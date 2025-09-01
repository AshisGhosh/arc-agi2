#!/usr/bin/env python3
"""preprocess ARC training data to tensor format"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_all_training_data, split_train_val, save_tensor_pairs


def main():
    # paths
    arc_data_dir = "ARC-AGI-2/data/training"
    output_dir = "data"

    print("loading ARC training data...")
    all_pairs = load_all_training_data(arc_data_dir, target_size=30)
    print(f"loaded {len(all_pairs)} input/output pairs")

    print("splitting train/val...")
    train_pairs, val_pairs = split_train_val(all_pairs, val_ratio=0.2)
    print(f"train: {len(train_pairs)}, val: {len(val_pairs)}")

    print("saving train tensors...")
    save_tensor_pairs(train_pairs, os.path.join(output_dir, "train"), "train")

    print("saving val tensors...")
    save_tensor_pairs(val_pairs, os.path.join(output_dir, "val"), "val")

    print("preprocessing complete!")


if __name__ == "__main__":
    main()
