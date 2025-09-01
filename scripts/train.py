#!/usr/bin/env python3
"""train HRM model on ARC data"""

import os
import sys
import yaml
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.hrm import HRM
from algo.trainer import HRMTrainer
from utils.data_loader import load_tensor_pairs


def main():
    # load configs
    with open("configs/hrm_default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # create model
    model = HRM(
        num_colors=config["model"]["num_colors"],
        d_model=config["model"]["d_model"],
        d_l=config["model"]["d_l"],
        d_h=config["model"]["d_h"],
        max_len=config["model"]["max_len"],
    ).to(device)

    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # load data
    print("loading preprocessed data...")
    train_pairs = load_tensor_pairs("data/train")
    val_pairs = load_tensor_pairs("data/val")

    print(f"train pairs: {len(train_pairs)}")
    print(f"val pairs: {len(val_pairs)}")

    # move to device
    train_pairs = [(inp.to(device), out.to(device)) for inp, out in train_pairs]
    val_pairs = [(inp.to(device), out.to(device)) for inp, out in val_pairs]

    # create trainer
    trainer = HRMTrainer(model, "configs/hrm_default.yaml")

    # train
    print("starting training...")
    history = trainer.train(train_pairs, val_pairs)

    print("training complete!")
    print(f"best validation perfect grid: {trainer.best_val_perfect:.3f}")

    print(history)


if __name__ == "__main__":
    main()
