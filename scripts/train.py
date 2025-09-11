#!/usr/bin/env python3
"""train HRM model on ARC tasks"""

import os
import sys
import json
import yaml
import torch
from torch.utils.data import DataLoader
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.hrm import HRM
from algo.trainer import HRMTrainer
from utils.hrm_dataset import HRMDataset


def custom_collate_fn(batch):
    """custom collate function to properly handle support_pairs and file_names"""
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    puzzle_identifiers = torch.tensor([item[2] for item in batch])  # convert to tensor
    support_pairs_list = [item[3] for item in batch]  # keep as list of lists
    file_names = [item[4] for item in batch]  # keep as list of strings
    
    return inputs, labels, puzzle_identifiers, support_pairs_list, file_names


def main():
    parser = argparse.ArgumentParser(description="train HRM model on ARC tasks")
    parser.add_argument(
        "--dataset",
        type=str,
        default="agi2",
        choices=["agi1", "agi2"],
        help="dataset to use (agi1 or agi2)",
    )
    args = parser.parse_args()

    # set determinism for reproducible results
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"using dataset: {args.dataset}")

    # load configs
    with open("configs/hrm.yaml", "r") as f:
        config = yaml.safe_load(f)

    # load task ID mapping to get num_tasks
    task_id_map_path = f"data/{args.dataset}/task_id_map.json"
    with open(task_id_map_path, "r") as f:
        task_id_map = json.load(f)
    num_tasks = task_id_map["num_tasks"]
    print(f"found {num_tasks} tasks in {args.dataset}")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # create model with correct parameters
    model = HRM(
        {
            "vocab_size": config["model"]["num_colors"],
            "seq_len": config["model"]["max_len"],
            "num_puzzle_identifiers": num_tasks,
            "batch_size": config["model"]["batch_size"],
            "H_cycles": config["H_cycles"],
            "L_cycles": config["L_cycles"],
            "H_layers": config["H_layers"],
            "L_layers": config["L_layers"],
            "hidden_size": config["hidden_size"],
            "num_heads": config["num_heads"],
            "expansion": config["expansion"],
            "pos_encodings": config["pos_encodings"],
            "rope_theta": config["rope_theta"],
            "rms_norm_eps": config["rms_norm_eps"],
            "puzzle_emb_ndim": config["puzzle_emb_ndim"],
            "halt_max_steps": config["halt_max_steps"],
            "halt_exploration_prob": config["halt_exploration_prob"],
        }
    ).to(device)

    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # build data loaders
    print("building data loaders...")
    train_ds = HRMDataset(root="data", split="train", dataset=args.dataset)
    val_ds = HRMDataset(root="data", split="val", dataset=args.dataset)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    print(f"train examples: {len(train_ds)}")
    print(f"val examples: {len(val_ds)}")

    # create trainer
    trainer = HRMTrainer(model, config)

    # train
    print("starting training...")
    history = trainer.train(train_loader, val_loader)

    print("training complete!")
    print(f"best validation perfect grid: {trainer.best_val_perfect:.3f}")

    print(history)


if __name__ == "__main__":
    main()
