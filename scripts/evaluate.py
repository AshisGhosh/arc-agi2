#!/usr/bin/env python3
"""evaluate trained HRM model"""

import os
import sys
import yaml
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.hrm import HRM
from algo.trainer import HRMTrainer
from utils.data_loader import load_tensor_pairs
from utils.metrics import compute_cycle_metrics


def main():
    # load configs
    with open("configs/hrm_default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # device
    device = torch.device("cuda" if torch.cuda.isavailable() else "cpu")
    print(f"using device: {device}")

    # create model
    model = HRM(
        num_colors=config["model"]["num_colors"],
        d_model=config["model"]["d_model"],
        d_l=config["model"]["d_l"],
        d_h=config["model"]["d_h"],
        max_len=config["model"]["max_len"],
    ).to(device)

    # create trainer and load checkpoint
    trainer = HRMTrainer(model, "configs/hrm_default.yaml")

    checkpoint_path = "checkpoints/best.pt"
    if os.path.exists(checkpoint_path):
        print(f"loading checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    else:
        print("no checkpoint found, using untrained model")

    # load validation data
    print("loading validation data...")
    val_pairs = load_tensor_pairs("data/val")
    val_pairs = [(inp.to(device), out.to(device)) for inp, out in val_pairs]

    print(f"validation pairs: {len(val_pairs)}")

    # evaluate
    print("evaluating...")
    model.eval()

    batch_size = config["training"]["batch_size"]
    k_support = config["data"]["k_support"]
    cycles = config["hrm"]["cycles"]
    inner_steps = config["hrm"]["inner_steps"]

    # create evaluation batch
    batch = trainer._create_few_shot_batch(val_pairs, k_support, batch_size)

    with torch.no_grad():
        outputs = model(
            batch, cycles=cycles, inner_steps=inner_steps, one_step_grad=False
        )

        # compute metrics
        metrics = compute_cycle_metrics(outputs["logits_per_cycle"], batch.query_out)

        # add loss
        from algo.hrm import hrm_loss, HRMLossCfg

        loss_cfg = HRMLossCfg(
            ds_weight=config["loss"]["ds_weight"],
            ds_decay=config["loss"]["ds_decay"],
            act_weight=config["loss"]["act_weight"],
        )
        loss, loss_metrics = hrm_loss(outputs, batch.query_out, loss_cfg)
        metrics["val_loss"] = loss.item()
        metrics.update(loss_metrics)

    # print results
    print("\nevaluation results:")
    print(f"validation loss: {metrics['val_loss']:.4f}")
    print(f"cell accuracy (last cycle): {metrics['cell_acc_last']:.3f}")
    print(f"perfect grid accuracy (last cycle): {metrics['perfect_grid_acc_last']:.3f}")

    print("\nper-cycle metrics:")
    for cycle in range(cycles):
        print(
            f"  cycle {cycle}: cell_acc={metrics[f'cell_acc_cycle_{cycle}']:.3f}, "
            f"perfect_grid={metrics[f'perfect_grid_acc_cycle_{cycle}']:.3f}"
        )


if __name__ == "__main__":
    main()
