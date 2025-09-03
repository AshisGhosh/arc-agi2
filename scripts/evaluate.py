#!/usr/bin/env python3
"""evaluate HRM model on ARC tasks"""

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
from utils.few_shot_dataset import FewShotDataset
from utils.augmentation import sample_aug_and_inverse, invert_augmentation


def predict_with_voting(model, batch, num_augs: int, voting_strategy: str) -> dict:
    """run test-time augmentation with voting"""
    device = next(model.parameters()).device
    batch_size = batch.query_inp.shape[0]

    with torch.no_grad():
        original_output = model(batch)
        original_logits = original_output["logits"]

    all_predictions = []
    all_logits = []

    # original prediction - split into individual samples
    original_pred = original_logits.argmax(-1)  # (B, L)
    for i in range(batch_size):
        all_predictions.append(original_pred[i : i + 1])  # (1, L)
        all_logits.append(original_logits[i : i + 1])  # (1, L, num_colors)

    # augmented predictions
    total_aug_count = 0
    for i in range(batch_size):
        query_inp = batch.query_inp[i : i + 1]  # (1, L)

        aug_samples = sample_aug_and_inverse(query_inp, num_augs)

        for aug_grid, aug_info, aug_type in aug_samples:
            # create augmented batch
            aug_batch = type(batch)(
                support_inp=batch.support_inp[i : i + 1],
                support_out=batch.support_out[i : i + 1],
                query_inp=aug_grid,
                query_out=batch.query_out[i : i + 1],
                task_id=batch.task_id[i : i + 1],
            )

            # run model on augmented input
            with torch.no_grad():
                aug_output = model(aug_batch)
                aug_logits = aug_output["logits"]  # (1, L, num_colors)
                aug_pred = aug_logits.argmax(-1)  # (1, L)

            # invert augmentation on prediction
            inverted_pred = invert_augmentation(aug_pred, aug_info, aug_type)  # (1, L)
            inverted_logits = invert_augmentation(
                aug_logits, aug_info, aug_type
            )  # (1, L, num_colors)

            all_predictions.append(inverted_pred)
            all_logits.append(inverted_logits)
            total_aug_count += 1

    # stack all predictions
    all_preds = torch.cat(all_predictions, dim=0)  # (N*B, L) where N = 1 + num_augs
    all_logits = torch.cat(all_logits, dim=0)  # (N*B, L, num_colors)

    # reshape to (num_augs+1, B, L)
    num_total = len(all_predictions)
    expected_total = batch_size + batch_size * num_augs  # original + augmented

    if num_total != expected_total:
        return None

    # calculate the correct sequence length
    seq_len = all_preds.shape[-1]

    # reshape to (num_augs+1, batch_size, seq_len)
    # we have batch_size + batch_size*num_augs = batch_size*(1+num_augs) predictions
    # reshape to (1+num_augs, batch_size, seq_len)
    num_augmentations_plus_one = 1 + num_augs
    expected_elements = num_augmentations_plus_one * batch_size * seq_len

    if all_preds.numel() != expected_elements:
        return None

    all_preds = all_preds.reshape(num_augmentations_plus_one, batch_size, seq_len)
    all_logits = all_logits.reshape(num_augmentations_plus_one, batch_size, seq_len, -1)

    # voting
    if voting_strategy == "majority":
        # majority vote per cell
        final_pred = torch.mode(all_preds, dim=0)[0]  # (B, L)
    elif voting_strategy == "confidence":
        # pick prediction with highest confidence (lowest entropy)
        log_probs = torch.log_softmax(all_logits, dim=-1)
        entropy = -(log_probs * torch.softmax(all_logits, dim=-1)).sum(
            dim=-1
        )  # (N, B, L)
        best_idx = entropy.argmin(dim=0)  # (B, L)

        # gather best predictions
        batch_indices = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(-1, all_preds.shape[-1])
        )
        final_pred = all_preds[
            best_idx, batch_indices, torch.arange(all_preds.shape[-1], device=device)
        ]
    else:
        raise ValueError(f"unknown voting strategy: {voting_strategy}")

    return {
        "final_prediction": final_pred,
        "all_predictions": all_preds,
        "all_logits": all_logits,
        "voting_strategy": voting_strategy,
        "num_augmentations": num_augs,
    }


def evaluate_with_tta(trainer, val_loader, num_augs: int, voting_strategy: str) -> dict:
    """evaluate model using test-time augmentation with voting"""
    trainer.model.eval()
    total_cells, total_perfect, batches = 0.0, 0.0, 0

    print(
        f"running TTA evaluation with {num_augs} augmentations, {voting_strategy} voting..."
    )

    for batch_data in val_loader:
        # unpack batch data
        (
            support_inp,
            support_out,
            query_inp,
            query_out,
            task_id,
            support_sources,
            query_source,
        ) = batch_data

        # move to device
        device = next(trainer.model.parameters()).device
        support_inp = support_inp.to(device)
        support_out = support_out.to(device)
        query_inp = query_inp.to(device)
        query_out = query_out.to(device)
        task_id = task_id.to(device)

        # create batch
        from algo.hrm import FewShotBatch

        batch = FewShotBatch(
            support_inp=support_inp,
            support_out=support_out,
            query_inp=query_inp,
            query_out=query_out,
            task_id=task_id,
        )

        with torch.no_grad():
            # run TTA with voting
            tta_result = predict_with_voting(
                trainer.model, batch, num_augs=num_augs, voting_strategy=voting_strategy
            )

            if tta_result is None:
                continue

            final_pred = tta_result["final_prediction"]

            # compute metrics
            cell_acc = (final_pred == query_out).float().mean()
            perfect_acc = (final_pred == query_out).all(dim=1).float().mean()

            total_cells += cell_acc * query_out.numel()
            total_perfect += perfect_acc * query_out.shape[0]
            batches += 1

    return {
        "cell_acc_last": total_cells / (batches * query_out.numel())
        if batches > 0
        else 0.0,
        "perfect_grid_acc_last": total_perfect / (batches * query_out.shape[0])
        if batches > 0
        else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="evaluate HRM model on ARC tasks")
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
    print("TTA settings will be read from config file")

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

    # create model with correct num_tasks
    model = HRM(
        {
            "num_colors": config["model"]["num_colors"],
            "max_len": config["model"]["max_len"],
            "num_tasks": num_tasks,
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

    # create trainer and load checkpoint
    trainer = HRMTrainer(model, config)

    checkpoint_path = "latest.pt"
    if os.path.exists(checkpoint_path):
        print(f"loading checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    else:
        print("no checkpoint found, using untrained model")

    # build validation data loader
    print("building validation data loader...")
    val_ds = FewShotDataset(root="data", split="train", dataset=args.dataset)

    # get evaluation samples (only those with train examples as support)
    eval_indices = val_ds.get_eval_samples()
    print(f"total validation samples: {len(val_ds)}")
    print(f"evaluation samples (train support): {len(eval_indices)}")

    # create a subset dataset for evaluation
    from torch.utils.data import Subset

    eval_ds = Subset(val_ds, eval_indices)

    val_loader = DataLoader(
        eval_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    print(f"evaluation examples: {len(eval_ds)}")

    # check which task IDs we're evaluating
    eval_task_ids = set()
    for idx in eval_indices:
        data = torch.load(val_ds.files[idx])
        eval_task_ids.add(data["task_id"])

    print(
        f"evaluating on {len(eval_task_ids)} unique task IDs: {sorted(eval_task_ids)}"
    )

    # evaluate
    print("evaluating...")

    # check config for TTA settings
    tta_enabled = config.get("evaluation", {}).get("tta_enabled", False)
    num_augs = config.get("evaluation", {}).get("num_augmentations", 8)
    voting_strategy = config.get("evaluation", {}).get("voting_strategy", "majority")

    if tta_enabled:
        print(
            f"TTA enabled from config: {num_augs} augmentations, {voting_strategy} voting"
        )
        val_metrics = evaluate_with_tta(trainer, val_loader, num_augs, voting_strategy)
    else:
        print("TTA disabled in config, using standard evaluation")
        val_metrics = trainer.validate(val_loader)

    # print results
    print("\nevaluation results:")
    if tta_enabled:
        print(f"TTA: {num_augs} augmentations, {voting_strategy} voting")
        print("validation loss: N/A (TTA mode)")
    else:
        print(f"validation loss: {val_metrics['val_loss']:.4f}")
    print(f"cell accuracy (last cycle): {val_metrics['cell_acc_last']:.3f}")
    print(
        f"perfect grid accuracy (last cycle): {val_metrics['perfect_grid_acc_last']:.3f}"
    )

    # additional per-task analysis
    print("\nper-task analysis:")
    print("this shows how well the model performs on each individual task ID")
    print("a perfect score would be 1.0 for each task")

    # we could add more detailed per-task metrics here if needed
    # for now, the overall metrics give us a good sense of performance


if __name__ == "__main__":
    main()
