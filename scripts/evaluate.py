#!/usr/bin/env python3
"""evaluate HRM model on ARC tasks"""

import os
import sys
import json
import yaml
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.hrm import HRM, hrm_loss
from algo.trainer import HRMTrainer
from utils.hrm_dataset import HRMDataset
from utils.augmentation import sample_aug_and_inverse, invert_augmentation


def custom_collate_fn(batch):
    """custom collate function to properly handle support_pairs and file_names"""
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    puzzle_identifiers = torch.tensor([item[2] for item in batch])  # convert to tensor
    support_pairs_list = [item[3] for item in batch]  # keep as list of lists
    file_names = [item[4] for item in batch]  # keep as list of strings
    
    return inputs, labels, puzzle_identifiers, support_pairs_list, file_names


def run_act_eval(model, batch, max_steps: int = 16):
    """run ACT evaluation with proper unrolling"""
    carry = model.initial_carry(batch)
    with torch.no_grad():
        for step in range(max_steps):
            carry, outputs = model(carry, batch)
    return outputs


def adapt_on_supports(model, support_batch, inner_steps=3, lr=1e-4, max_steps=16):
    """adapt model on support examples using few-shot learning"""
    # snapshot weights
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for _ in range(inner_steps):
        # pack supports as a batch; run ACT unroll with grads
        carry = model.initial_carry(support_batch)
        
        # unroll ACT for proper multi-step reasoning
        for step in range(max_steps):
            carry, outputs = model(carry, support_batch)
        
        loss, _ = hrm_loss(outputs, support_batch["labels"], carry, {"act_weight": 1.0})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # return a function that predicts on the query, and a restore hook
    @torch.inference_mode()
    def predict_query(query_batch):
        model.eval()
        return run_act_eval(model, query_batch, max_steps)  # forward-only ACT

    def restore():
        model.load_state_dict(state_before)

    return predict_query, restore


def predict_with_voting(model, batch, num_augs: int, voting_strategy: str, max_steps: int = 16) -> dict:
    """run test-time augmentation with voting"""
    device = next(model.parameters()).device
    batch_size = batch["inputs"].shape[0]

    # initialize carry for original prediction
    carry = model.initial_carry(batch)

    with torch.no_grad():
        # unroll ACT for proper multi-step reasoning
        for step in range(max_steps):
            carry, original_output = model(carry, batch)
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
    if num_augs > 0:
        print(f"applying {num_augs} augmentations for TTA...")
        
    for i in tqdm(range(batch_size), desc="processing batch samples"):
        # get original input for this sample
        original_input = batch["inputs"][i : i + 1]  # (1, L)
        
        # generate augmentation samples
        aug_samples = sample_aug_and_inverse(original_input, num_augs)
        
        for aug_input, aug_info, aug_type in aug_samples:
            # create augmented batch
            aug_batch = batch.copy()
            aug_batch["inputs"] = aug_input
            
            # run model on augmented input
            carry = model.initial_carry(aug_batch)
            with torch.no_grad():
                for step in range(max_steps):
                    carry, aug_output = model(carry, aug_batch)
                aug_logits = aug_output["logits"]
            
            # invert the augmentation on the logits
            # for dihedral transforms, we need to apply the inverse transform to the spatial dimensions
            # for color permutations, we need to permute the logits along the color dimension
            if aug_type == "dihedral":
                # for dihedral transforms, we need to reshape logits to 2D spatial + color, apply inverse, then reshape back
                # logits shape: (B, L, num_colors) where L = h*w
                h = w = int((aug_logits.shape[1]) ** 0.5)  # L  the= h*w, so h = w = sqrt(L)
                num_colors = aug_logits.shape[-1]
                
                # reshape to (B, h, w, num_colors)
                logits_2d = aug_logits.view(-1, h, w, num_colors)
                
                # apply inverse dihedral transform to each color channel
                inverted_logits_2d = torch.zeros_like(logits_2d)
                for c in range(num_colors):
                    color_logits = logits_2d[:, :, :, c]  # (B, h, w)
                    inverted_color = invert_augmentation(color_logits, aug_info, aug_type)
                    inverted_logits_2d[:, :, :, c] = inverted_color
                
                # reshape back to (B, L, num_colors)
                inverted_logits = inverted_logits_2d.view(aug_logits.shape)
                
            elif aug_type == "color":
                # for color permutations, we need to permute the logits along the color dimension
                perm = aug_info
                inv_perm = torch.argsort(perm)
                inverted_logits = aug_logits[:, :, inv_perm]
            else:
                inverted_logits = aug_logits
            
            # get prediction from inverted logits
            aug_pred = inverted_logits.argmax(-1)
            
            all_predictions.append(aug_pred)
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


def evaluate_with_few_shot_adaptation(trainer, val_loader, num_augs: int, voting_strategy: str, max_steps: int = 16, inner_steps: int = 3) -> dict:
    """evaluate model using few-shot adaptation + TTA"""
    trainer.model.eval()
    total_cells, total_perfect, batches = 0.0, 0.0, 0

    print(
        f"running few-shot evaluation with {num_augs} augmentations, {voting_strategy} voting, {inner_steps} adaptation steps..."
    )

    for batch_data in tqdm(val_loader, desc="evaluating with few-shot adaptation"):
        # unpack batch data (now includes support_pairs)
        inputs, labels, puzzle_identifiers, support_pairs_list = batch_data
        batch_size = inputs.shape[0]

        # move to device
        device = next(trainer.model.parameters()).device
        inputs = inputs.to(device)
        labels = labels.to(device)
        puzzle_identifiers = puzzle_identifiers.to(device)

        all_predictions = []
        all_logits = []

        # process each sample individually for few-shot adaptation
        for i in tqdm(range(batch_size), desc="processing batch samples"):
            support_pairs = support_pairs_list[i]
            
            if len(support_pairs) == 0:
                print(f"warning: no support pairs for sample {i}, skipping few-shot adaptation")
                # fallback to standard prediction
                query_batch = {
                    "inputs": inputs[i:i+1],
                    "labels": labels[i:i+1], 
                    "puzzle_identifiers": puzzle_identifiers[i:i+1],
                }
                outputs = run_act_eval(trainer.model, query_batch, max_steps)
                all_predictions.append(outputs["logits"].argmax(-1))
                all_logits.append(outputs["logits"])
                continue

            # create support batch from support pairs
            support_inputs = torch.stack([pair["inp"] for pair in support_pairs]).to(device)
            support_labels = torch.stack([pair["out"] for pair in support_pairs]).to(device)
            support_task_ids = puzzle_identifiers[i:i+1].expand(len(support_pairs))
            
            support_batch = {
                "inputs": support_inputs,
                "labels": support_labels,
                "puzzle_identifiers": support_task_ids,
            }

            # adapt model on support examples
            predict_query, restore = adapt_on_supports(
                trainer.model, support_batch, inner_steps=inner_steps, max_steps=max_steps
            )

            try:
                # create query batch
                query_batch = {
                    "inputs": inputs[i:i+1],
                    "labels": labels[i:i+1],
                    "puzzle_identifiers": puzzle_identifiers[i:i+1],
                }

                # predict on query with adapted model
                query_outputs = predict_query(query_batch)
                original_logits = query_outputs["logits"]
                original_pred = original_logits.argmax(-1)

                all_predictions.append(original_pred)
                all_logits.append(original_logits)

                # apply TTA if requested
                if num_augs > 0:
                    for aug_input, aug_info, aug_type in sample_aug_and_inverse(inputs[i:i+1], num_augs):
                        # create augmented query batch
                        aug_query_batch = query_batch.copy()
                        aug_query_batch["inputs"] = aug_input
                        
                        # predict on augmented query
                        aug_outputs = predict_query(aug_query_batch)
                        aug_logits = aug_outputs["logits"]
                        
                        # invert augmentation on logits (same logic as before)
                        if aug_type == "dihedral":
                            h = w = int((aug_logits.shape[1]) ** 0.5)
                            num_colors = aug_logits.shape[-1]
                            logits_2d = aug_logits.view(-1, h, w, num_colors)
                            inverted_logits_2d = torch.zeros_like(logits_2d)
                            for c in range(num_colors):
                                color_logits = logits_2d[:, :, :, c]
                                inverted_color = invert_augmentation(color_logits, aug_info, aug_type)
                                inverted_logits_2d[:, :, :, c] = inverted_color
                            inverted_logits = inverted_logits_2d.view(aug_logits.shape)
                        elif aug_type == "color":
                            perm = aug_info
                            inv_perm = torch.argsort(perm)
                            inverted_logits = aug_logits[:, :, inv_perm]
                        else:
                            inverted_logits = aug_logits
                        
                        aug_pred = inverted_logits.argmax(-1)
                        all_predictions.append(aug_pred)
                        all_logits.append(inverted_logits)

            finally:
                # always restore model state
                restore()

        # stack all predictions and apply voting
        if len(all_predictions) > 0:
            all_preds = torch.cat(all_predictions, dim=0)
            all_logits = torch.cat(all_logits, dim=0)
            
            # reshape for voting
            num_total = len(all_predictions)
            expected_total = batch_size + batch_size * num_augs
            if num_total == expected_total:
                num_augmentations_plus_one = 1 + num_augs
                all_preds = all_preds.reshape(num_augmentations_plus_one, batch_size, -1)
                all_logits = all_logits.reshape(num_augmentations_plus_one, batch_size, -1, all_logits.shape[-1])
                
                # voting
                if voting_strategy == "majority":
                    final_pred = torch.mode(all_preds, dim=0)[0]
                elif voting_strategy == "confidence":
                    log_probs = torch.log_softmax(all_logits, dim=-1)
                    entropy = -(log_probs * torch.softmax(all_logits, dim=-1)).sum(dim=-1)
                    best_idx = entropy.argmin(dim=0)
                    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, all_preds.shape[-1])
                    final_pred = all_preds[best_idx, batch_indices, torch.arange(all_preds.shape[-1], device=device)]
                else:
                    final_pred = all_preds[0]  # fallback to first prediction
                
                # compute metrics
                cell_acc = (final_pred == labels).float().mean()
                perfect_acc = (final_pred == labels).all(dim=1).float().mean()

                total_cells += cell_acc * labels.numel()
                total_perfect += perfect_acc * labels.shape[0]
                batches += 1

    return {
        "cell_acc_last": total_cells / (batches * labels.numel()) if batches > 0 else 0.0,
        "perfect_grid_acc_last": total_perfect / (batches * labels.shape[0]) if batches > 0 else 0.0,
    }


def evaluate_with_tta(trainer, val_loader, num_augs: int, voting_strategy: str, max_steps: int = 16) -> dict:
    """evaluate model using test-time augmentation with voting"""
    trainer.model.eval()
    total_cells, total_perfect, batches = 0.0, 0.0, 0

    print(
        f"running TTA evaluation with {num_augs} augmentations, {voting_strategy} voting..."
    )

    for batch_data in tqdm(val_loader, desc="evaluating with TTA"):
        # unpack batch data (standard format)
        inputs, labels, puzzle_identifiers = batch_data

        # move to device
        device = next(trainer.model.parameters()).device
        inputs = inputs.to(device)
        labels = labels.to(device)
        puzzle_identifiers = puzzle_identifiers.to(device)

        # create batch dict
        batch = {
            "inputs": inputs,
            "labels": labels,
            "puzzle_identifiers": puzzle_identifiers,
        }

        with torch.no_grad():
            # run TTA with voting
            tta_result = predict_with_voting(
                trainer.model, batch, num_augs=num_augs, voting_strategy=voting_strategy, max_steps=max_steps
            )

            if tta_result is None:
                continue

            final_pred = tta_result["final_prediction"]

            # compute metrics
            cell_acc = (final_pred == labels).float().mean()
            perfect_acc = (final_pred == labels).all(dim=1).float().mean()

            total_cells += cell_acc * labels.numel()
            total_perfect += perfect_acc * labels.shape[0]
            batches += 1

    return {
        "cell_acc_last": total_cells / (batches * labels.numel())
        if batches > 0
        else 0.0,
        "perfect_grid_acc_last": total_perfect / (batches * labels.shape[0])
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

    # create trainer and load checkpoint
    trainer = HRMTrainer(model, config)

    checkpoint = "latest.pt"
    checkpoint_path = os.path.join("checkpoints", checkpoint)
    if os.path.exists(checkpoint_path):
        print(f"loading checkpoint: {checkpoint_path}")
        success = trainer.load_checkpoint(checkpoint)
        if not success:
            print("Checkpoint loading failed due to NaN weights. Using untrained model.")
    else:
        print("no checkpoint found, using untrained model")

    # build validation data loader
    print("building validation data loader...")
    # NOTE: we are using the train split to test overfitting
    val_ds = HRMDataset(root="data", split="train", dataset=args.dataset)

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
        collate_fn=custom_collate_fn,
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

    # check config for evaluation settings
    tta_enabled = config.get("evaluation", {}).get("tta_enabled", False)
    few_shot_enabled = config.get("evaluation", {}).get("few_shot_enabled", True)
    num_augs = config.get("evaluation", {}).get("num_augmentations", 8)
    voting_strategy = config.get("evaluation", {}).get("voting_strategy", "majority")
    inner_steps = config.get("evaluation", {}).get("inner_steps", 3)

    # get max_steps from config
    max_steps = config.get("halt_max_steps", 16)

    if few_shot_enabled:
        print(f"Few-shot adaptation enabled: {inner_steps} adaptation steps")
        if tta_enabled:
            print(f"TTA also enabled: {num_augs} augmentations, {voting_strategy} voting")
            print("using proper augmentation with dihedral transforms and color permutations")
        val_metrics = evaluate_with_few_shot_adaptation(
            trainer, val_loader, num_augs, voting_strategy, max_steps, inner_steps
        )
    elif tta_enabled:
        print(
            f"TTA enabled from config: {num_augs} augmentations, {voting_strategy} voting"
        )
        print("using proper augmentation with dihedral transforms and color permutations")
        val_metrics = evaluate_with_tta(trainer, val_loader, num_augs, voting_strategy, max_steps)
    else:
        print("Standard evaluation (no few-shot, no TTA)")
        val_metrics = trainer.validate(val_loader)

    # print results
    print("\nevaluation results:")
    if few_shot_enabled:
        print(f"Few-shot adaptation: {inner_steps} steps")
        if tta_enabled:
            print(f"TTA: {num_augs} augmentations, {voting_strategy} voting")
        print("validation loss: N/A (few-shot mode)")
        print(f"cell accuracy (last cycle): {val_metrics['cell_acc_last']:.3f}")
        print(
            f"perfect grid accuracy (last cycle): {val_metrics['perfect_grid_acc_last']:.3f}"
        )
    elif tta_enabled:
        print(f"TTA: {num_augs} augmentations, {voting_strategy} voting")
        print("validation loss: N/A (TTA mode)")
        print(f"cell accuracy (last cycle): {val_metrics['cell_acc_last']:.3f}")
        print(
            f"perfect grid accuracy (last cycle): {val_metrics['perfect_grid_acc_last']:.3f}"
        )
    else:
        print(f"validation loss: {val_metrics['val_loss']:.4f}")
        print(f"cell accuracy: {val_metrics['val_accuracy']:.3f}")
        print(f"perfect grid accuracy: {val_metrics['val_exact_accuracy']:.3f}")
        print(f"average steps: {val_metrics['val_avg_steps']:.1f}")

    # additional per-task analysis
    print("\nper-task analysis:")
    print("this shows how well the model performs on each individual task ID")
    print("a perfect score would be 1.0 for each task")

if __name__ == "__main__":
    main()
