#!/usr/bin/env python3
"""
overfit experiment script for n-task overfitting.

allows you to select n specific tasks, overfit on them, and evaluate
with clean organization and reporting.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

from algo.config import Config
from algo.models import create_model
from algo.data import ARCDataset, custom_collate_fn
from algo.data.task_subset import TaskSubset
from algo.training import ARCTrainer
from scripts.evaluate import (
    calculate_perfect_accuracy,
    calculate_pixel_accuracy,
    calculate_near_miss_accuracy,
    calculate_perfect_accuracy_foreground,
    calculate_pixel_accuracy_foreground,
    calculate_near_miss_accuracy_foreground,
)


class OverfitExperiment:
    """manages n-task overfitting experiments with clean organization."""

    def __init__(self, config: Config, experiment_name: str = None):
        self.config = config
        self.experiment_name = (
            experiment_name or f"overfit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # use existing logs directory structure
        self.logs_dir = Path(config.log_dir)
        self.checkpoints_dir = Path(config.checkpoint_dir)

        # create experiment subdirectory in logs
        self.experiment_dir = self.logs_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        print(f"experiment directory: {self.experiment_dir}")

    def select_tasks(
        self, n_tasks: int, task_indices: List[int] = None, random_seed: int = 42
    ) -> List[int]:
        """
        select n tasks for overfitting.

        args:
            n_tasks: number of tasks to select
            task_indices: specific task indices to use (if none, random selection)
            random_seed: random seed for reproducible selection

        returns:
            list of selected task indices
        """
        # load full dataset to get valid task indices (only tasks with multiple test pairs)
        full_dataset = ARCDataset(
            self.config.arc_agi1_dir,
            self.config,
            holdout=True,
            require_multiple_test_pairs=False,
        )
        valid_task_indices = set(full_dataset.valid_tasks)
        total_tasks = len(full_dataset.tasks)  # Total tasks in dataset (0-399)

        if task_indices is not None:
            # validate indices - check if they exist and are valid
            invalid_indices = []
            for idx in task_indices:
                if idx < 0 or idx >= total_tasks:
                    invalid_indices.append(f"{idx} (out of range 0-{total_tasks-1})")
                elif idx not in valid_task_indices:
                    invalid_indices.append(f"{idx} (insufficient training examples)")

            if invalid_indices:
                raise ValueError(
                    f"invalid task indices: {invalid_indices}. "
                    f"Valid range: 0-{total_tasks-1}, Valid tasks (with multiple test pairs): {len(valid_task_indices)}"
                )
            selected_indices = task_indices[:n_tasks]
        else:
            # random selection from all valid task indices
            random.seed(random_seed)
            valid_indices_list = list(valid_task_indices)
            selected_indices = random.sample(
                valid_indices_list, min(n_tasks, len(valid_indices_list))
            )

        print(f"selected {len(selected_indices)} tasks for overfitting:")
        print(f"  task indices: {selected_indices}")

        # save selection info
        selection_info = {
            "n_tasks": len(selected_indices),
            "task_indices": selected_indices,
            "random_seed": random_seed,
            "total_available_tasks": total_tasks,
            "selection_method": "random" if task_indices is None else "manual",
            "require_multiple_test_pairs": False,
            "valid_tasks_count": len(valid_task_indices),
        }

        with open(self.experiment_dir / "task_selection.json", "w") as f:
            json.dump(selection_info, f, indent=2)

        return selected_indices

    def create_task_subset(self, task_indices: List[int]) -> TaskSubset:
        """create dataset subset with only selected tasks."""
        return TaskSubset(
            task_indices=task_indices,
            config=self.config,
            arc_agi1_dir=self.config.arc_agi1_dir,
            holdout=True,
            use_first_combination_only=False,  # Use ALL combinations for training
            require_multiple_test_pairs=False,
        )

    def train_on_tasks(
        self,
        task_indices: List[int],
        epochs: int = 1000,
        early_stopping_patience: int = 50,
    ) -> str:
        """
        train model on selected tasks with overfitting focus.

        args:
            task_indices: indices of tasks to train on
            epochs: maximum number of epochs
            early_stopping_patience: epochs to wait before early stopping

        returns:
            path to best checkpoint
        """
        print(f"\nstarting overfitting training on {len(task_indices)} tasks...")

        # create task subset
        task_dataset = self.create_task_subset(task_indices)

        # create data loader (no validation split for overfitting)
        train_loader = DataLoader(
            task_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,  # keep workers alive between epochs
            collate_fn=custom_collate_fn,
        )

        # create model
        model = create_model(self.config)

        # create trainer with overfitting config
        trainer = ARCTrainer(model, self.config, task_dataset)

        # override config for overfitting
        trainer.config.num_epochs = epochs
        trainer.config.save_interval = 10  # save more frequently for overfitting

        print(f"training on {len(task_dataset)} samples")
        print(f"batch size: {self.config.batch_size}")
        print(f"learning rate: {self.config.learning_rate}")

        # custom overfitting training loop
        start_time = time.time()
        best_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            # update trainer's current epoch for proper logging
            trainer.current_epoch = epoch

            # train one epoch using rule latent training method
            train_metrics = trainer.train_epoch_rule_latent(train_loader)
            avg_loss = train_metrics["total_loss"]

            # update scheduler
            trainer.scheduler.step()

            # log progress
            if epoch % 10 == 0 or epoch < 10:
                elapsed = time.time() - start_time
                current_lr = trainer.optimizer.param_groups[0]["lr"]
                reg_loss = train_metrics.get("rule_latent_regularization", 0.0)
                active_groups = train_metrics.get("active_groups", 0)
                print(
                    f"epoch {epoch:4d}: loss={avg_loss:.6f}, reg={reg_loss:.6f}, groups={active_groups}, lr={current_lr:.2e} (elapsed: {elapsed:.1f}s)"
                )

            # save best model
            if avg_loss < best_loss:
                print(
                    f"  New best! {avg_loss:.6f} < {best_loss:.6f} (improvement: {best_loss - avg_loss:.6f})"
                )
                best_loss = avg_loss
                best_epoch = epoch
                patience_counter = 0

                # save checkpoint using existing method (saves to main checkpoints dir)
                trainer.save_checkpoint(epoch, is_best=True)

                # also save to experiment directory for easy access
                experiment_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.scheduler.state_dict(),
                    "best_loss": best_loss,
                    "config": self.config,
                }
                checkpoint_path = self.experiment_dir / "best_model.pt"
                torch.save(experiment_checkpoint, checkpoint_path)
            else:
                patience_counter += 1

            # early stopping
            if patience_counter >= early_stopping_patience:
                print(
                    f"early stopping at epoch {epoch} (patience: {early_stopping_patience})"
                )
                break

        training_time = time.time() - start_time
        print("\ntraining completed!")
        print(f"  best epoch: {best_epoch}")
        print(f"  best loss: {best_loss:.6f}")
        print(f"  training time: {training_time:.1f}s")

        # save training info
        training_info = {
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "total_epochs": epoch + 1,
            "training_time_seconds": training_time,
            "early_stopping_patience": early_stopping_patience,
            "tasks": {
                "n_tasks": len(task_indices),
                "task_indices": task_indices,
            },
        }

        with open(self.experiment_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

        return str(self.experiment_dir / "best_model.pt")

    def evaluate_on_tasks(
        self, task_indices: List[int], checkpoint_path: str
    ) -> Dict[str, Any]:
        """
        evaluate model on selected tasks using existing evaluation functions.

        args:
            task_indices: indices of tasks to evaluate on
            checkpoint_path: path to model checkpoint

        returns:
            evaluation results dictionary
        """
        print(f"\nevaluating on {len(task_indices)} tasks...")

        # create task subset
        task_dataset = self.create_task_subset(task_indices)

        # create data loader
        eval_loader = DataLoader(
            task_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,  # keep workers alive between epochs
            collate_fn=custom_collate_fn,
        )

        # create model and load checkpoint
        model = create_model(self.config)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.config.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.config.device)
        model.eval()

        # use existing evaluation logic from evaluate.py
        total_samples = 0
        perfect_matches = 0
        pixel_correct = 0
        near_miss_correct = 0

        # foreground metrics
        perfect_matches_foreground = 0
        pixel_correct_foreground = 0
        near_miss_correct_foreground = 0

        # per-task results
        per_task_results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                # Move batch to device
                device = next(model.parameters()).device
                rule_latent_inputs = batch["rule_latent_inputs"].to(device)
                test_inputs = batch["test_inputs"].to(device)
                test_outputs = batch["test_outputs"].to(device)
                holdout_inputs = batch["holdout_inputs"].to(device)
                holdout_outputs = batch["holdout_outputs"].to(device)
                has_holdout = batch["has_holdout"].to(device)

                # Get task indices for this batch
                task_indices = batch["task_indices"]

                # Get all training examples for each task in the batch
                batch_size = rule_latent_inputs.size(0)
                max_train = 0
                all_train_inputs_list = []
                num_train_list = []

                for i in range(batch_size):
                    task_idx = task_indices[i]  # Already an integer from the list
                    training_examples = task_dataset.get_all_training_examples_for_task(
                        task_idx
                    )

                    # Extract inputs
                    train_inputs = [
                        ex["input"].squeeze(0) for ex in training_examples
                    ]  # Remove batch dim

                    max_train = max(max_train, len(train_inputs))
                    all_train_inputs_list.append(train_inputs)
                    num_train_list.append(len(train_inputs))

                # Pad all training inputs to consistent shape
                all_train_inputs = torch.zeros([batch_size, max_train, 1, 30, 30]).to(
                    device
                )

                for i, train_inputs in enumerate(all_train_inputs_list):
                    for j, train_input in enumerate(train_inputs):
                        all_train_inputs[i, j] = train_input

                num_train = torch.tensor(num_train_list, dtype=torch.long).to(device)

                # Forward pass with batched rule latent training
                outputs = model.forward_rule_latent_training(
                    rule_latent_inputs, all_train_inputs, num_train
                )

                # Process each task in the batch
                for i in range(rule_latent_inputs.size(0)):
                    # Get number of test examples for this task
                    num_test = batch["num_test_examples"][i]

                    # Evaluate on all test examples for this task
                    for test_idx in range(num_test):
                        # Get specific test example
                        test_input = test_inputs[
                            i, test_idx : test_idx + 1
                        ]  # [1, 30, 30]
                        test_output = test_outputs[
                            i, test_idx : test_idx + 1
                        ]  # [1, 30, 30]

                        # Evaluate on this test example
                        test_logits = model.decoder(
                            outputs["rule_latents"][i : i + 1], test_input
                        )
                        test_target = test_output

                        # calculate metrics using existing functions
                        batch_size = test_logits.size(0)
                        total_samples += batch_size

                        # use existing accuracy functions
                        perfect_matches += (
                            calculate_perfect_accuracy(test_logits, test_target)
                            * batch_size
                        )

                        pixel_correct += (
                            calculate_pixel_accuracy(test_logits, test_target)
                            * batch_size
                        )

                        near_miss_correct += (
                            calculate_near_miss_accuracy(test_logits, test_target)
                            * batch_size
                        )

                        # foreground metrics
                        perfect_matches_foreground += (
                            calculate_perfect_accuracy_foreground(
                                test_logits, test_target
                            )
                            * batch_size
                        )

                        pixel_correct_foreground += (
                            calculate_pixel_accuracy_foreground(
                                test_logits, test_target
                            )
                            * batch_size
                        )

                        near_miss_correct_foreground += (
                            calculate_near_miss_accuracy_foreground(
                                test_logits, test_target
                            )
                            * batch_size
                        )

                    # evaluate on holdout target (if available)
                    if has_holdout[i]:
                        holdout_logits = model.decoder(
                            outputs["rule_latents"][i : i + 1],
                            holdout_inputs[i : i + 1],
                        )
                        holdout_target = holdout_outputs[i : i + 1]

                        # calculate holdout metrics
                        holdout_perfect = calculate_perfect_accuracy(
                            holdout_logits, holdout_target
                        )
                        holdout_pixel = calculate_pixel_accuracy(
                            holdout_logits, holdout_target
                        )
                        holdout_near_miss = calculate_near_miss_accuracy(
                            holdout_logits, holdout_target
                        )

                        # store holdout metrics for this batch
                        if not hasattr(self, "holdout_metrics"):
                            self.holdout_metrics = {
                                "perfect_matches": 0,
                                "pixel_correct": 0,
                                "near_miss_correct": 0,
                                "total_samples": 0,
                            }

                        self.holdout_metrics["perfect_matches"] += (
                            holdout_perfect * batch_size
                        )
                        self.holdout_metrics["pixel_correct"] += (
                            holdout_pixel * batch_size
                        )
                        self.holdout_metrics["near_miss_correct"] += (
                            holdout_near_miss * batch_size
                        )
                        self.holdout_metrics["total_samples"] += batch_size

                    # per-task results for detailed analysis
                    task_idx = (
                        task_indices[batch_idx * self.config.batch_size + i]
                        if batch_idx * self.config.batch_size + i < len(task_indices)
                        else batch_idx
                    )

                    # calculate per-sample metrics using test target
                    sample_logits = test_logits[0:1]  # [1, 10, 30, 30]
                    sample_target = test_target[0:1]  # [1, 1, 30, 30]

                    per_task_results.append(
                        {
                            "task_index": task_idx,
                            "perfect_match": calculate_perfect_accuracy(
                                sample_logits, sample_target
                            ),
                            "pixel_accuracy": calculate_pixel_accuracy(
                                sample_logits, sample_target
                            ),
                            "near_miss_accuracy": calculate_near_miss_accuracy(
                                sample_logits, sample_target
                            ),
                            "perfect_match_foreground": calculate_perfect_accuracy_foreground(
                                sample_logits, sample_target
                            ),
                            "pixel_accuracy_foreground": calculate_pixel_accuracy_foreground(
                                sample_logits, sample_target
                            ),
                            "near_miss_accuracy_foreground": calculate_near_miss_accuracy_foreground(
                                sample_logits, sample_target
                            ),
                        }
                    )

        # calculate final metrics (same as evaluate.py)
        results = {
            "perfect_accuracy": perfect_matches / total_samples,
            "pixel_accuracy": pixel_correct / total_samples,
            "near_miss_accuracy": near_miss_correct / total_samples,
            "total_samples": total_samples,
            "per_task_results": per_task_results,
            # foreground metrics
            "perfect_accuracy_foreground": perfect_matches_foreground / total_samples,
            "pixel_accuracy_foreground": pixel_correct_foreground / total_samples,
            "near_miss_accuracy_foreground": near_miss_correct_foreground
            / total_samples,
        }

        # add holdout metrics if available
        if (
            hasattr(self, "holdout_metrics")
            and self.holdout_metrics["total_samples"] > 0
        ):
            results.update(
                {
                    "holdout_perfect_accuracy": self.holdout_metrics["perfect_matches"]
                    / self.holdout_metrics["total_samples"],
                    "holdout_pixel_accuracy": self.holdout_metrics["pixel_correct"]
                    / self.holdout_metrics["total_samples"],
                    "holdout_near_miss_accuracy": self.holdout_metrics[
                        "near_miss_correct"
                    ]
                    / self.holdout_metrics["total_samples"],
                    "holdout_total_samples": self.holdout_metrics["total_samples"],
                }
            )

        # save results
        with open(self.experiment_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # print results (same format as evaluate.py)
        print("\nevaluation results:")
        print(
            f"  perfect accuracy: {results['perfect_accuracy']:.4f} ({results['perfect_accuracy']*100:.2f}%)"
        )
        print(
            f"  pixel accuracy: {results['pixel_accuracy']:.4f} ({results['pixel_accuracy']*100:.2f}%)"
        )
        print(
            f"  near-miss accuracy: {results['near_miss_accuracy']:.4f} ({results['near_miss_accuracy']*100:.2f}%)"
        )
        print(f"  total samples: {results['total_samples']}")

        print("\nforeground results (non-background pixels only):")
        print(
            f"  perfect accuracy (foreground): {results['perfect_accuracy_foreground']:.4f} ({results['perfect_accuracy_foreground']*100:.2f}%)"
        )
        print(
            f"  pixel accuracy (foreground): {results['pixel_accuracy_foreground']:.4f} ({results['pixel_accuracy_foreground']*100:.2f}%)"
        )
        print(
            f"  near-miss accuracy (foreground): {results['near_miss_accuracy_foreground']:.4f} ({results['near_miss_accuracy_foreground']*100:.2f}%)"
        )

        # print holdout results if available
        if "holdout_perfect_accuracy" in results:
            print("\nholdout validation results:")
            print(
                f"  perfect accuracy (holdout): {results['holdout_perfect_accuracy']:.4f} ({results['holdout_perfect_accuracy']*100:.2f}%)"
            )
            print(
                f"  pixel accuracy (holdout): {results['holdout_pixel_accuracy']:.4f} ({results['holdout_pixel_accuracy']*100:.2f}%)"
            )
            print(
                f"  near-miss accuracy (holdout): {results['holdout_near_miss_accuracy']:.4f} ({results['holdout_near_miss_accuracy']*100:.2f}%)"
            )
            print(f"  holdout samples: {results['holdout_total_samples']}")

        return results


def evaluate_existing_checkpoint(experiment_dir: str, config: Config):
    """evaluate an existing checkpoint from an experiment directory."""
    experiment_path = Path(experiment_dir)
    if not experiment_path.exists():
        print(f"experiment directory not found: {experiment_dir}")
        return

    checkpoint_path = experiment_path / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"checkpoint not found: {checkpoint_path}")
        return

    print(f"evaluating checkpoint: {checkpoint_path}")

    # create experiment instance to use existing evaluation logic
    experiment = OverfitExperiment(config, experiment_path.name)

    # load the task selection from the experiment
    task_selection_file = experiment_path / "task_selection.json"
    if task_selection_file.exists():
        with open(task_selection_file, "r") as f:
            task_data = json.load(f)
        task_indices = task_data.get(
            "task_indices", list(range(10))
        )  # fallback to first 10 tasks
    else:
        # if no task selection file, evaluate on first 10 tasks
        task_indices = list(range(10))

    print(f"evaluating on {len(task_indices)} tasks...")

    # use existing evaluation function
    results = experiment.evaluate_on_tasks(task_indices, str(checkpoint_path))

    # print results
    print("\nevaluation results:")
    print(
        f"  perfect accuracy: {results['perfect_accuracy']:.4f} ({results['perfect_accuracy']*100:.2f}%)"
    )
    print(
        f"  pixel accuracy: {results['pixel_accuracy']:.4f} ({results['pixel_accuracy']*100:.2f}%)"
    )
    print(
        f"  near-miss accuracy: {results['near_miss_accuracy']:.4f} ({results['near_miss_accuracy']*100:.2f}%)"
    )
    print(f"  total samples: {results['total_samples']}")

    print("\nforeground results (non-background pixels only):")
    print(
        f"  perfect accuracy (foreground): {results['perfect_accuracy_foreground']:.4f} ({results['perfect_accuracy_foreground']*100:.2f}%)"
    )
    print(
        f"  pixel accuracy (foreground): {results['pixel_accuracy_foreground']:.4f} ({results['pixel_accuracy_foreground']*100:.2f}%)"
    )
    print(
        f"  near-miss accuracy (foreground): {results['near_miss_accuracy_foreground']:.4f} ({results['near_miss_accuracy_foreground']*100:.2f}%)"
    )

    # save results
    results_file = experiment_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nresults saved to: {results_file}")


def main():
    """main experiment function."""
    parser = argparse.ArgumentParser(description="n-task overfitting experiment")
    parser.add_argument(
        "--n_tasks", "-n", type=int, default=10, help="number of tasks to overfit on"
    )
    parser.add_argument(
        "--task-indices", type=int, nargs="+", help="specific task indices to use"
    )
    parser.add_argument("--epochs", type=int, help="maximum epochs")
    parser.add_argument("--patience", type=int, help="early stopping patience")
    parser.add_argument("--experiment-name", type=str, help="experiment name")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="config file path"
    )
    parser.add_argument("--batch-size", type=int, help="override batch size")
    parser.add_argument("--lr", type=float, help="override learning rate")
    parser.add_argument(
        "--evaluate",
        type=str,
        help="evaluate existing checkpoint (path to experiment directory)",
    )

    args = parser.parse_args()

    # load configuration
    config = Config()

    # override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    if args.epochs:
        config.num_epochs = args.epochs
    if args.patience:
        config.early_stopping_patience = args.patience

    # set up deterministic training
    config.set_deterministic_training()

    # handle evaluation mode
    if args.evaluate:
        evaluate_existing_checkpoint(args.evaluate, config)
        return

    print(f"overfitting experiment: {args.n_tasks} tasks")
    print("configuration:")
    print(f"  device: {config.device}")
    print(f"  batch size: {config.batch_size}")
    print(f"  learning rate: {config.learning_rate}")
    print(f"  max epochs: {config.num_epochs}")
    print(f"  early stopping patience: {config.early_stopping_patience}")

    # create experiment
    experiment = OverfitExperiment(config, args.experiment_name)

    # select tasks
    task_indices = experiment.select_tasks(
        args.n_tasks, args.task_indices, config.random_seed
    )

    # train on selected tasks
    checkpoint_path = experiment.train_on_tasks(
        task_indices, config.num_epochs, config.early_stopping_patience
    )

    # evaluate on same tasks
    results = experiment.evaluate_on_tasks(task_indices, checkpoint_path)

    print("\nexperiment completed!")
    print(f"results saved to: {experiment.experiment_dir}")
    print(f"checkpoint saved to: {experiment.checkpoints_dir}/best_model.pt")
    print(f"perfect accuracy: {results['perfect_accuracy']:.4f}")


if __name__ == "__main__":
    main()
