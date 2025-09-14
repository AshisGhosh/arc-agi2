#!/usr/bin/env python3
"""
overfit experiment script for n-task overfitting.

allows you to select n specific tasks, overfit on them, and evaluate
with clean organization and reporting.
"""

import torch
from torch.utils.data import DataLoader, Subset
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime

from algo.config import Config
from algo.models import SimpleARCModel
from algo.data import ARCDataset
from algo.training import ARCTrainer
from scripts.evaluate import (
    calculate_perfect_accuracy,
    calculate_pixel_accuracy,
    calculate_near_miss_accuracy,
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
        # load full dataset to get total number of tasks
        full_dataset = ARCDataset(self.config.processed_dir, self.config)
        total_tasks = len(full_dataset)

        if task_indices is not None:
            # validate indices
            invalid_indices = [
                idx for idx in task_indices if idx >= total_tasks or idx < 0
            ]
            if invalid_indices:
                raise ValueError(
                    f"invalid task indices: {invalid_indices}. total tasks: {total_tasks}"
                )
            selected_indices = task_indices[:n_tasks]
        else:
            # random selection
            random.seed(random_seed)
            selected_indices = random.sample(
                range(total_tasks), min(n_tasks, total_tasks)
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
        }

        with open(self.experiment_dir / "task_selection.json", "w") as f:
            json.dump(selection_info, f, indent=2)

        return selected_indices

    def create_task_subset(self, task_indices: List[int]) -> ARCDataset:
        """create dataset subset with only selected tasks."""
        full_dataset = ARCDataset(self.config.processed_dir, self.config)
        subset = Subset(full_dataset, task_indices)

        # wrap in custom dataset class to maintain interface
        class TaskSubset(ARCDataset):
            def __init__(self, subset, original_dataset):
                self.subset = subset
                self.original_dataset = original_dataset
                self.config = original_dataset.config
                self.processed_dir = original_dataset.processed_dir
                self.dataset_name = original_dataset.dataset_name
                self.dataset_path = original_dataset.dataset_path

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                return self.subset[idx]

        return TaskSubset(subset, full_dataset)

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
        )

        # create model
        model = SimpleARCModel(self.config)

        # create trainer with overfitting config
        trainer = ARCTrainer(model, self.config)

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

            # train one epoch using existing trainer method
            train_metrics = trainer.train_epoch(train_loader)
            avg_loss = train_metrics["total_loss"]

            # update scheduler
            trainer.scheduler.step()

            # log progress
            if epoch % 10 == 0 or epoch < 10:
                elapsed = time.time() - start_time
                print(
                    f"epoch {epoch:4d}: loss={avg_loss:.6f} (elapsed: {elapsed:.1f}s)"
                )

            # save best model
            if avg_loss < best_loss:
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
            "task_indices": task_indices,
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
        )

        # create model and load checkpoint
        model = SimpleARCModel(self.config)
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
        total_l1_loss = 0.0
        total_l2_loss = 0.0

        # per-task results
        per_task_results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                # move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # forward pass
                solution = model(
                    batch["example1_input"],
                    batch["example1_output"],
                    batch["example2_input"],
                    batch["example2_output"],
                    batch["target_input"],
                )

                # calculate metrics using existing functions
                batch_size = solution.size(0)
                total_samples += batch_size

                # use existing accuracy functions
                perfect_matches += (
                    calculate_perfect_accuracy(solution, batch["target_output"])
                    * batch_size
                )

                pixel_correct += (
                    calculate_pixel_accuracy(solution, batch["target_output"])
                    * batch_size
                )

                near_miss_correct += (
                    calculate_near_miss_accuracy(solution, batch["target_output"])
                    * batch_size
                )

                # losses
                l1_loss = torch.nn.functional.l1_loss(solution, batch["target_output"])
                l2_loss = torch.nn.functional.mse_loss(solution, batch["target_output"])
                total_l1_loss += l1_loss.item() * batch_size
                total_l2_loss += l2_loss.item() * batch_size

                # per-task results for detailed analysis
                for i in range(batch_size):
                    task_idx = (
                        task_indices[batch_idx * self.config.batch_size + i]
                        if batch_idx * self.config.batch_size + i < len(task_indices)
                        else batch_idx
                    )

                    # calculate per-sample metrics
                    sample_solution = solution[i : i + 1]
                    sample_target = batch["target_output"][i : i + 1]

                    per_task_results.append(
                        {
                            "task_index": task_idx,
                            "perfect_match": calculate_perfect_accuracy(
                                sample_solution, sample_target
                            ),
                            "pixel_accuracy": calculate_pixel_accuracy(
                                sample_solution, sample_target
                            ),
                            "near_miss_accuracy": calculate_near_miss_accuracy(
                                sample_solution, sample_target
                            ),
                            "l1_loss": l1_loss.item(),
                            "l2_loss": l2_loss.item(),
                        }
                    )

        # calculate final metrics (same as evaluate.py)
        results = {
            "perfect_accuracy": perfect_matches / total_samples,
            "pixel_accuracy": pixel_correct / total_samples,
            "near_miss_accuracy": near_miss_correct / total_samples,
            "l1_loss": total_l1_loss / total_samples,
            "l2_loss": total_l2_loss / total_samples,
            "total_samples": total_samples,
            "per_task_results": per_task_results,
        }

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
        print(f"  l1 loss: {results['l1_loss']:.4f}")
        print(f"  l2 loss: {results['l2_loss']:.4f}")
        print(f"  total samples: {results['total_samples']}")

        return results


def main():
    """main experiment function."""
    parser = argparse.ArgumentParser(description="n-task overfitting experiment")
    parser.add_argument(
        "--n_tasks", "-n", type=int, default=10, help="number of tasks to overfit on"
    )
    parser.add_argument(
        "--task-indices", type=int, nargs="+", help="specific task indices to use"
    )
    parser.add_argument("--epochs", type=int, default=1000, help="maximum epochs")
    parser.add_argument(
        "--patience", type=int, default=50, help="early stopping patience"
    )
    parser.add_argument("--experiment-name", type=str, help="experiment name")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="config file path"
    )
    parser.add_argument("--batch-size", type=int, help="override batch size")
    parser.add_argument("--lr", type=float, help="override learning rate")

    args = parser.parse_args()

    # load configuration
    config = Config()

    # override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    # set up deterministic training
    config.set_deterministic_training()

    print(f"overfitting experiment: {args.n_tasks} tasks")
    print("configuration:")
    print(f"  device: {config.device}")
    print(f"  batch size: {config.batch_size}")
    print(f"  learning rate: {config.learning_rate}")
    print(f"  max epochs: {args.epochs}")
    print(f"  early stopping patience: {args.patience}")

    # create experiment
    experiment = OverfitExperiment(config, args.experiment_name)

    # select tasks
    task_indices = experiment.select_tasks(
        args.n_tasks, args.task_indices, config.random_seed
    )

    # train on selected tasks
    checkpoint_path = experiment.train_on_tasks(
        task_indices, args.epochs, args.patience
    )

    # evaluate on same tasks
    results = experiment.evaluate_on_tasks(task_indices, checkpoint_path)

    print("\nexperiment completed!")
    print(f"results saved to: {experiment.experiment_dir}")
    print(f"checkpoint saved to: {experiment.checkpoints_dir}/best_model.pt")
    print(f"perfect accuracy: {results['perfect_accuracy']:.4f}")


if __name__ == "__main__":
    main()
