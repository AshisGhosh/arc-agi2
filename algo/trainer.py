import os
import yaml
import torch
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR

from .hrm import HRM, FewShotBatch, HRMLossCfg, hrm_loss
from utils.metrics import compute_cycle_metrics
from utils.augmentation import augment_pair


class HRMTrainer:
    """extended HRMSystem with training infrastructure"""

    def __init__(self, model: HRM, config_path: str):
        self.model = model
        self.config = self._load_config(config_path)

        # optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["training"]["lr"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        # cosine scheduler with warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config["training"]["epochs"], eta_min=1e-6
        )

        # training state
        self.epoch = 0
        self.best_val_perfect = 0.0
        self.patience_counter = 0
        self.patience = 15

        # loss config
        self.loss_cfg = HRMLossCfg(
            ds_weight=self.config["loss"]["ds_weight"],
            ds_decay=self.config["loss"]["ds_decay"],
            act_weight=self.config["loss"]["act_weight"],
        )

        # create checkpoint directory
        os.makedirs("checkpoints", exist_ok=True)

    def _load_config(self, config_path: str) -> Dict:
        """load YAML config"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _warmup_lr(self, step: int):
        """linear warmup for first warmup_steps"""
        if step < self.config["training"]["warmup_steps"]:
            warmup_factor = step / self.config["training"]["warmup_steps"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.config["training"]["lr"] * warmup_factor

    def _create_few_shot_batch(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        k_support: int,
        batch_size: int,
    ) -> FewShotBatch:
        """create few-shot batch from pairs"""
        # randomly sample batch_size tasks
        task_indices = torch.randperm(len(pairs))[:batch_size]

        support_inp = []
        support_out = []
        query_inp = []
        query_out = []

        for task_idx in task_indices:
            # for each task, randomly sample k_support pairs for support
            task_pairs = pairs[task_idx]
            if len(task_pairs) < k_support:
                # duplicate if not enough pairs
                task_pairs = task_pairs * (k_support // len(task_pairs) + 1)

            # sample k_support support pairs
            support_indices = torch.randperm(len(task_pairs))[:k_support]
            task_support_inp = []
            task_support_out = []

            for sup_idx in support_indices:
                inp, out = task_pairs[sup_idx]
                # apply augmentation
                inp_aug, out_aug = augment_pair(inp, out)
                task_support_inp.append(inp_aug)
                task_support_out.append(out_aug)

            # sample one pair for query
            query_idx = torch.randint(0, len(task_pairs), (1,)).item()
            inp, out = task_pairs[query_idx]
            inp_aug, out_aug = augment_pair(inp, out)

            support_inp.append(torch.stack(task_support_inp))
            support_out.append(torch.stack(task_support_out))
            query_inp.append(inp_aug)
            query_out.append(out_aug)

        return FewShotBatch(
            support_inp=torch.stack(support_inp),  # (B, K, L)
            support_out=torch.stack(support_out),  # (B, K, L)
            query_inp=torch.stack(query_inp),  # (B, L)
            query_out=torch.stack(query_out),  # (B, L)
        )

    def train_epoch(
        self, train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """train for one epoch"""
        self.model.train()

        batch_size = self.config["training"]["batch_size"]
        k_support = self.config["data"]["k_support"]
        cycles = self.config["hrm"]["cycles"]
        inner_steps = self.config["hrm"]["inner_steps"]

        # create batches
        n_batches = len(train_pairs) // batch_size
        total_loss = 0.0
        total_metrics = {}

        for batch_idx in range(n_batches):
            # create few-shot batch
            batch = self._create_few_shot_batch(train_pairs, k_support, batch_size)

            # forward pass
            outputs = self.model(
                batch,
                cycles=cycles,
                inner_steps=inner_steps,
                one_step_grad=self.config["hrm"]["one_step_grad"],
            )

            # compute loss
            loss, metrics = hrm_loss(outputs, batch.query_out, self.loss_cfg)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["training"]["grad_clip"]
            )
            self.optimizer.step()

            # update scheduler
            step = self.epoch * n_batches + batch_idx
            self._warmup_lr(step)
            if step >= self.config["training"]["warmup_steps"]:
                self.scheduler.step()

            # accumulate metrics
            total_loss += loss.item()
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value

        # average metrics
        avg_metrics = {key: value / n_batches for key, value in total_metrics.items()}
        avg_metrics["loss"] = total_loss / n_batches

        return avg_metrics

    def validate(
        self, val_pairs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """validate model"""
        self.model.eval()

        batch_size = self.config["training"]["batch_size"]
        k_support = self.config["data"]["k_support"]
        cycles = self.config["hrm"]["cycles"]
        inner_steps = self.config["hrm"]["inner_steps"]

        # create validation batch
        batch = self._create_few_shot_batch(val_pairs, k_support, batch_size)

        with torch.no_grad():
            outputs = self.model(
                batch,
                cycles=cycles,
                inner_steps=inner_steps,
                one_step_grad=False,  # no gradient needed for validation
            )

            # compute metrics
            metrics = compute_cycle_metrics(
                outputs["logits_per_cycle"], batch.query_out
            )

            # add loss
            loss, loss_metrics = hrm_loss(outputs, batch.query_out, self.loss_cfg)
            metrics["val_loss"] = loss.item()
            metrics.update(loss_metrics)

        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """save model checkpoint"""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_perfect": self.best_val_perfect,
            "config": self.config,
        }

        # save latest
        torch.save(checkpoint, "checkpoints/latest.pt")

        # save best if better
        if is_best:
            torch.save(checkpoint, "checkpoints/best.pt")

    def load_checkpoint(self, checkpoint_path: str):
        """load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_perfect = checkpoint["best_val_perfect"]

    def train(
        self,
        train_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        val_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, List[float]]:
        """main training loop"""
        history = {"train_loss": [], "val_loss": [], "val_perfect_grid_last": []}

        for epoch in range(self.config["training"]["epochs"]):
            self.epoch = epoch

            # training
            train_metrics = self.train_epoch(train_pairs)

            # validation
            val_metrics = self.validate(val_pairs)

            # update learning rate
            if epoch >= self.config["training"]["warmup_steps"]:
                self.scheduler.step()

            # early stopping check
            val_perfect = val_metrics["perfect_grid_acc_last"]
            is_best = val_perfect > self.best_val_perfect

            if is_best:
                self.best_val_perfect = val_perfect
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1

            # save checkpoint
            self.save_checkpoint()

            # log metrics
            print(
                f"epoch {epoch + 1:3d}: "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"val_perfect={val_perfect:.3f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # store history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_perfect_grid_last"].append(val_perfect)

            # early stopping
            if self.patience_counter >= self.patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

        return history
