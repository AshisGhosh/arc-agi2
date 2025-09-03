import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict
import os
from tqdm import tqdm

from .hrm import HRM, FewShotBatch, hrm_loss


class HRMTrainer:
    """HRM trainer with ACT support"""

    def __init__(self, model: HRM, config: Dict):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # separate optimizers for main model and puzzle embeddings
        main_params = []
        puzzle_emb_params = []

        for name, param in model.named_parameters():
            if "puzzle_emb" in name:
                puzzle_emb_params.append(param)
            else:
                main_params.append(param)

        # main optimizer
        self.optimizer = AdamW(
            main_params, lr=config["training"]["lr"], weight_decay=0.1
        )

        # puzzle embedding optimizer (higher learning rate)
        if puzzle_emb_params:
            self.puzzle_emb_optimizer = AdamW(
                puzzle_emb_params,
                lr=config["training"]["puzzle_emb_lr"],
                weight_decay=0.1,
            )
        else:
            self.puzzle_emb_optimizer = None

        # per-epoch cosine annealing
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config["training"]["epochs"], eta_min=1e-6
        )

        # training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_perfect = 0.0

        # create checkpoint directory
        os.makedirs("checkpoints", exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_q_loss = 0.0
        total_cell_acc = 0.0
        total_perfect_acc = 0.0
        num_batches = 0

        for batch_data in tqdm(train_loader, desc="Training", total=len(train_loader)):
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
            support_inp = support_inp.to(self.device)
            support_out = support_out.to(self.device)
            query_inp = query_inp.to(self.device)
            query_out = query_out.to(self.device)
            task_id = task_id.to(self.device)

            # create few-shot batch
            batch = FewShotBatch(
                support_inp=support_inp,
                support_out=support_out,
                query_inp=query_inp,
                query_out=query_out,
                task_id=task_id,
            )

            # forward pass
            outputs = self.model(batch)

            # compute loss
            loss, metrics = hrm_loss(outputs, query_out, self.config["loss"])

            # backward pass
            self.optimizer.zero_grad()
            if self.puzzle_emb_optimizer:
                self.puzzle_emb_optimizer.zero_grad()

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["training"]["grad_clip"]
            )

            self.optimizer.step()
            if self.puzzle_emb_optimizer:
                self.puzzle_emb_optimizer.step()

            # accumulate metrics
            total_loss += metrics["loss"]
            total_ce_loss += metrics["ce_loss"]
            total_q_loss += metrics["q_loss"]
            total_cell_acc += metrics["cell_acc"]
            total_perfect_acc += metrics["perfect_acc"]
            num_batches += 1

        return {
            "train_loss": total_loss / num_batches,
            "train_ce_loss": total_ce_loss / num_batches,
            "train_q_loss": total_q_loss / num_batches,
            "train_cell_acc": total_cell_acc / num_batches,
            "train_perfect_acc": total_perfect_acc / num_batches,
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """validate model"""
        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_q_loss = 0.0
        total_cell_acc = 0.0
        total_perfect_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in tqdm(val_loader):
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
                support_inp = support_inp.to(self.device)
                support_out = support_out.to(self.device)
                query_inp = query_inp.to(self.device)
                query_out = query_out.to(self.device)
                task_id = task_id.to(self.device)

                # create few-shot batch
                batch = FewShotBatch(
                    support_inp=support_inp,
                    support_out=support_out,
                    query_inp=query_inp,
                    query_out=query_out,
                    task_id=task_id,
                )

                # forward pass
                outputs = self.model(batch)

                # compute loss
                loss, metrics = hrm_loss(outputs, query_out, self.config["loss"])

                # accumulate metrics
                total_loss += metrics["loss"]
                total_ce_loss += metrics["ce_loss"]
                total_q_loss += metrics["q_loss"]
                total_cell_acc += metrics["cell_acc"]
                total_perfect_acc += metrics["perfect_acc"]
                num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_ce_loss": total_ce_loss / num_batches,
            "val_q_loss": total_q_loss / num_batches,
            "val_cell_acc": total_cell_acc / num_batches,
            "val_perfect_acc": total_perfect_acc / num_batches,
            "cell_acc_last": total_cell_acc / num_batches,
            "perfect_grid_acc_last": total_perfect_acc / num_batches,
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """full training loop"""
        history = {
            "train_loss": [],
            "train_ce_loss": [],
            "train_q_loss": [],
            "train_cell_acc": [],
            "train_perfect_acc": [],
            "val_loss": [],
            "val_ce_loss": [],
            "val_q_loss": [],
            "val_cell_acc": [],
            "val_perfect_acc": [],
        }

        for epoch in range(self.config["training"]["epochs"]):
            self.epoch = epoch

            # train
            train_metrics = self.train_epoch(train_loader)

            # validate
            val_metrics = self.validate(val_loader)

            # update scheduler
            self.scheduler.step()

            # update history
            for key in history:
                if key.startswith("train_"):
                    history[key].append(train_metrics[key])
                elif key.startswith("val_"):
                    history[key].append(val_metrics[key])

            # print progress
            print(
                f"epoch {epoch + 1}/{self.config['training']['epochs']}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"val_perfect={val_metrics['val_perfect_acc']:.3f}"
            )

            # save checkpoint if best
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_val_perfect = val_metrics["val_perfect_acc"]
                self.save_checkpoint("best.pt")

            # save latest checkpoint
            self.save_checkpoint("latest.pt")

        return history

    def save_checkpoint(self, filename: str):
        """save model checkpoint"""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_perfect": self.best_val_perfect,
            "config": self.config,
        }

        if self.puzzle_emb_optimizer:
            checkpoint["puzzle_emb_optimizer_state_dict"] = (
                self.puzzle_emb_optimizer.state_dict()
            )

        torch.save(checkpoint, f"checkpoints/{filename}")

    def load_checkpoint(self, filename: str):
        """load model checkpoint"""
        checkpoint = torch.load(f"checkpoints/{filename}", map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if (
            self.puzzle_emb_optimizer
            and "puzzle_emb_optimizer_state_dict" in checkpoint
        ):
            self.puzzle_emb_optimizer.load_state_dict(
                checkpoint["puzzle_emb_optimizer_state_dict"]
            )

        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_perfect = checkpoint["best_val_perfect"]

        print(f"loaded checkpoint from epoch {self.epoch}")
        print(f"best validation loss: {self.best_val_loss:.4f}")
        print(f"best validation perfect: {self.best_val_perfect:.3f}")
