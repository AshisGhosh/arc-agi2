import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from typing import Dict
import os
from tqdm import tqdm

from .hrm import HRM, hrm_loss


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

        # create parameter groups with proper weight decay separation
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if "puzzle_emb" in name:
                continue  # handle puzzle embeddings separately
            if param.ndim >= 2 and "embedding" not in name:
                decay_params.append(param)  # weights
            else:
                no_decay_params.append(param)  # biases, norms, embeddings

        # main optimizer with proper weight decay
        param_groups = [
            {"params": decay_params, "weight_decay": 1e-4},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(param_groups, lr=config["training"]["lr"])

        # puzzle embedding optimizer (higher learning rate, no weight decay)
        if puzzle_emb_params:
            self.puzzle_emb_optimizer = AdamW(
                puzzle_emb_params,
                lr=config["training"]["puzzle_emb_lr"],
                weight_decay=0.0,  # no weight decay for embeddings
            )
        else:
            self.puzzle_emb_optimizer = None

        # warmup + cosine annealing scheduler
        warmup_steps = config.get("warmup_steps", 1000)
        total_epochs = config["training"]["epochs"]
        
        # linear warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.1, 
            total_iters=warmup_steps
        )
        
        # cosine annealing scheduler (after warmup)
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=total_epochs - warmup_steps, 
            eta_min=1e-6
        )
        
        # sequential scheduler: warmup then cosine
        self.scheduler = SequentialLR(
            self.optimizer, 
            [warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_steps]
        )

        # training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_perfect = 0.0

        # create checkpoint directory
        os.makedirs("checkpoints", exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """train for one epoch with ACT"""
        self.model.train()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_q_halt_loss = 0.0
        total_q_continue_loss = 0.0
        total_accuracy = 0.0
        total_exact_accuracy = 0.0
        total_steps = 0.0
        total_count = 0.0
        num_batches = 0

        for batch_data in tqdm(train_loader, desc="Training", total=len(train_loader)):
            # unpack batch data (must include support_pairs and file_names)
            if len(batch_data) == 5:
                inputs, labels, puzzle_identifiers, support_pairs_list, file_names = batch_data
            else:
                raise ValueError(f"Expected 5 items in batch (inputs, labels, puzzle_identifiers, support_pairs, file_names), got {len(batch_data)}")

            # move to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            puzzle_identifiers = puzzle_identifiers.to(self.device)

            # create batch dict
            batch = {
                "inputs": inputs,
                "labels": labels,
                "puzzle_identifiers": puzzle_identifiers,
            }

            # initialize carry for each batch
            carry = self.model.initial_carry(batch)

            # forward pass with ACT unrolling and support examples
            max_steps = self.config.get("halt_max_steps", 16)
            for step in range(max_steps):
                carry, outputs = self.model(carry, batch, support_pairs_list)

            # compute loss
            loss, metrics = hrm_loss(
                outputs, labels, carry, self.config.get("loss", {})
            )

            # backward pass
            self.optimizer.zero_grad()
            if self.puzzle_emb_optimizer:
                self.puzzle_emb_optimizer.zero_grad()

            loss.backward()
            
            # Check for NaN gradients before clipping
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    assert False, f"NaN gradient detected in {name} before clipping!"

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["training"]["grad_clip"]
            )
            
            # Check for NaN gradients after clipping
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    assert False, f"NaN gradient detected in {name} after clipping!"

            # Check weights before optimizer step
            for name, param in self.model.named_parameters():
                assert not torch.isnan(param).any(), f"NaN weight detected in {name} before optimizer step!"

            self.optimizer.step()
            if self.puzzle_emb_optimizer:
                self.puzzle_emb_optimizer.step()
            
            # Check weights after optimizer step
            for name, param in self.model.named_parameters():
                assert not torch.isnan(param).any(), f"NaN weight detected in {name} after optimizer step!"

            # accumulate metrics
            total_loss += float(loss.item())
            total_lm_loss += float(metrics["lm_loss"].item())
            total_q_halt_loss += float(metrics["q_halt_loss"].item())
            total_q_continue_loss += float(metrics["q_continue_loss"].item())
            total_accuracy += float(metrics["accuracy"].item())
            total_exact_accuracy += float(metrics["exact_accuracy"].item())
            total_steps += float(metrics["steps"].item())
            total_count += float(metrics["count"].item())
            num_batches += 1

        return {
            "train_loss": total_loss / num_batches,
            "train_lm_loss": total_lm_loss / num_batches,
            "train_q_halt_loss": total_q_halt_loss / num_batches,
            "train_q_continue_loss": total_q_continue_loss / num_batches,
            "train_accuracy": total_accuracy / max(total_count, 1),
            "train_exact_accuracy": total_exact_accuracy / max(total_count, 1),
            "train_avg_steps": total_steps / max(total_count, 1),
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """validate model with ACT"""
        self.model.eval()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_q_halt_loss = 0.0
        total_q_continue_loss = 0.0
        total_accuracy = 0.0
        total_exact_accuracy = 0.0
        total_steps = 0.0
        total_count = 0.0
        num_batches = 0
        
        # track perfect accuracy samples
        perfect_samples = []
        perfect_files = []

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                # unpack batch data (must include support_pairs and file_names)
                if len(batch_data) == 5:
                    inputs, labels, puzzle_identifiers, support_pairs_list, file_names = batch_data
                else:
                    raise ValueError(f"Expected 5 items in batch (inputs, labels, puzzle_identifiers, support_pairs, file_names), got {len(batch_data)}")

                # move to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                puzzle_identifiers = puzzle_identifiers.to(self.device)

                # create batch dict
                batch = {
                    "inputs": inputs,
                    "labels": labels,
                    "puzzle_identifiers": puzzle_identifiers,
                }

                # initialize carry for each batch
                carry = self.model.initial_carry(batch)

                # forward pass with ACT unrolling and support examples (same as training for consistency)
                max_steps = self.config.get("halt_max_steps", 16)
                for step in range(max_steps):
                    carry, outputs = self.model(carry, batch, support_pairs_list)

                # compute loss
                loss, metrics = hrm_loss(
                    outputs, labels, carry, self.config.get("loss", {})
                )

                # check for perfect accuracy samples
                predictions = torch.argmax(outputs["logits"], dim=-1)
                perfect_mask = (predictions == labels).all(dim=1)  # (B,)
                perfect_indices = torch.where(perfect_mask)[0]
                
                # track perfect samples with their puzzle identifiers and file names
                for idx in perfect_indices:
                    puzzle_id = puzzle_identifiers[idx].item()
                    file_name = file_names[idx]
                    perfect_samples.append(puzzle_id)
                    perfect_files.append(file_name)
                
                # accumulate metrics
                total_loss += float(loss.item())
                total_lm_loss += float(metrics["lm_loss"].item())
                total_q_halt_loss += float(metrics["q_halt_loss"].item())
                total_q_continue_loss += float(metrics["q_continue_loss"].item())
                total_accuracy += float(metrics["accuracy"].item())
                total_exact_accuracy += float(metrics["exact_accuracy"].item())
                total_steps += float(metrics["steps"].item())
                total_count += float(metrics["count"].item())
                num_batches += 1

        # print perfect accuracy samples
        if perfect_samples:
            print(f"\n🎯 Found {len(perfect_samples)} samples with perfect accuracy!")
            perfect_tasks = set([puzzle_id for puzzle_id in perfect_samples])
            print(f"Perfect accuracy tasks: {perfect_tasks}")
            print("Perfect accuracy files (one per task):")
            
            # Group by task and show one file per task
            task_to_files = {}
            for puzzle_id, file_name in zip(perfect_samples, perfect_files):
                if puzzle_id not in task_to_files:
                    task_to_files[puzzle_id] = file_name
            
            for i, (puzzle_id, file_name) in enumerate(task_to_files.items()):
                print(f"  {i+1}. {file_name} (Puzzle ID: {puzzle_id})")
            
            if len(perfect_samples) > len(task_to_files):
                print(f"  ... and {len(perfect_samples) - len(task_to_files)} more files from these tasks")
        else:
            print("\n❌ No samples achieved perfect accuracy")
        
        return {
            "val_loss": total_loss / num_batches,
            "val_lm_loss": total_lm_loss / num_batches,
            "val_q_halt_loss": total_q_halt_loss / num_batches,
            "val_q_continue_loss": total_q_continue_loss / num_batches,
            "val_accuracy": total_accuracy / max(total_count, 1),
            "val_exact_accuracy": total_exact_accuracy / max(total_count, 1),
            "val_avg_steps": total_steps / max(total_count, 1),
            "perfect_samples": perfect_samples,
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """full training loop"""
        history = {
            "train_loss": [],
            "train_lm_loss": [],
            "train_q_halt_loss": [],
            "train_q_continue_loss": [],
            "train_accuracy": [],
            "train_exact_accuracy": [],
            "train_avg_steps": [],
            "val_loss": [],
            "val_lm_loss": [],
            "val_q_halt_loss": [],
            "val_q_continue_loss": [],
            "val_accuracy": [],
            "val_exact_accuracy": [],
            "val_avg_steps": [],
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
                f"val_exact_acc={val_metrics['val_exact_accuracy']:.3f}, "
                f"val_avg_steps={val_metrics['val_avg_steps']:.1f}"
            )

            # save checkpoint if best
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_val_perfect = val_metrics["val_exact_accuracy"]
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

        # Debug: Check for NaN in checkpoint before loading
        print("DEBUG: Checking checkpoint for NaN weights...")
        model_state = checkpoint["model_state_dict"]
        nan_count = 0
        for key, tensor in model_state.items():
            if torch.isnan(tensor).any():
                nan_count += 1
                print(f"ERROR: NaN found in checkpoint key: {key}")
                if "embed_tokens" in key:
                    print(f"  embed_tokens weight stats: mean={tensor.mean():.6f}, std={tensor.std():.6f}")
        
        if nan_count > 0:
            print(f"WARNING: Found {nan_count} tensors with NaN in checkpoint!")
            print("Attempting to fix NaN weights by reinitializing...")
            
            # Fix NaN weights by reinitializing them
            for key, tensor in model_state.items():
                if torch.isnan(tensor).any():
                    print(f"Reinitializing {key}...")
                    if "embed_tokens" in key:
                        # Reinitialize embedding weights
                        with torch.no_grad():
                            model_state[key] = torch.randn_like(tensor, dtype=tensor.dtype) * 0.001
                    elif "weight" in key:
                        # Reinitialize other weights
                        with torch.no_grad():
                            model_state[key] = torch.randn_like(tensor, dtype=tensor.dtype) * 0.001
                    elif "bias" in key:
                        # Reinitialize biases to zero
                        with torch.no_grad():
                            model_state[key] = torch.zeros_like(tensor, dtype=tensor.dtype)
            
            print("NaN weights fixed. Proceeding with loading...")

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
