from dataclasses import dataclass
from typing import Tuple, List
import os
import torch
import random
import numpy as np


@dataclass
class Config:
    """Configuration for SimpleARC model training."""

    # Model architecture
    rule_dim: int = 128
    input_size: Tuple[int, int] = (30, 30)
    process_size: Tuple[int, int] = (64, 64)

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 1000
    max_grad_norm: float = 1.0
    dropout: float = 0.1

    # Data paths
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    arc_agi1_dir: str = "ARC-AGI/data/training"
    arc_agi2_dir: str = "ARC-AGI-2/data/training"

    # Training dataset selection (choose which preprocessed dataset to use)
    training_dataset: str = "arc_agi1"  # "arc_agi1" or "arc_agi2"

    # Model paths
    model_dir: str = "models"
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_dir: str = "logs"
    log_interval: int = 2
    save_interval: int = 10

    # Early stopping
    early_stopping_patience: int = 50

    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    # Deterministic training
    random_seed: int = 42
    deterministic: bool = True

    # Color palette (ARC official 10 colors)
    color_palette: List[List[float]] = None

    def __post_init__(self):
        """Initialize color palette after object creation."""
        if self.color_palette is None:
            self.color_palette = [
                [0.0, 0.0, 0.0],  # 0: Black (#000)
                [0.0, 0.455, 0.851],  # 1: Blue (#0074D9)
                [1.0, 0.255, 0.212],  # 2: Red (#FF4136)
                [0.180, 0.800, 0.251],  # 3: Green (#2ECC40)
                [1.0, 0.863, 0.0],  # 4: Yellow (#FFDC00)
                [0.667, 0.667, 0.667],  # 5: Grey (#AAAAAA)
                [0.941, 0.071, 0.745],  # 6: Fuschia (#F012BE)
                [1.0, 0.522, 0.106],  # 7: Orange (#FF851B)
                [0.498, 0.859, 1.0],  # 8: Teal (#7FDBFF)
                [0.529, 0.047, 0.145],  # 9: Brown (#870C25)
            ]

    def set_deterministic_training(self):
        """Set up deterministic training for reproducible results."""
        if self.deterministic:
            # set random seeds
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

            # for cuda
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_seed)
                torch.cuda.manual_seed_all(self.random_seed)

            # set deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # set environment variable for additional determinism
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)
