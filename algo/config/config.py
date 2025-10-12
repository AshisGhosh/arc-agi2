from dataclasses import dataclass, asdict
from typing import Tuple, List
import os
import torch
import random
import numpy as np


@dataclass
class Config:
    """Configuration for ARC model training."""

    # =============================================================================
    # MODEL SELECTION
    # =============================================================================
    model_type: str = (
        "transformer_arc"  # "simple_arc", "patch_attention", or "transformer_arc"
    )

    # =============================================================================
    # COMMON MODEL PARAMETERS
    # =============================================================================
    input_size: Tuple[int, int] = (30, 30)
    process_size: Tuple[int, int] = (64, 64)
    dropout: float = 0.1

    # =============================================================================
    # SIMPLE ARC (ResNet + MLP) MODEL PARAMETERS
    # =============================================================================
    # ResNet encoder parameters
    rule_dim: int = 32  # Dimension of rule latent space

    # Rule latent regularization
    rule_latent_regularization_weight: float = 0.1

    # =============================================================================
    # PATCH ATTENTION MODEL PARAMETERS
    # =============================================================================
    # Patch processing
    patch_size: int = 3  # Size of patches (3x3)
    model_dim: int = 128  # Model dimension (d_model)
    num_heads: int = 4  # Number of attention heads
    num_layers: int = 3  # Number of transformer layers

    # Patch model specific training options
    use_support_as_test: bool = True  # Use support examples as additional test inputs

    # =============================================================================
    # TRANSFORMER ARC MODEL PARAMETERS
    # =============================================================================
    # Model architecture
    d_model: int = 256  # Model dimension
    num_rule_tokens: int = 4  # Number of rule tokens from PMA
    num_encoder_layers: int = 8  # Number of transformer encoder layers
    num_decoder_layers: int = 8  # Number of self and cross-attention transformer decoder layers, 3x per layer: self-attn, cross-attn, ff
    num_heads: int = 8  # Number of attention heads for transformer
    patch_size: int = 3  # Patch size for transformer input processing
    num_cls_tokens: int = 4  # Number of CLS tokens from pairwise encoder

    # Rule bottleneck (optional compression)
    use_rule_bottleneck: bool = True  # Enable rule token bottleneck compression
    rule_bottleneck_dim: int = 8  # Compressed dimension for rule tokens

    # Auxiliary loss weights
    support_reconstruction_weight: float = 0.5  # Weight for support reconstruction loss
    cls_regularization_weight: float = 0.0  # Weight for CLS regularization loss
    rule_token_consistency_weight: float = (
        0.01  # Weight for rule token consistency loss
    )
    contrastive_temperature: float = 0.07  # Temperature for contrastive learning
    cls_l2_weight: float = 0.01  # Weight for L2 regularization in CLS loss

    # =============================================================================
    # TRAINING PARAMETERS
    # =============================================================================
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 2000
    max_grad_norm: float = 1.0

    # =============================================================================
    # DATA CONFIGURATION
    # =============================================================================
    # Data paths
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    arc_agi1_dir: str = "ARC-AGI/data/training"
    arc_agi2_dir: str = "ARC-AGI-2/data/training"

    # Training dataset selection
    training_dataset: str = "arc_agi1"  # "arc_agi1" or "arc_agi2"

    # =============================================================================
    # AUGMENTATION PARAMETERS
    # =============================================================================
    # Color augmentation
    use_color_relabeling: bool = False
    augmentation_variants: int = 1  # Number of augmented versions per original example
    preserve_background: bool = True  # Keep background color (0) unchanged

    # Counterfactual augmentation
    enable_counterfactuals: bool = True
    counterfactual_Y: bool = True  # Apply transformation to output (Y)
    counterfactual_X: bool = True  # Apply transformation to input (X)
    counterfactual_transform: str = (
        "rotate_90"  # "rotate_90", "rotate_180", "rotate_270", "reflect_h", "reflect_v"
    )

    # Cycling combinations
    use_cycling: bool = True  # Enable cycling combinations (A,B)->T, (A,T)->B, (T,B)->A

    # =============================================================================
    # TRAINING INFRASTRUCTURE
    # =============================================================================
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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

    def get_model_specific_params(self) -> dict:
        """Get model-specific parameters based on model_type."""
        if self.model_type == "simple_arc":
            return {
                "rule_dim": self.rule_dim,
                "rule_latent_regularization_weight": self.rule_latent_regularization_weight,
            }
        elif self.model_type == "patch_attention":
            return {
                "patch_size": self.patch_size,
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "use_support_as_test": self.use_support_as_test,
            }
        elif self.model_type == "transformer_arc":
            return {
                "patch_size": self.patch_size,  # transformer_arc specific patch_size
                "d_model": self.d_model,
                "num_rule_tokens": self.num_rule_tokens,
                "num_encoder_layers": self.num_encoder_layers,
                "num_decoder_layers": self.num_decoder_layers,
                "num_heads": self.num_heads,  # transformer_arc specific num_heads
                "num_cls_tokens": self.num_cls_tokens,  # transformer_arc specific num_cls_tokens
                "use_rule_bottleneck": self.use_rule_bottleneck,
                "rule_bottleneck_dim": self.rule_bottleneck_dim,
                "support_reconstruction_weight": self.support_reconstruction_weight,
                "cls_regularization_weight": self.cls_regularization_weight,
                "rule_token_consistency_weight": self.rule_token_consistency_weight,
                "contrastive_temperature": self.contrastive_temperature,
            }
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def validate_config(self):
        """Validate configuration parameters."""
        valid_model_types = ["simple_arc", "patch_attention", "transformer_arc"]
        if self.model_type not in valid_model_types:
            raise ValueError(
                f"model_type must be one of {valid_model_types}, got {self.model_type}"
            )

        # Validate transformer-specific parameters
        if self.model_type == "transformer_arc":
            if self.d_model <= 0:
                raise ValueError("d_model must be positive")
            if self.num_rule_tokens <= 0:
                raise ValueError("num_rule_tokens must be positive")
            if self.num_encoder_layers <= 0:
                raise ValueError("num_encoder_layers must be positive")
            if self.num_decoder_layers <= 0:
                raise ValueError("num_decoder_layers must be positive")
            if self.num_heads <= 0:
                raise ValueError("num_heads must be positive")
            if self.patch_size <= 0:
                raise ValueError("patch_size must be positive")
            if self.num_cls_tokens <= 0:
                raise ValueError("num_cls_tokens must be positive")
            if self.rule_bottleneck_dim <= 0:
                raise ValueError("rule_bottleneck_dim must be positive")
            if self.support_reconstruction_weight < 0:
                raise ValueError("support_reconstruction_weight must be non-negative")
            if self.cls_regularization_weight < 0:
                raise ValueError("cls_regularization_weight must be non-negative")
            if self.rule_token_consistency_weight < 0:
                raise ValueError("rule_token_consistency_weight must be non-negative")
            if self.contrastive_temperature <= 0:
                raise ValueError("contrastive_temperature must be positive")

    def to_dict(self) -> dict:
        """convert config to dictionary for json serialization."""
        config_dict = asdict(self)

        # convert tuples to lists for json serialization
        config_dict["input_size"] = list(config_dict["input_size"])
        config_dict["process_size"] = list(config_dict["process_size"])

        # convert device to string
        config_dict["device"] = str(config_dict["device"])

        return config_dict

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
