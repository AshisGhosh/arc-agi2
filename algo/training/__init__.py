from .base_trainer import BaseTrainer
from .resnet_trainer import ResNetTrainer
from .patch_trainer import PatchTrainer
from .transformer_trainer import TransformerTrainer


def create_trainer(model, config, dataset=None) -> BaseTrainer:
    """Factory function to create trainer based on config."""
    model_type = getattr(config, "model_type", "simple_arc")

    if model_type == "simple_arc":
        return ResNetTrainer(model, config, dataset)
    elif model_type == "patch_attention":
        return PatchTrainer(model, config, dataset)
    elif model_type == "transformer_arc":
        return TransformerTrainer(model, config, dataset)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: ['simple_arc', 'patch_attention', 'transformer_arc']"
        )


__all__ = [
    "BaseTrainer",
    "ResNetTrainer",
    "PatchTrainer",
    "TransformerTrainer",
    "create_trainer",
]
