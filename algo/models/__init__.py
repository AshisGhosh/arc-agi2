from .base import BaseARCModel
from .encoder import ResNetEncoder
from .decoder import MLPDecoder
from .simple_arc import SimpleARCModel
from .patch_attention import PatchCrossAttentionModel


def create_model(config) -> BaseARCModel:
    """Factory function to create model based on config."""
    model_type = getattr(config, "model_type", "simple_arc")

    if model_type == "simple_arc":
        return SimpleARCModel(config)
    elif model_type == "patch_attention":
        return PatchCrossAttentionModel(config)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: ['simple_arc', 'patch_attention']"
        )


__all__ = [
    "BaseARCModel",
    "ResNetEncoder",
    "MLPDecoder",
    "SimpleARCModel",
    "PatchCrossAttentionModel",
    "create_model",
]
