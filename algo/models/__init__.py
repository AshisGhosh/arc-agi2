from .base import BaseARCModel
from .encoder import ResNetEncoder
from .decoder import MLPDecoder
from .simple_arc import SimpleARCModel


def create_model(config) -> BaseARCModel:
    """Factory function to create model based on config."""
    decoder_type = getattr(config, "decoder_type", "mlp")

    if decoder_type == "mlp":
        return SimpleARCModel(config)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}. Available: ['mlp']")


__all__ = [
    "BaseARCModel",
    "ResNetEncoder",
    "MLPDecoder",
    "SimpleARCModel",
    "create_model",
]
