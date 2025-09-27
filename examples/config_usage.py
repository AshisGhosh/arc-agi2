#!/usr/bin/env python3
"""
Example showing how to use the reorganized Config class for different model types.
"""

from algo.config import Config
from algo.models import create_model
from algo.training import create_trainer


def main():
    """Demonstrate config usage for different model types."""
    
    print("=== Config Usage Examples ===\n")
    
    # Example 1: Default transformer config
    print("1. Default Transformer Config:")
    config = Config()  # Defaults to transformer_arc
    print(f"   Model type: {config.model_type}")
    print(f"   Model-specific params: {config.get_model_specific_params()}")
    
    # Create and test model
    model = create_model(config)
    print(f"   Model created: {type(model).__name__}")
    print(f"   Parameters: {model.get_model_info()['total_parameters']:,}")
    print()
    
    # Example 2: Custom transformer config
    print("2. Custom Transformer Config:")
    config = Config()
    config.model_type = "transformer_arc"
    config.d_model = 256  # Larger model
    config.num_rule_tokens = 16  # More rule tokens
    config.num_encoder_layers = 4  # Deeper encoder
    config.support_reconstruction_weight = 0.2  # Higher auxiliary loss weight
    
    print(f"   d_model: {config.d_model}")
    print(f"   num_rule_tokens: {config.num_rule_tokens}")
    print(f"   num_encoder_layers: {config.num_encoder_layers}")
    print(f"   support_reconstruction_weight: {config.support_reconstruction_weight}")
    
    # Validate config
    try:
        config.validate_config()
        print("   ✅ Config validation passed")
    except Exception as e:
        print(f"   ❌ Config validation failed: {e}")
    print()
    
    # Example 3: Patch attention config
    print("3. Patch Attention Config:")
    config = Config()
    config.model_type = "patch_attention"
    config.model_dim = 256
    config.num_heads = 8
    config.num_layers = 4
    
    print(f"   Model-specific params: {config.get_model_specific_params()}")
    
    model = create_model(config)
    print(f"   Model created: {type(model).__name__}")
    print(f"   Parameters: {model.get_model_info()['total_parameters']:,}")
    print()
    
    # Example 4: Simple ARC config
    print("4. Simple ARC Config:")
    config = Config()
    config.model_type = "simple_arc"
    config.rule_dim = 64  # Larger rule latent space
    
    print(f"   Model-specific params: {config.get_model_specific_params()}")
    
    model = create_model(config)
    print(f"   Model created: {type(model).__name__}")
    print(f"   Parameters: {model.get_model_info()['total_parameters']:,}")
    print()
    
    # Example 5: Trainer creation with config
    print("5. Trainer Creation:")
    config = Config()
    config.model_type = "transformer_arc"
    
    model = create_model(config)
    trainer = create_trainer(model, config)
    
    print(f"   Trainer created: {type(trainer).__name__}")
    print(f"   Support reconstruction weight: {trainer.support_reconstruction_weight}")
    print(f"   CLS regularization weight: {trainer.cls_regularization_weight}")
    print(f"   Contrastive temperature: {trainer.contrastive_temperature}")


if __name__ == "__main__":
    main()
