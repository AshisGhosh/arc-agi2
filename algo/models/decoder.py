import torch
import torch.nn as nn
from typing import Tuple

class MLPDecoder(nn.Module):
    """
    MLP decoder for generating solutions from rule latent and target input.
    
    Takes rule latent vector and flattened target input, outputs solution.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rule_dim = config.rule_dim
        self.input_size = config.input_size
        
        # Calculate input dimension: rule_dim + flattened target
        input_dim = config.rule_dim + config.input_size[0] * config.input_size[1]
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.input_size[0] * config.input_size[1])
        )
        
    def forward(self, rule_latent: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP decoder.
        
        Args:
            rule_latent: Rule latent vector [B, 128]
            target_input: Target input image [B, 1, 30, 30]
            
        Returns:
            Solution image [B, 1, 30, 30]
        """
        # Flatten target input
        target_flat = target_input.view(target_input.size(0), -1)  # [B, 900]
        
        # Concatenate rule latent with flattened target
        combined = torch.cat([rule_latent, target_flat], dim=1)  # [B, 1028]
        
        # Pass through MLP
        output = self.mlp(combined)  # [B, 900]
        
        # Reshape to image format
        output = output.view(-1, 1, self.input_size[0], self.input_size[1])  # [B, 1, 30, 30]
        
        return output
