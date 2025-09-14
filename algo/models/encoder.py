import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetEncoder(nn.Module):
    """
    ResNet-18 encoder for extracting rule latents from example pairs.

    Processes 4 images (2 input/output pairs) through frozen ResNet-18,
    concatenates features, and projects to rule latent space.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Load pretrained ResNet-18 and freeze weights
        self.backbone = resnet18(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Rule projection head
        # Input: 4 images × 512 features = 2048
        # Output: rule_dim (128)
        self.rule_head = nn.Linear(512 * 4, config.rule_dim)

    def forward(
        self,
        example1_input: torch.Tensor,
        example1_output: torch.Tensor,
        example2_input: torch.Tensor,
        example2_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through ResNet encoder.

        Args:
            example1_input: First input image [B, 3, 64, 64]
            example1_output: First output image [B, 3, 64, 64]
            example2_input: Second input image [B, 3, 64, 64]
            example2_output: Second output image [B, 3, 64, 64]

        Returns:
            Rule latent vector [B, 128]
        """
        # Extract features from each image
        input1_feat = self.backbone(example1_input).view(
            example1_input.size(0), -1
        )  # [B, 512]
        output1_feat = self.backbone(example1_output).view(
            example1_output.size(0), -1
        )  # [B, 512]
        input2_feat = self.backbone(example2_input).view(
            example2_input.size(0), -1
        )  # [B, 512]
        output2_feat = self.backbone(example2_output).view(
            example2_output.size(0), -1
        )  # [B, 512]

        # Concatenate all features
        combined_features = torch.cat(
            [input1_feat, output1_feat, input2_feat, output2_feat], dim=1
        )  # [B, 2048]

        # Project to rule latent space
        rule_latent = self.rule_head(combined_features)  # [B, 128]

        return rule_latent
