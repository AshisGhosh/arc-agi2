import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from ..config import Config


def grayscale_to_rgb(
    grayscale_img: torch.Tensor, color_palette: List[List[float]]
) -> torch.Tensor:
    """
    Convert grayscale image to RGB using ARC color palette.

    Args:
        grayscale_img: Grayscale image [H, W] with values 0-9
        color_palette: Color palette [10, 3] with RGB values

    Returns:
        RGB image [H, W, 3] with values in [0, 1]
    """
    color_palette = torch.tensor(color_palette, dtype=torch.float32)
    rgb_img = color_palette[grayscale_img.long()]
    return rgb_img


def preprocess_rgb_image(image_data: np.ndarray, config: Config) -> torch.Tensor:
    """
    Preprocess example image for ResNet encoder.

    Args:
        image_data: Raw ARC image data [H, W] with values 0-9
        config: Configuration object

    Returns:
        Preprocessed RGB tensor [3, 64, 64] normalized to [-1, 1]
    """
    # Convert to tensor and ensure float32
    image_data = image_data.copy()
    img_tensor = torch.tensor(image_data, dtype=torch.float32)

    # Pad or crop to 30x30
    if img_tensor.shape[0] < 30:
        img_tensor = F.pad(img_tensor, (0, 0, 0, 30 - img_tensor.shape[0]), value=0)
    elif img_tensor.shape[0] > 30:
        img_tensor = img_tensor[:30, :30]

    if img_tensor.shape[1] < 30:
        img_tensor = F.pad(img_tensor, (0, 30 - img_tensor.shape[1], 0, 0), value=0)
    elif img_tensor.shape[1] > 30:
        img_tensor = img_tensor[:, :30]

    # Add batch and channel dimensions for interpolation
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 30, 30]

    # Upscale to 60x60 (exact 2x scaling)
    img_60x60 = F.interpolate(
        img_tensor, size=(60, 60), mode="nearest"
    )  # [1, 1, 60, 60]

    # Pad to 64x64 (2px padding on each side)
    img_64x64 = F.pad(
        img_60x60, (2, 2, 2, 2), mode="constant", value=0
    )  # [1, 1, 64, 64]

    # Remove batch dimension and convert to RGB
    img_64x64 = img_64x64.squeeze(0).squeeze(0)  # [64, 64]
    img_rgb = grayscale_to_rgb(img_64x64, config.color_palette)  # [64, 64, 3]

    # Convert to [C, H, W] format
    img_rgb = img_rgb.permute(2, 0, 1)  # [3, 64, 64]

    # Normalize to [-1, 1] (ImageNet standard)
    img_rgb = (img_rgb - 0.5) * 2.0

    return img_rgb


def preprocess_grid_image(image_data: np.ndarray, config: Config) -> torch.Tensor:
    """
    Preprocess target image for MLP decoder.

    Args:
        image_data: Raw ARC image data [H, W] with values 0-9
        config: Configuration object

    Returns:
        Preprocessed grayscale tensor [1, 30, 30]
    """
    # Convert to tensor and ensure float32
    image_data = image_data.copy()
    img_tensor = torch.tensor(image_data, dtype=torch.float32)

    # Pad or crop to 30x30
    if img_tensor.shape[0] < 30:
        img_tensor = F.pad(img_tensor, (0, 0, 0, 30 - img_tensor.shape[0]), value=0)
    elif img_tensor.shape[0] > 30:
        img_tensor = img_tensor[:30, :30]

    if img_tensor.shape[1] < 30:
        img_tensor = F.pad(img_tensor, (0, 30 - img_tensor.shape[1], 0, 0), value=0)
    elif img_tensor.shape[1] > 30:
        img_tensor = img_tensor[:, :30]

    # Add channel dimension
    img_tensor = img_tensor.unsqueeze(0)  # [1, 30, 30]

    return img_tensor
