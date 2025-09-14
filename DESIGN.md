# ARC-AGI2: Simple Basic Model Design

## Overview

This document describes a simple but effective model for solving ARC (Abstraction and Reasoning Corpus) tasks using a ResNet encoder + MLP decoder architecture. The model learns rules from input/output example pairs and applies them to generate solutions. This is designed as a basic, minimal viable product for rapid prototyping and validation.

## Architecture

### Core Components

**1. ResNet Encoder (Rule Learning)**
- **Backbone**: ResNet-18 pretrained on ImageNet
- **Input**: 2 example pairs (input + output images)
- **Output**: Rule latent vector (128 dimensions)
- **Purpose**: Extract abstract rule representation from visual examples

**2. MLP Decoder (Solution Generation)**
- **Architecture**: Simple MLP decoder
- **Input**: Rule latent + target input (concatenated)
- **Output**: Solution image (30x30 ARC standard)
- **Purpose**: Generate solution from learned rule and target input

### Model Flow

```
Input: [Example1_input, Example1_output, Example2_input, Example2_output] (64x64x3 RGB)
       [Target_input] (30x30 grayscale)
↓
ResNet Encoder: Extract rule from 2 example pairs (64x64x3 RGB)
↓
Rule Latent Vector (128 dims)
↓
MLP Decoder: Generate solution from rule + target (30x30 grayscale)
↓
Solution Image (30x30)
```

## File Structure

```
arc-agi2/
├── algo/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── simple_arc.py          # Main model class
│   │   ├── encoder.py             # ResNet encoder
│   │   └── decoder.py             # Simple MLP decoder
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # ARC dataset class
│   │   └── preprocessing.py       # Image preprocessing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop
│   │   ├── losses.py              # Loss functions
│   │   └── metrics.py             # Evaluation metrics
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py              # Configuration management
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py       # Plotting and visualization
│       └── helpers.py             # Utility functions
├── data/
│   ├── raw/                       # Original ARC data
│   ├── processed/                 # Preprocessed data
│   └── data_processing.py         # Data pipeline
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs
├── results/                       # Evaluation results
├── requirements.txt
├── train.py                       # Main training script
├── evaluate.py                    # Evaluation script
├── DESIGN.md                      # This design document
└── README.md
```

## Model Specifications

### ResNet Encoder
```python
import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        # Freeze ResNet weights for simplicity
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.rule_head = nn.Linear(512 * 4, config.rule_dim)  # 2048 input dims (4 images × 512 features)
    
    def forward(self, example1_input, example1_output, example2_input, example2_output):
        # Process each image separately through ResNet (images are already RGB from preprocessing)
        input1_feat = self.backbone(example1_input)
        output1_feat = self.backbone(example1_output)
        input2_feat = self.backbone(example2_input)
        output2_feat = self.backbone(example2_output)
        
        # Combine features from all 4 images
        combined_features = torch.cat([input1_feat, output1_feat, input2_feat, output2_feat], dim=1)
        
        # Project to rule latent space
        rule_latent = self.rule_head(combined_features)
        return rule_latent
```

### MLP Decoder
```python
import torch
import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rule_dim = config.rule_dim
        self.input_size = config.input_size
        
        # Calculate input dimension: rule latent + flattened target input
        input_dim = config.rule_dim + config.input_size[0] * config.input_size[1]
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.input_size[0] * config.input_size[1])
        )
    
    def forward(self, rule_latent, target_input):
        # Flatten target input
        target_flat = target_input.view(target_input.size(0), -1)
        
        # Concatenate rule latent with target input
        combined = torch.cat([rule_latent, target_flat], dim=1)
        
        # Process through MLP
        output = self.mlp(combined)
        
        # Reshape to image dimensions
        output = output.view(-1, 1, self.input_size[0], self.input_size[1])
        return output
```

### Why MLP Decoder

**Simplicity**: MLPs are much simpler than CNNs
- **Easy implementation**: Straightforward forward pass
- **Fast training**: Fewer parameters, faster convergence
- **Easy debugging**: Clear gradient flow, simple architecture

**Basic Model Focus**: Get a working baseline quickly
- **Minimal viable product**: Start simple, iterate later
- **Fast iteration**: Quick training cycles for experimentation
- **Stable training**: Less prone to training instabilities

**Modular Design**: Easy to upgrade later
- **Clean separation**: Easy to swap out decoder
- **Clear upgrade path**: Add spatial reasoning later
- **Proven approach**: MLP works for many vision tasks

## Complete Model Prototype

### Full Model Architecture
```python
import torch
import torch.nn as nn

class SimpleARCModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ResNetEncoder(config)
        self.decoder = MLPDecoder(config)
        self.config = config
    
    def forward(self, example1_input, example1_output, example2_input, example2_output, target_input):
        # Step 1: Extract rule from example pairs
        rule_latent = self.encoder(example1_input, example1_output, example2_input, example2_output)
        
        # Step 2: Generate solution from rule + target
        solution = self.decoder(rule_latent, target_input)
        
        return solution
```

### Data Flow Visualization
```python
# Input shapes (batch_size=32) - after preprocessing
example1_input:  [32, 3, 64, 64]    # RGB input image 1 (preprocessed: 30x30→60x60→64x64, grayscale→RGB)
example1_output: [32, 3, 64, 64]    # RGB output image 1 (preprocessed: 30x30→60x60→64x64, grayscale→RGB)
example2_input:  [32, 3, 64, 64]    # RGB input image 2 (preprocessed: 30x30→60x60→64x64, grayscale→RGB)
example2_output: [32, 3, 64, 64]    # RGB output image 2 (preprocessed: 30x30→60x60→64x64, grayscale→RGB)
target_input:    [32, 1, 30, 30]    # Grayscale target input (30x30 for MLP decoder)

# ResNet feature extraction (frozen)
input1_feat:  [32, 512]    # ResNet(input1_rgb)
output1_feat: [32, 512]    # ResNet(output1_rgb)
input2_feat:  [32, 512]    # ResNet(input2_rgb)
output2_feat: [32, 512]    # ResNet(output2_rgb)

# Concatenate all features
combined_feat: [32, 2048]    # torch.cat([input1_feat, output1_feat, input2_feat, output2_feat])

# Rule latent projection
rule_latent: [32, 128]    # Linear(2048 → 128)

# Decoder processing
# Flatten target input (30x30)
target_flat: [32, 900]    # target_input → flattened (30×30 = 900)

# Concatenate rule latent with target
combined: [32, 1028]    # torch.cat([rule_latent, target_flat]) (128 + 900 = 1028)

# MLP processing
mlp1_out: [32, 512]    # Linear(1028 → 512)
mlp2_out: [32, 256]    # Linear(512 → 256)
mlp3_out: [32, 900]    # Linear(256 → 900)

# Reshape to image dimensions
solution: [32, 1, 30, 30]    # Generated solution image (30x30)
```

### Memory Flow Analysis
```python
# Memory usage per batch (batch_size=32)
# Example images (4 × 64×64×3): 4 × 12,288 = 49,152 pixels per task
# Target images (1 × 30×30×1): 1 × 900 = 900 pixels per task
# Total: 50,052 pixels per task × 32 tasks = 1,601,664 pixels = 6.4 MB

# ResNet processing (frozen):
# 4 images × 512 features = 2,048 features per task
# 32 tasks × 2,048 = 65,536 features = 262 KB

# Rule latent:
# 32 tasks × 128 = 4,096 values = 16 KB

# MLP Decoder:
# Input: 32×1028 = 32,896 values (rule + target)
# Hidden layers: 32×512 + 32×256 = 24,576 values
# Output: 32×900 = 28,800 values
# Memory: ~86,272 × 4 bytes = 345 KB

# Total forward pass: ~7-8 MB per batch
# With gradients: ~14-16 MB per batch
# With optimizer states: ~28-32 MB per batch
```

### Training Step Example
```python
import torch
import torch.nn.functional as F

def training_step(model, batch, criterion, optimizer):
    # Unpack batch
    example1_input = batch['example1_input']    # [32, 3, 64, 64] - RGB for ResNet
    example1_output = batch['example1_output']  # [32, 3, 64, 64] - RGB for ResNet
    example2_input = batch['example2_input']    # [32, 3, 64, 64] - RGB for ResNet
    example2_output = batch['example2_output']  # [32, 3, 64, 64] - RGB for ResNet
    target_input = batch['target_input']        # [32, 1, 30, 30] - Grayscale for MLP
    target_output = batch['target_output']      # [32, 1, 30, 30] - Grayscale for MLP
    
    # Forward pass
    pred = model(example1_input, example1_output, example2_input, example2_output, target_input)
    
    # Calculate loss
    l1_loss = F.l1_loss(pred, target_output)
    l2_loss = F.mse_loss(pred, target_output)
    partial_loss = partial_credit_loss(pred, target_output)
    
    total_loss = 1.0 * l1_loss + 0.5 * l2_loss + 0.3 * partial_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'l1_loss': l1_loss.item(),
        'l2_loss': l2_loss.item(),
        'partial_loss': partial_loss.item()
    }
```

### Inference Example
```python
import torch

def inference(model, task_data):
    # Extract 2 examples from task
    example1_input = task_data['train'][0]['input']
    example1_output = task_data['train'][0]['output']
    example2_input = task_data['train'][1]['input']
    example2_output = task_data['train'][1]['output']
    target_input = task_data['test'][0]['input']
    
    # Preprocess examples for ResNet (30x30 → 60x60 → 64x64 → RGB)
    example1_input = preprocess_example_image(example1_input)
    example1_output = preprocess_example_image(example1_output)
    example2_input = preprocess_example_image(example2_input)
    example2_output = preprocess_example_image(example2_output)
    
    # Preprocess target for MLP decoder (30x30 grayscale)
    target_input = preprocess_target_image(target_input)
    
    # Add batch dimension
    example1_input = example1_input.unsqueeze(0)
    example1_output = example1_output.unsqueeze(0)
    example2_input = example2_input.unsqueeze(0)
    example2_output = example2_output.unsqueeze(0)
    target_input = target_input.unsqueeze(0)
    
    # Generate solution
    with torch.no_grad():
        solution = model(example1_input, example1_output, example2_input, example2_output, target_input)
    
    # Solution is already 30x30 grayscale, no postprocessing needed
    solution = solution.squeeze(0)
    
    return solution
```

## Data Processing

### Input Format
- **Example pairs**: Exactly 2 input/output examples from ARC training data
- **Target input**: Test input image
- **Example images**: 30x30 → 60x60 → 64x64 → RGB (for ResNet processing)
- **Target images**: 30x30 grayscale (for MLP decoder)
- **Normalization**: [-1, 1] range (ImageNet standard for ResNet)

### Preprocessing Pipeline
1. Load ARC task data
2. **Filter tasks**: Only keep tasks with exactly 2+ example pairs
3. **Generate data report**: Create `data_processing_report.txt` with:
   - Total tasks processed
   - Tasks with < 2 examples (filtered out)
   - Tasks with 2+ examples (kept)
   - Image size statistics
   - Color distribution
4. Extract exactly 2 example pairs (input + output)
5. **Preprocess examples**: 30x30 → 60x60 → 64x64 → RGB → normalize (for ResNet)
6. **Preprocess targets**: 30x30 grayscale (for MLP decoder)
7. Convert to PyTorch tensors

### ARC Color Palette Mapping
ARC uses a specific 10-color palette defined in the official testing interface. The mapping from grayscale values (0-9) to RGB colors is:

```python
import torch

def grayscale_to_rgb(grayscale_img):
    # ARC official color palette (from ARC-AGI/apps/css/common.css)
    color_palette = torch.tensor([
        [0.0, 0.0, 0.0],           # 0: Black (#000)
        [0.0, 0.455, 0.851],       # 1: Blue (#0074D9)
        [1.0, 0.255, 0.212],       # 2: Red (#FF4136)
        [0.180, 0.800, 0.251],     # 3: Green (#2ECC40)
        [1.0, 0.863, 0.0],         # 4: Yellow (#FFDC00)
        [0.667, 0.667, 0.667],     # 5: Grey (#AAAAAA)
        [0.941, 0.071, 0.745],     # 6: Fuschia (#F012BE)
        [1.0, 0.522, 0.106],       # 7: Orange (#FF851B)
        [0.498, 0.859, 1.0],       # 8: Teal (#7FDBFF)
        [0.529, 0.047, 0.145],     # 9: Brown (#870C25)
    ])
    
    # Map each pixel value to RGB
    rgb_img = color_palette[grayscale_img.long()]
    return rgb_img
```

**Reference**: Color definitions found in `ARC-AGI/apps/css/common.css` (lines 16-45), which defines the `.symbol_0` through `.symbol_9` CSS classes used by the official ARC testing interface.

### Preprocessing Strategy

**For Example Images (ResNet processing):**
- **30x30 → 60x60**: Exact 2x upscaling using nearest neighbor (preserves discrete ARC colors)
- **60x60 → 64x64**: 2px padding on all sides (ResNet compatibility)
- **Grayscale → RGB**: Using ARC official 10-color palette
- **Normalization**: [-1, 1] range (ImageNet standard)

**For Target Images (MLP decoder):**
- **30x30 grayscale**: Keep original size and format for MLP decoder
- **No preprocessing**: Direct use in decoder concatenation

### Data Processing Notes
- **Consistent data**: Same images every training epoch
- **Focus**: Prove model can learn from training data

## Data Preprocessing Strategy

### One-Time Preprocessing Approach

**Single Phase: Preprocessing + Direct Loading**
- Load and filter ARC data from `ARC-AGI/` and `ARC-AGI-2/` directories
- Process all images once and save as preprocessed tensors
- Generate comprehensive data processing report
- Direct loading during training

### Data Sources

**Primary Dataset**: `ARC-AGI/data/training/` (400 tasks)
- **Purpose**: Main training data
- **Format**: JSON files with nested lists
- **Filtering**: Keep only tasks with exactly 2+ example pairs
- **Expected**: ~350-380 valid tasks after filtering

**Secondary Dataset**: `ARC-AGI-2/data/training/` (400 tasks)
- **Purpose**: Additional training data
- **Format**: Same as ARC-AGI
- **Usage**: Combine with ARC-AGI for larger training set

### Preprocessing Pipeline

```python
import torch
import json
import os
from pathlib import Path

def preprocess_arc_data(data_dir, output_path, min_examples=2):
    """
    One-time preprocessing of ARC data
    
    Args:
        data_dir: Path to ARC data directory (e.g., "ARC-AGI/data/training/")
        output_path: Path to save preprocessed data
        min_examples: Minimum number of examples required per task
    """
    tasks = []
    filtered_count = 0
    total_count = 0
    
    # Load all task files
    for task_file in Path(data_dir).glob("*.json"):
        total_count += 1
        
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        # Filter tasks with sufficient examples
        if len(task_data['train']) < min_examples:
            filtered_count += 1
            continue
        
        # Extract exactly 2 examples
        example1 = task_data['train'][0]
        example2 = task_data['train'][1]
        test_case = task_data['test'][0]
        
        # Preprocess images (examples for ResNet, targets for MLP)
        processed_task = {
            'task_id': task_file.stem,
            'example1_input': preprocess_example_image(example1['input']),
            'example1_output': preprocess_example_image(example1['output']),
            'example2_input': preprocess_example_image(example2['input']),
            'example2_output': preprocess_example_image(example2['output']),
            'target_input': preprocess_target_image(test_case['input']),
            'target_output': preprocess_target_image(test_case['output'])
        }
        
        tasks.append(processed_task)
    
    # Save preprocessed data
    torch.save(tasks, output_path)
    
    # Generate data report
    generate_data_report(tasks, filtered_count, total_count, output_path)
    
    return len(tasks)

def preprocess_example_image(image_data):
    """Convert ARC image data to 30x30, upscale to 60x60, pad to 64x64, convert to RGB (for ResNet)"""
    import torch
    import torch.nn.functional as F
    
    # Convert to tensor and ensure 30x30 size
    img_tensor = torch.tensor(image_data, dtype=torch.float32)
    
    # Pad or crop to 30x30 if needed
    if img_tensor.shape[0] < 30:
        pad_h = 30 - img_tensor.shape[0]
        img_tensor = F.pad(img_tensor, (0, 0, 0, pad_h), value=0)
    elif img_tensor.shape[0] > 30:
        img_tensor = img_tensor[:30, :30]
    
    if img_tensor.shape[1] < 30:
        pad_w = 30 - img_tensor.shape[1]
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, 0), value=0)
    elif img_tensor.shape[1] > 30:
        img_tensor = img_tensor[:, :30]
    
    # Add batch and channel dimensions for processing
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 30, 30]
    
    # Upscale to 60x60 (exact 2x scaling)
    img_60x60 = F.interpolate(img_tensor, size=(60, 60), mode='nearest')
    
    # Pad to 64x64 (2px border)
    img_64x64 = F.pad(img_60x60, (2, 2, 2, 2), mode='constant', value=0)
    
    # Convert grayscale to RGB using ARC color palette
    img_rgb = grayscale_to_rgb(img_64x64.squeeze(0).squeeze(0))  # [64, 64, 3]
    
    # Normalize to [-1, 1] range
    img_rgb = (img_rgb - 0.5) * 2.0
    
    return img_rgb

def preprocess_target_image(image_data):
    """Convert ARC image data to 30x30 tensor (for MLP decoder)"""
    import torch
    import torch.nn.functional as F
    
    # Convert to tensor and ensure 30x30 size
    img_tensor = torch.tensor(image_data, dtype=torch.float32)
    
    # Pad or crop to 30x30 if needed
    if img_tensor.shape[0] < 30:
        pad_h = 30 - img_tensor.shape[0]
        img_tensor = F.pad(img_tensor, (0, 0, 0, pad_h), value=0)
    elif img_tensor.shape[0] > 30:
        img_tensor = img_tensor[:30, :30]
    
    if img_tensor.shape[1] < 30:
        pad_w = 30 - img_tensor.shape[1]
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, 0), value=0)
    elif img_tensor.shape[1] > 30:
        img_tensor = img_tensor[:, :30]
    
    return img_tensor

def grayscale_to_rgb(grayscale_img):
    """Convert grayscale to RGB using ARC official color palette"""
    # ARC official color palette (from ARC-AGI/apps/css/common.css)
    color_palette = torch.tensor([
        [0.0, 0.0, 0.0],           # 0: Black (#000)
        [0.0, 0.455, 0.851],       # 1: Blue (#0074D9)
        [1.0, 0.255, 0.212],       # 2: Red (#FF4136)
        [0.180, 0.800, 0.251],     # 3: Green (#2ECC40)
        [1.0, 0.863, 0.0],         # 4: Yellow (#FFDC00)
        [0.667, 0.667, 0.667],     # 5: Grey (#AAAAAA)
        [0.941, 0.071, 0.745],     # 6: Fuschia (#F012BE)
        [1.0, 0.522, 0.106],       # 7: Orange (#FF851B)
        [0.498, 0.859, 1.0],       # 8: Teal (#7FDBFF)
        [0.529, 0.047, 0.145],     # 9: Brown (#870C25)
    ])
    
    # Map each pixel value to RGB
    rgb_img = color_palette[grayscale_img.long()]
    return rgb_img

def generate_data_report(tasks, filtered_count, total_count, output_path):
    """Generate comprehensive data processing report"""
    report_path = output_path.replace('.pt', '_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("ARC Data Processing Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total tasks processed: {total_count}\n")
        f.write(f"Tasks filtered out (< 2 examples): {filtered_count}\n")
        f.write(f"Tasks kept: {len(tasks)}\n")
        f.write(f"Success rate: {len(tasks)/total_count*100:.1f}%\n\n")
        
        # Image size statistics
        f.write("Image size statistics:\n")
        f.write(f"  Example images: 64x64x3 RGB (preprocessed for ResNet)\n")
        f.write(f"  Target images: 30x30 grayscale (for MLP decoder)\n")
        f.write(f"  Total images processed: {len(tasks) * 5}\n\n")
        
        # Color distribution
        all_pixels = torch.cat([task['example1_input'].flatten() for task in tasks])
        unique_colors = torch.unique(all_pixels)
        f.write(f"Color distribution:\n")
        f.write(f"  Unique values: {unique_colors.tolist()}\n")
        f.write(f"  Min value: {all_pixels.min().item()}\n")
        f.write(f"  Max value: {all_pixels.max().item()}\n")
```

### Runtime Data Loading

```python
import torch
from torch.utils.data import Dataset

class ARCDataset(Dataset):
    def __init__(self, preprocessed_path):
        """
        Dataset for ARC training
        
        Args:
            preprocessed_path: Path to preprocessed .pt file
        """
        self.data = torch.load(preprocessed_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        task = self.data[idx]
        
        return {
            'example1_input': task['example1_input'],
            'example1_output': task['example1_output'],
            'example2_input': task['example2_input'],
            'example2_output': task['example2_output'],
            'target_input': task['target_input'],
            'target_output': task['target_output']
        }
```

### Data Loading Strategy

**Train/Validation Split:**
```python
# Load preprocessed data and split
full_dataset = ARCDataset("data/processed/arc_agi_preprocessed.pt")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
```

### Data Processing Workflow

1. **One-time preprocessing** (run once):
   ```bash
   python data_processing.py --input_dir ARC-AGI/data/training/ --output data/processed/arc_agi_preprocessed.pt
   python data_processing.py --input_dir ARC-AGI-2/data/training/ --output data/processed/arc_agi2_preprocessed.pt
   ```

2. **Runtime training** (every training run):
   ```python
   from torch.utils.data import DataLoader
   
   # Load preprocessed data
   train_dataset = ARCDataset("data/processed/arc_agi_preprocessed.pt")
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   ```

### Benefits of This Approach

**Efficiency:**
- **One-time preprocessing**: Expensive operations done once
- **Fast training**: No preprocessing overhead during training
- **Memory efficient**: Load only what's needed per batch

**Simplicity:**
- **Focus on core model learning**: Direct data loading without complexity
- **Reproducible**: Same data every training run
- **Fast iteration**: Quick training cycles for experimentation

**Debugging:**
- **Data report**: Comprehensive statistics about processed data
- **Reproducible**: Same preprocessing every time
- **Inspectable**: Can examine preprocessed data easily

## Training Strategy

### Loss Function
```python
L_total = α * L1_loss + β * L2_loss + γ * Partial_credit_loss
```
- **L1 Loss**: Exact pixel matching (α=1.0)
- **L2 Loss**: Smooth reconstruction (β=0.5)
- **Partial Credit Loss**: Rewards near-misses with decreasing credit (γ=0.3)

### Loss Function Details

**L1 Loss**: `|pred - target|` - Encourages exact pixel matching
**L2 Loss**: `(pred - target)²` - Provides smooth gradients and penalizes large errors
**Partial Credit Loss**: Custom loss that gives credit for near-misses
```python
import torch

def partial_credit_loss(pred, target, max_distance=2, credit_decay=0.5):
    """
    Partial credit loss with configurable distance-based credit
    
    Args:
        pred: Predicted values
        target: Target values  
        max_distance: Maximum distance for partial credit
        credit_decay: Credit decay factor per distance unit
    """
    diff = torch.abs(pred - target)
    
    # Perfect match: 0 loss
    perfect_mask = (diff == 0).float()
    
    # Partial credit: exponential decay based on distance
    distance_mask = (diff > 0).float()
    credit = torch.exp(-credit_decay * diff) * distance_mask
    
    # Full loss for distances beyond max_distance
    far_miss_mask = (diff > max_distance).float()
    
    loss = 0.0 * perfect_mask + (1.0 - credit) * distance_mask + 1.0 * far_miss_mask
    return loss.mean()
```



### Training Configuration
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR
- **Batch size**: 32 (smaller for faster iteration)
- **Epochs**: 100-150
- **Dropout**: 0.1
- **Gradient clipping**: 1.0 (standard value for stability)

### Training Process
1. Load pretrained ResNet-18 weights (frozen)
2. Load preprocessed ARC data (examples as 64x64x3 RGB, targets as 30x30 grayscale)
3. Train end-to-end on ARC training data with 2 examples per task
4. Apply gradient clipping for training stability
5. Save checkpoints every 10 epochs
6. Monitor training with validation metrics

## Trainer Implementation

### Trainer Class
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from pathlib import Path

class ARCTrainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.epochs
        )
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create output directories
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
    
    def _create_loss_function(self):
        """Create combined loss function"""
        def combined_loss(pred, target):
            l1_loss = nn.functional.l1_loss(pred, target)
            l2_loss = nn.functional.mse_loss(pred, target)
            partial_loss = partial_credit_loss(pred, target)
            
            total_loss = (
                self.config.l1_weight * l1_loss + 
                self.config.l2_weight * l2_loss + 
                self.config.partial_credit_weight * partial_loss
            )
            return total_loss, l1_loss, l2_loss, partial_loss
        
        return combined_loss
    
    def train_epoch(self, train_loader):
        """Train for one epoch with progress bar"""
        from tqdm import tqdm
        
        self.model.train()
        epoch_losses = {'total': 0, 'l1': 0, 'l2': 0, 'partial': 0}
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            pred = self.model(
                batch['example1_input'],
                batch['example1_output'], 
                batch['example2_input'],
                batch['example2_output'],
                batch['target_input']
            )
            
            # Calculate loss
            total_loss, l1_loss, l2_loss, partial_loss = self.criterion(
                pred, batch['target_output']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            # Update running losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['l1'] += l1_loss.item()
            epoch_losses['l2'] += l2_loss.item()
            epoch_losses['partial'] += partial_loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'L1': f'{l1_loss.item():.4f}',
                'L2': f'{l2_loss.item():.4f}',
                'Partial': f'{partial_loss.item():.4f}'
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        return epoch_losses
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch with progress bar"""
        from tqdm import tqdm
        
        self.model.eval()
        epoch_losses = {'total': 0, 'l1': 0, 'l2': 0, 'partial': 0}
        
        # Progress bar for validation
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                pred = self.model(
                    batch['example1_input'],
                    batch['example1_output'],
                    batch['example2_input'], 
                    batch['example2_output'],
                    batch['target_input']
                )
                
                # Calculate loss
                total_loss, l1_loss, l2_loss, partial_loss = self.criterion(
                    pred, batch['target_output']
                )
                
                # Update running losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['l1'] += l1_loss.item()
                epoch_losses['l2'] += l2_loss.item()
                epoch_losses['partial'] += partial_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Val Loss': f'{total_loss.item():.4f}'
                })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(val_loader)
        
        return epoch_losses
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {self.current_epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, train_loader, val_loader, resume_from=None):
        """Main training loop"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses = self.validate_epoch(val_loader)
            self.val_losses.append(val_losses)
            
            # Update scheduler
            self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.epochs} - {epoch_time:.2f}s")
            print(f"Train - Total: {train_losses['total']:.4f}, L1: {train_losses['l1']:.4f}, "
                  f"L2: {train_losses['l2']:.4f}, Partial: {train_losses['partial']:.4f}")
            print(f"Val   - Total: {val_losses['total']:.4f}, L1: {val_losses['l1']:.4f}, "
                  f"L2: {val_losses['l2']:.4f}, Partial: {val_losses['partial']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to file
            self.log_epoch(epoch, train_losses, val_losses, epoch_time)
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                print(f"🎉 New best validation loss: {self.best_val_loss:.4f}")
            
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(is_best)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def log_epoch(self, epoch, train_losses, val_losses, epoch_time):
        """Log epoch results to file"""
        log_entry = f"{epoch},{train_losses['total']:.4f},{val_losses['total']:.4f},{epoch_time:.2f}\n"
        with open(self.log_dir / "training.log", "a") as f:
            f.write(log_entry)
    
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        test_losses = {'total': 0, 'l1': 0, 'l2': 0, 'partial': 0}
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                pred = self.model(
                    batch['example1_input'],
                    batch['example1_output'],
                    batch['example2_input'],
                    batch['example2_output'], 
                    batch['target_input']
                )
                
                # Calculate loss
                total_loss, l1_loss, l2_loss, partial_loss = self.criterion(
                    pred, batch['target_output']
                )
                
                # Store predictions and targets
                all_predictions.append(pred.cpu())
                all_targets.append(batch['target_output'].cpu())
                
                # Update losses
                test_losses['total'] += total_loss.item()
                test_losses['l1'] += l1_loss.item()
                test_losses['l2'] += l2_loss.item()
                test_losses['partial'] += partial_loss.item()
        
        # Average losses
        for key in test_losses:
            test_losses[key] /= len(test_loader)
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return test_losses, metrics
    
    def _calculate_metrics(self, predictions, targets):
        """Calculate evaluation metrics"""
        # Perfect accuracy (exact pixel match)
        perfect_matches = torch.all(predictions == targets, dim=(1, 2, 3))
        perfect_accuracy = perfect_matches.float().mean().item()
        
        # Pixel accuracy
        pixel_matches = (predictions == targets).float()
        pixel_accuracy = pixel_matches.mean().item()
        
        # Near-miss accuracy (within 1 pixel)
        diff = torch.abs(predictions - targets)
        near_misses = torch.all(diff <= 1, dim=(1, 2, 3))
        near_miss_accuracy = near_misses.float().mean().item()
        
        return {
            'perfect_accuracy': perfect_accuracy,
            'pixel_accuracy': pixel_accuracy,
            'near_miss_accuracy': near_miss_accuracy
        }
```

### Training Script
```python
import torch
from torch.utils.data import DataLoader
from algo.models.simple_arc import SimpleARCModel
from algo.data.dataset import ARCDataset
from algo.config.config import Config
from algo.training.trainer import ARCTrainer

def main():
    # Configuration
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load datasets
    train_dataset = ARCDataset("data/processed/arc_agi_preprocessed.pt")
    val_dataset = ARCDataset("data/processed/arc_agi2_preprocessed.pt")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    model = SimpleARCModel(config)
    
    # Create trainer
    trainer = ARCTrainer(model, config, device)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    test_losses, metrics = trainer.evaluate(val_loader)
    
    print("\nFinal Evaluation Results:")
    print(f"Perfect Accuracy: {metrics['perfect_accuracy']:.4f}")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Near-Miss Accuracy: {metrics['near_miss_accuracy']:.4f}")

if __name__ == "__main__":
    main()
```

### Training Features

**Comprehensive Logging:**
- Real-time loss tracking during training
- Per-epoch train/validation metrics
- Learning rate scheduling information
- Training time and progress monitoring

**Checkpoint Management:**
- Automatic checkpoint saving every 10 epochs
- Best model saving based on validation loss
- Resume training from any checkpoint
- Complete state preservation (model, optimizer, scheduler)

**Robust Training:**
- Gradient clipping for stability
- Learning rate scheduling
- Mixed precision support ready

**Evaluation Tools:**
- Perfect accuracy (exact pixel match)
- Pixel accuracy (overall match percentage)
- Near-miss accuracy (within 1 pixel)
- Comprehensive loss breakdown

**Error Handling:**
- Device management (CPU/GPU)
- Data loading error handling
- Memory management
- Graceful checkpoint recovery

### Training Monitoring

**Progress Visualization:**
- **tqdm progress bars** for real-time batch progress
- **Live loss updates** in progress bar postfix
- **Epoch timing** to track training speed
- **Best model notifications** with emoji indicators

**Logging Strategy:**
- **Console output**: Real-time progress with detailed metrics
- **File logging**: CSV format for analysis (`logs/training.log`)
- **Checkpoint logging**: Complete training state preservation

**What's Monitored:**
```python
# Per-batch (in progress bar)
Loss: 0.1234, L1: 0.0987, L2: 0.0123, Partial: 0.0456

# Per-epoch (console + file)
Epoch 15/100 - 45.23s
Train - Total: 0.1234, L1: 0.0987, L2: 0.0123, Partial: 0.0456
Val   - Total: 0.1456, L1: 0.1123, L2: 0.0156, Partial: 0.0567
LR: 0.000123
🎉 New best validation loss: 0.1456
```

**Why This Approach:**
- **Simple**: No complex dependencies beyond tqdm
- **Immediate**: Real-time feedback during training
- **Lightweight**: Minimal overhead on training speed
- **Debuggable**: Easy to add print statements anywhere
- **Portable**: Works in terminal, Jupyter, remote servers

## Memory Requirements

### Model Parameters
```python
# ResNet-18 (frozen): ~11M parameters
# Rule head: 2048 → 128 = 262,144 parameters
# MLP Decoder:
#   - Linear1: 1028 → 512 = 526,336 parameters
#   - Linear2: 512 → 256 = 131,072 parameters  
#   - Linear3: 256 → 900 = 230,400 parameters
# Total trainable: ~1.15M parameters
```

### Memory Calculation (Batch Size 32)
```python
# Preprocessed data:
# - Example images: 4 × 64×64×3 = 49,152 pixels per task
# - Target images: 1 × 30×30×1 = 900 pixels per task
# - Total: 50,052 pixels per task × 32 tasks = 1,601,664 pixels = 6.4 MB

# ResNet features (frozen):
# - 4 images × 512 features = 2,048 features per task
# - 32 tasks = 65,536 features
# - Memory: 65,536 × 4 bytes = 262 KB

# Rule latent:
# - 32 tasks × 128 dims = 4,096 values
# - Memory: 4,096 × 4 bytes = 16 KB

# MLP Decoder:
# - Input: 32×1028 = 32,896 values (rule + target)
# - Hidden layers: 32×512 + 32×256 = 24,576 values
# - Output: 32×900 = 28,800 values
# - Memory: ~86,272 × 4 bytes = 345 KB

# Total estimated memory: ~7-8 MB per batch
# GPU memory needed: ~2-4 GB (including gradients, optimizer states)
```

### VRAM Requirements
- **Minimum**: 4 GB GPU memory
- **Recommended**: 8 GB GPU memory
- **Batch size scaling**: Linear with batch size

## Evaluation Metrics

### Quantitative Metrics
- **Perfect Accuracy**: Percentage of tasks solved exactly (100% pixel match)
- **Near-Miss Accuracy**: Percentage of tasks with 1-pixel errors (partial credit)
- **Pixel Accuracy**: Overall pixel match percentage
- **L1/L2 Loss**: Reconstruction error
- **Rule Consistency**: Latent space clustering by rule type

### Partial Credit Scoring
```python
import torch

def calculate_partial_credit(pred, target, max_distance=2, credit_decay=0.5):
    """
    Calculate partial credit score with configurable distance-based credit
    
    Args:
        pred: Predicted values
        target: Target values  
        max_distance: Maximum distance for partial credit
        credit_decay: Credit decay factor per distance unit
    """
    diff = torch.abs(pred - target)
    
    # Perfect match: 1.0 score
    perfect_mask = (diff == 0).float()
    
    # Partial credit: exponential decay based on distance
    distance_mask = (diff > 0).float()
    credit = torch.exp(-credit_decay * diff) * distance_mask
    
    # No credit for distances beyond max_distance
    far_miss_mask = (diff > max_distance).float()
    
    score = perfect_mask + credit * distance_mask + 0.0 * far_miss_mask
    return score.mean()
```

### Qualitative Assessment
- **Visual Inspection**: Side-by-side comparison of predictions vs targets
- **Failure Analysis**: Identify common failure patterns
- **Rule Understanding**: Analyze what rules the model learns

## Configuration Management

### Model Parameters
```python
from dataclasses import dataclass

@dataclass
class Config:
    # Model parameters
    input_size: tuple = (30, 30)      # ARC standard input/output size
    process_size: tuple = (64, 64)    # Size for ResNet processing
    rule_dim: int = 128
    dropout: float = 0.1
    num_examples: int = 2  # Fixed to 2 example pairs
    freeze_resnet: bool = True  # Freeze ResNet for simplicity
    
    # Training parameters
    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 100
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # Data parameters
    data_path: str = "data/raw"
    normalize: bool = True
    min_examples: int = 2  # Filter tasks with < 2 examples
    
    # Loss weights
    l1_weight: float = 1.0
    l2_weight: float = 0.5
    partial_credit_weight: float = 0.3
```

## Expected Performance

### Target Metrics
- **Perfect Accuracy**: 10%+ on ARC evaluation set
- **Simple Tasks**: 20-30% accuracy (color changes, object moves)
- **Medium Tasks**: 10-15% accuracy (pattern completion)
- **Complex Tasks**: 5-10% accuracy (abstract reasoning)

### Success Criteria
- Achieve 10%+ perfect accuracy on ARC evaluation
- Demonstrate learning of basic visual transformations
- Show improvement over random baseline
- Provide clear path for architectural improvements

## Model Simplifications

This design implements a basic, minimal viable product for rapid prototyping and validation:

### **Simplified Architecture**
- **2 examples only**: Fixed to exactly 2 example pairs (not all available)
- **30x30 input/output**: ARC standard size for compatibility
- **Example preprocessing**: 60x60 upscale + 64x64 padding for ResNet processing
- **128-dim rule latent**: Smaller latent space for simpler learning
- **Simple MLP decoder**: No spatial reasoning, just direct mapping
- **Concatenation**: Simple aggregation instead of attention

### **Faster Training**
- **Batch size 32**: Smaller batches for faster iteration
- **Learning rate 1e-4**: Standard rate for faster convergence
- **100 epochs**: Shorter training time

### **Easy Debugging**
- **Simple data flow**: Clear path from input to output
- **Fewer parameters**: Easier to understand and debug
- **Stable training**: Gradient clipping and conservative settings
- **Quick iteration**: Fast training cycles for rapid experimentation

### **Future Enhancements**
- Add attention mechanisms
- Increase to 64x64 images
- Use all available examples
- Increase rule latent size

## Implementation Philosophy

### Simplicity First
- Start with minimal viable architecture
- Use pretrained components where possible
- Focus on getting working baseline quickly
- Iterate and improve incrementally

### Modularity
- Clean separation between components
- Easy to swap out individual parts
- Clear interfaces between modules
- Extensible for future improvements

### Debugging
- Simple architecture for easy debugging
- Clear data flow and gradient paths
- Comprehensive logging and visualization
- Incremental testing of components

---

## TODO List

### Core Implementation
- [ ] Create file structure and module organization
- [ ] Implement ResNet encoder with pretrained weights (4 images → 2048 features)
- [ ] Implement simple MLP decoder (rule + target → solution)
- [ ] Create main SimpleARCModel class
- [ ] Add configuration management system

### Data Pipeline
- [ ] Implement ARC dataset loading class (exactly 2 examples per task)
- [ ] Create image preprocessing pipeline (30x30 → 60x60 → 64x64, grayscale→RGB, [-1,1] normalization)
- [ ] Implement data validation and error handling
- [ ] Create data processing report generator

### Training Infrastructure
- [ ] Implement training loop with proper logging
- [ ] Create loss functions (L1 + L2 + Partial Credit)
- [ ] Add evaluation metrics and scoring
- [ ] Implement checkpoint saving/loading
- [ ] Create training visualization tools

### Utilities and Tools
- [ ] Add model visualization and debugging tools
- [ ] Create evaluation script for ARC tasks
- [ ] Implement result analysis and reporting
- [ ] Add configuration validation
- [ ] Create setup and installation scripts

### Testing and Validation
- [ ] Test individual components in isolation
- [ ] Validate data loading and preprocessing
- [ ] Test training loop with small dataset
- [ ] Verify model forward pass and gradients
- [ ] Test evaluation metrics and scoring

### Future Enhancements
- [ ] Experiment with CNN decoder for spatial reasoning
- [ ] Add attention mechanisms between examples
- [ ] Try different rule latent dimensions
- [ ] Test various loss function combinations
- [ ] Implement learning rate scheduling
- [ ] Support variable number of examples
- [ ] Implement perceptual loss
- [ ] Add gradient accumulation
- [ ] Create comprehensive evaluation suite
- [ ] Implement contrastive learning for rule similarity
