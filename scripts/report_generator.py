#!/usr/bin/env python3
"""
HTML report generator for ARC-AGI data preprocessing.

Generates interactive HTML reports with visualizations and statistics.
"""

import json
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

class HTMLReportBuilder:
    """Builder class for generating HTML reports with visualizations."""
    
    def __init__(self, title: str = "ARC-AGI Data Processing Report"):
        self.title = title
        self.sections = []
        self.charts = []
        self.metadata = {}
        
    def set_metadata(self, dataset_name: str, generated_at: str = None):
        """Set report metadata."""
        self.metadata = {
            'dataset_name': dataset_name,
            'generated_at': generated_at or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def add_overview_section(self, stats: Dict[str, Any]):
        """Add overview statistics section."""
        success_rate = (stats['successfully_processed'] / stats['total_raw_tasks']) * 100
        
        overview_html = f"""
        <div class="section">
            <h2>📊 Processing Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{stats['total_raw_tasks']:,}</div>
                    <div class="stat-label">Total Tasks Loaded</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['tasks_filtered_out']:,}</div>
                    <div class="stat-label">Tasks Filtered Out</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{stats['successfully_processed']:,}</div>
                    <div class="stat-label">Successfully Processed</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-number">{success_rate:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
            </div>
        </div>
        """
        self.sections.append(overview_html)
    
    def add_processing_pipeline_section(self):
        """Add processing pipeline visualization."""
        pipeline_html = """
        <div class="section">
            <h2>🔄 Processing Pipeline</h2>
            <div class="pipeline">
                <div class="pipeline-step">
                    <div class="step-number">1</div>
                    <div class="step-content">
                        <h3>Load Raw Data</h3>
                        <p>Load ARC JSON files from dataset directory</p>
                    </div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-step">
                    <div class="step-number">2</div>
                    <div class="step-content">
                        <h3>Filter Tasks</h3>
                        <p>Keep only tasks with ≥2 training examples</p>
                    </div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-step">
                    <div class="step-number">3</div>
                    <div class="step-content">
                        <h3>Preprocess Images</h3>
                        <p>30x30 → 64x64, grayscale → RGB, normalize</p>
                    </div>
                </div>
                <div class="pipeline-arrow">→</div>
                <div class="pipeline-step">
                    <div class="step-number">4</div>
                    <div class="step-content">
                        <h3>Save Tensors</h3>
                        <p>Save preprocessed data as PyTorch tensors</p>
                    </div>
                </div>
            </div>
        </div>
        """
        self.sections.append(pipeline_html)
    
    def add_data_specs_section(self, data: List[Dict[str, torch.Tensor]]):
        """Add data specifications section."""
        if not data:
            return
            
        # Get shapes from first sample
        sample = data[0]
        shapes = {key: list(tensor.shape) for key, tensor in sample.items()}
        
        # Calculate memory usage
        memory_per_task = 4 * 3 * 64 * 64 * 4 + 2 * 1 * 30 * 30 * 4  # bytes
        total_memory = len(data) * memory_per_task / (1024 * 1024)  # MB
        
        specs_html = f"""
        <div class="section">
            <h2>📋 Data Specifications</h2>
            <div class="specs-grid">
                <div class="spec-card">
                    <h3>Image Dimensions</h3>
                    <ul>
                        <li><strong>Example images:</strong> {shapes['example1_input']} (RGB, 64×64)</li>
                        <li><strong>Target images:</strong> {shapes['target_input']} (Grayscale, 30×30)</li>
                    </ul>
                </div>
                <div class="spec-card">
                    <h3>Data Types</h3>
                    <ul>
                        <li><strong>Example images:</strong> {sample['example1_input'].dtype}</li>
                        <li><strong>Target images:</strong> {sample['target_input'].dtype}</li>
                    </ul>
                </div>
                <div class="spec-card">
                    <h3>Memory Usage</h3>
                    <ul>
                        <li><strong>Per task:</strong> {memory_per_task / 1024 / 1024:.1f} MB</li>
                        <li><strong>Total dataset:</strong> {total_memory:.1f} MB</li>
                    </ul>
                </div>
            </div>
        </div>
        """
        self.sections.append(specs_html)
    
    def add_sample_images_section(self, data: List[Dict[str, torch.Tensor]], num_samples: int = 4):
        """Add sample images visualization."""
        if not data:
            return
            
        # Create sample images
        sample_html = """
        <div class="section">
            <h2>🖼️ Sample Images</h2>
            <div class="image-gallery">
        """
        
        for i in range(min(num_samples, len(data))):
            sample = data[i]
            
            # Convert tensors to images
            example1_input = self._tensor_to_image(sample['example1_input'])
            example1_output = self._tensor_to_image(sample['example1_output'])
            target_input = self._tensor_to_image(sample['target_input'], is_target=True)
            target_output = self._tensor_to_image(sample['target_output'], is_target=True)
            
            sample_html += f"""
                <div class="sample-group">
                    <h3>Sample {i+1}</h3>
                    <div class="image-row">
                        <div class="image-pair">
                            <h4>Example 1</h4>
                            <div class="image-container">
                                <img src="data:image/png;base64,{example1_input}" alt="Input">
                                <span class="arrow">→</span>
                                <img src="data:image/png;base64,{example1_output}" alt="Output">
                            </div>
                        </div>
                        <div class="image-pair">
                            <h4>Target</h4>
                            <div class="image-container">
                                <img src="data:image/png;base64,{target_input}" alt="Target Input">
                                <span class="arrow">→</span>
                                <img src="data:image/png;base64,{target_output}" alt="Target Output">
                            </div>
                        </div>
                    </div>
                </div>
            """
        
        sample_html += """
            </div>
        </div>
        """
        self.sections.append(sample_html)
    
    def add_color_distribution_section(self, data: List[Dict[str, torch.Tensor]]):
        """Add color distribution analysis."""
        if not data:
            return
            
        # Analyze color distributions
        sample = data[0]
        
        # Example image colors (RGB)
        example_colors = {}
        for key in ['example1_input', 'example1_output', 'example2_input', 'example2_output']:
            if key in sample:
                img = (sample[key] + 1) / 2  # Convert from [-1,1] to [0,1]
                unique_vals, counts = torch.unique(img.flatten(), return_counts=True)
                example_colors[key] = {
                    'values': unique_vals.tolist(),
                    'counts': counts.tolist()
                }
        
        # Target image colors (grayscale 0-9)
        target_img = sample['target_input']
        unique_vals, counts = torch.unique(target_img.flatten(), return_counts=True)
        target_colors = {
            'values': unique_vals.tolist(),
            'counts': counts.tolist()
        }
        
        colors_html = f"""
        <div class="section">
            <h2>🎨 Color Distribution</h2>
            <div class="color-analysis">
                <div class="color-section">
                    <h3>Example Images (RGB)</h3>
                    <div class="color-bars">
        """
        
        for key, colors in example_colors.items():
            colors_html += f"""
                        <div class="color-bar">
                            <h4>{key.replace('_', ' ').title()}</h4>
                            <div class="color-values">
            """
            for val, count in zip(colors['values'], colors['counts']):
                colors_html += f'<span class="color-value">Value {val:.3f}: {count} pixels</span>'
            colors_html += "</div></div>"
        
        colors_html += f"""
                    </div>
                </div>
                <div class="color-section">
                    <h3>Target Images (Grayscale 0-9)</h3>
                    <div class="color-bars">
                        <div class="color-bar">
                            <div class="color-values">
        """
        
        for val, count in zip(target_colors['values'], target_colors['counts']):
            val_int = val.item() if hasattr(val, 'item') else val
            colors_html += f'<span class="color-value">Color {val_int}: {count} pixels</span>'
        
        colors_html += """
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        self.sections.append(colors_html)
    
    def _tensor_to_image(self, tensor: torch.Tensor, is_target: bool = False) -> str:
        """Convert tensor to base64 encoded image."""
        if is_target:
            # Target image: grayscale 0-9
            img_array = tensor.numpy()
            # Remove extra dimension if present
            if len(img_array.shape) == 3 and img_array.shape[0] == 1:
                img_array = img_array.squeeze(0)
            # Use ARC color palette
            colors = np.array([
                [0, 0, 0],      # 0: black
                [255, 255, 255], # 1: white
                [255, 0, 0],     # 2: red
                [0, 255, 0],     # 3: green
                [0, 0, 255],     # 4: blue
                [255, 255, 0],   # 5: yellow
                [255, 0, 255],   # 6: magenta
                [0, 255, 255],   # 7: cyan
                [128, 128, 128], # 8: gray
                [255, 128, 0]    # 9: orange
            ])
            rgb_array = colors[img_array.astype(int)]
        else:
            # Example image: RGB normalized [-1, 1]
            img_array = ((tensor + 1) / 2 * 255).numpy().astype(np.uint8)
            rgb_array = img_array.transpose(1, 2, 0)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(rgb_array)
        ax.axis('off')
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64
    
    def build(self) -> str:
        """Build the complete HTML report."""
        sections_html = '\n'.join(self.sections)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.title}</h1>
            <div class="metadata">
                <p><strong>Dataset:</strong> {self.metadata.get('dataset_name', 'Unknown')}</p>
                <p><strong>Generated:</strong> {self.metadata.get('generated_at', 'Unknown')}</p>
            </div>
        </header>
        
        <main>
            {sections_html}
        </main>
        
        <footer>
            <p>Generated by ARC-AGI preprocessing pipeline</p>
        </footer>
    </div>
</body>
</html>
        """
        return html
    
    def _get_css(self) -> str:
        """Get CSS styles for the report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .metadata {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .section {
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        
        .stat-card.success {
            border-left-color: #28a745;
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .pipeline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .pipeline-step {
            display: flex;
            align-items: center;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            flex: 1;
            min-width: 200px;
        }
        
        .step-number {
            background: #667eea;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
        }
        
        .pipeline-arrow {
            font-size: 1.5em;
            color: #667eea;
            font-weight: bold;
        }
        
        .specs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .spec-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }
        
        .spec-card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        .spec-card ul {
            list-style: none;
        }
        
        .spec-card li {
            padding: 5px 0;
            border-bottom: 1px solid #e9ecef;
        }
        
        .image-gallery {
            display: grid;
            gap: 30px;
        }
        
        .sample-group h3 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        .image-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        .image-pair h4 {
            color: #495057;
            margin-bottom: 10px;
        }
        
        .image-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .image-container img {
            border: 2px solid #e9ecef;
            border-radius: 4px;
            max-width: 100px;
            height: auto;
        }
        
        .arrow {
            font-size: 1.5em;
            color: #667eea;
            font-weight: bold;
        }
        
        .color-analysis {
            display: grid;
            gap: 30px;
        }
        
        .color-section h3 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        .color-bars {
            display: grid;
            gap: 15px;
        }
        
        .color-bar {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }
        
        .color-bar h4 {
            color: #495057;
            margin-bottom: 10px;
        }
        
        .color-values {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .color-value {
            background: #e9ecef;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        footer {
            text-align: center;
            padding: 20px;
            color: #6c757d;
            margin-top: 30px;
        }
        
        @media (max-width: 768px) {
            .pipeline {
                flex-direction: column;
            }
            
            .pipeline-arrow {
                transform: rotate(90deg);
            }
            
            .image-row {
                grid-template-columns: 1fr;
            }
        }
        """

def generate_html_report(data: List[Dict[str, torch.Tensor]], 
                        total_raw_tasks: int, 
                        filtered_tasks: int, 
                        dataset_name: str) -> str:
    """
    Generate a complete HTML report for preprocessed data.
    
    Args:
        data: List of preprocessed tasks
        total_raw_tasks: Total number of raw tasks loaded
        filtered_tasks: Number of tasks after filtering
        dataset_name: Name of the dataset
        
    Returns:
        HTML report as string
    """
    builder = HTMLReportBuilder()
    
    # Set metadata
    builder.set_metadata(dataset_name)
    
    # Calculate statistics
    stats = {
        'total_raw_tasks': total_raw_tasks,
        'tasks_filtered_out': total_raw_tasks - filtered_tasks,
        'successfully_processed': len(data)
    }
    
    # Add sections
    builder.add_overview_section(stats)
    builder.add_processing_pipeline_section()
    builder.add_data_specs_section(data)
    builder.add_sample_images_section(data)
    builder.add_color_distribution_section(data)
    
    return builder.build()
