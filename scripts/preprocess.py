#!/usr/bin/env python3
"""
Data preprocessing script for ARC-AGI dataset.

Preprocesses raw ARC JSON files into tensors for efficient training.
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

from algo.config import Config
from algo.data.preprocessing import preprocess_example_image, preprocess_target_image
from scripts.report_generator import generate_html_report

def load_arc_tasks(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load ARC tasks from JSON files.
    
    Args:
        data_dir: Directory containing ARC JSON files
        
    Returns:
        List of task dictionaries
    """
    tasks = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load all JSON files
    for json_file in data_path.glob("*.json"):
        with open(json_file, 'r') as f:
            task_data = json.load(f)
            tasks.append(task_data)
    
    print(f"Loaded {len(tasks)} tasks from {data_dir}")
    return tasks

def filter_tasks_with_examples(tasks: List[Dict[str, Any]], min_examples: int = 2) -> List[Dict[str, Any]]:
    """
    Filter tasks that have at least min_examples training examples.
    
    Args:
        tasks: List of task dictionaries
        min_examples: Minimum number of examples required
        
    Returns:
        Filtered list of tasks
    """
    filtered_tasks = []
    
    for task in tasks:
        if len(task.get('train', [])) >= min_examples:
            filtered_tasks.append(task)
    
    print(f"Filtered to {len(filtered_tasks)} tasks with >= {min_examples} examples")
    return filtered_tasks

def preprocess_task(task: Dict[str, Any], config: Config) -> Dict[str, torch.Tensor]:
    """
    Preprocess a single ARC task.
    
    Args:
        task: Task dictionary
        config: Configuration object
        
    Returns:
        Dictionary of preprocessed tensors
    """
    # Get first two examples
    example1 = task['train'][0]
    example2 = task['train'][1]
    
    # Get first test case
    test_case = task['test'][0]
    
    # Preprocess example images (for ResNet)
    example1_input = preprocess_example_image(example1['input'], config)
    example1_output = preprocess_example_image(example1['output'], config)
    example2_input = preprocess_example_image(example2['input'], config)
    example2_output = preprocess_example_image(example2['output'], config)
    
    # Preprocess target images (for MLP decoder)
    target_input = preprocess_target_image(test_case['input'], config)
    target_output = preprocess_target_image(test_case['output'], config)
    
    return {
        'example1_input': example1_input,
        'example1_output': example1_output,
        'example2_input': example2_input,
        'example2_output': example2_output,
        'target_input': target_input,
        'target_output': target_output
    }

def preprocess_dataset(tasks: List[Dict[str, Any]], dataset_name: str, config: Config) -> int:
    """
    Preprocess a single dataset and save to disk.
    
    Args:
        tasks: List of task dictionaries
        dataset_name: Name of the dataset (e.g., 'arc_agi1', 'arc_agi2')
        config: Configuration object
        
    Returns:
        Number of successfully processed tasks
    """
    print(f"Preprocessing {dataset_name}...")
    
    # Create dataset-specific directory
    dataset_dir = Path(config.processed_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Store original task count
    total_raw_tasks = len(tasks)
    
    # Filter tasks with at least 2 examples
    filtered_tasks = filter_tasks_with_examples(tasks, min_examples=2)
    
    if not filtered_tasks:
        print(f"No tasks found with at least 2 examples in {dataset_name}")
        return 0
    
    print(f"Filtered to {len(filtered_tasks)} tasks with >= 2 examples")
    
    # Preprocess all tasks
    preprocessed_data = []
    
    for i, task in enumerate(filtered_tasks):
        try:
            preprocessed_task = preprocess_task(task, config)
            preprocessed_data.append(preprocessed_task)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(filtered_tasks)} tasks")
                
        except Exception as e:
            print(f"Error processing task {i}: {e}")
            continue
    
    # Save preprocessed data
    output_file = dataset_dir / "preprocessed_data.pt"
    torch.save(preprocessed_data, output_file)
    
    print(f"{dataset_name} preprocessing completed!")
    print(f"Processed {len(preprocessed_data)} tasks")
    print(f"Saved to: {output_file}")
    
    # Generate data report
    generate_data_report(preprocessed_data, dataset_dir, total_raw_tasks, len(filtered_tasks), dataset_name)
    
    return len(preprocessed_data)

def preprocess_arc_data(config: Config):
    """
    Preprocess all ARC data and save to separate dataset directories.
    
    Args:
        config: Configuration object
    """
    print("Starting ARC data preprocessing...")
    
    # Create main processed directory
    processed_dir = Path(config.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    
    # Process ARC-AGI-1
    if Path(config.arc_agi1_dir).exists():
        agi1_tasks = load_arc_tasks(config.arc_agi1_dir)
        count = preprocess_dataset(agi1_tasks, "arc_agi1", config)
        total_processed += count
    else:
        print(f"ARC-AGI-1 directory not found: {config.arc_agi1_dir}")
    
    # Process ARC-AGI-2
    if Path(config.arc_agi2_dir).exists():
        agi2_tasks = load_arc_tasks(config.arc_agi2_dir)
        count = preprocess_dataset(agi2_tasks, "arc_agi2", config)
        total_processed += count
    else:
        print(f"ARC-AGI-2 directory not found: {config.arc_agi2_dir}")
    
    if total_processed == 0:
        raise FileNotFoundError("No ARC tasks found in specified directories")
    
    print(f"\nTotal preprocessing completed!")
    print(f"Processed {total_processed} tasks across all datasets")
    print(f"Datasets available in: {processed_dir}")

def generate_data_report(data: List[Dict[str, torch.Tensor]], output_dir: Path, 
                        total_raw_tasks: int, filtered_tasks: int, dataset_name: str):
    """
    Generate an HTML report about the preprocessed data.
    
    Args:
        data: List of preprocessed tasks
        output_dir: Output directory for the report
        total_raw_tasks: Total number of raw tasks loaded
        filtered_tasks: Number of tasks after filtering
        dataset_name: Name of the dataset
    """
    if not data:
        print("No data to generate report for")
        return
    
    # Generate HTML report
    html_content = generate_html_report(data, total_raw_tasks, filtered_tasks, dataset_name)
    
    # Save HTML report
    html_report_file = output_dir / "data_processing_report.html"
    with open(html_report_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {html_report_file}")
    print(f"Open in browser to view interactive report")

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess ARC-AGI dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--data-dir", type=str, help="Override data directory")
    parser.add_argument("--processed-dir", type=str, help="Override processed directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.processed_dir:
        config.processed_dir = args.processed_dir
    
    # Run preprocessing
    preprocess_arc_data(config)

if __name__ == "__main__":
    main()
