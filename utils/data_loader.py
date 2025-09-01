import json
import os
from typing import List, Tuple, Dict
import torch
import numpy as np


def load_arc_json(filepath: str) -> Dict:
    """load single ARC json file"""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_train_pairs(data: Dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    """extract (input, output) pairs from ARC task"""
    pairs = []
    for item in data["train"]:
        inp = np.array(item["input"], dtype=np.int64)
        out = np.array(item["output"], dtype=np.int64)
        pairs.append((inp, out))
    return pairs


def pad_grid(grid: np.ndarray, target_size: int = 30) -> np.ndarray:
    """pad grid to target_size x target_size with zeros"""
    h, w = grid.shape
    padded = np.zeros((target_size, target_size), dtype=np.int64)
    padded[:h, :w] = grid
    return padded


def process_arc_file(
    filepath: str, target_size: int = 30
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """process single ARC file to tensor pairs"""
    data = load_arc_json(filepath)
    pairs = extract_train_pairs(data)

    tensor_pairs = []
    for inp, out in pairs:
        inp_padded = pad_grid(inp, target_size)
        out_padded = pad_grid(out, target_size)

        # flatten to (L,) where L = target_size * target_size
        inp_flat = torch.from_numpy(inp_padded.flatten())
        out_flat = torch.from_numpy(out_padded.flatten())

        tensor_pairs.append((inp_flat, out_flat))

    return tensor_pairs


def load_all_training_data(
    data_dir: str, target_size: int = 30
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """load all training data from ARC-AGI-2/data/training/"""
    all_pairs = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir, filename)
            pairs = process_arc_file(filepath, target_size)
            all_pairs.extend(pairs)

    return all_pairs


def split_train_val(
    pairs: List[Tuple[torch.Tensor, torch.Tensor]], val_ratio: float = 0.2
) -> Tuple[List, List]:
    """split pairs into train/val sets"""
    n_val = int(len(pairs) * val_ratio)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    return train_pairs, val_pairs


def save_tensor_pairs(
    pairs: List[Tuple[torch.Tensor, torch.Tensor]], save_dir: str, prefix: str
):
    """save tensor pairs as .pt files"""
    os.makedirs(save_dir, exist_ok=True)

    for i, (inp, out) in enumerate(pairs):
        torch.save(
            {"input": inp, "output": out},
            os.path.join(save_dir, f"{prefix}_{i:06d}.pt"),
        )


def load_tensor_pairs(load_dir: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """load tensor pairs from .pt files"""
    pairs = []
    for filename in sorted(os.listdir(load_dir)):
        if filename.endswith(".pt"):
            filepath = os.path.join(load_dir, filename)
            data = torch.load(filepath)
            pairs.append((data["input"], data["output"]))
    return pairs
