import json
import numpy as np


def load_arc_json(filepath: str) -> dict:
    """load single ARC json file"""
    with open(filepath, "r") as f:
        return json.load(f)


def pad_grid_center(arr: np.ndarray, target_size: int = 30) -> np.ndarray:
    """pad grid to target_size x target_size with zeros, centered"""
    H, W = arr.shape
    TH = TW = target_size
    out = np.zeros((TH, TW), dtype=np.int64)
    y0 = (TH - H) // 2
    x0 = (TW - W) // 2
    out[y0 : y0 + H, x0 : x0 + W] = arr
    return out
