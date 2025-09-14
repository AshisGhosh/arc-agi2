import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Any
from ..config import Config


class ARCDataset(Dataset):
    """
    Dataset for ARC tasks with preprocessed data.

    Loads preprocessed tensors from disk for efficient training.
    """

    def __init__(self, processed_dir: str, config: Config):
        """
        Initialize dataset.

        Args:
            processed_dir: Directory containing preprocessed data
            config: Configuration object
        """
        self.processed_dir = Path(processed_dir)
        self.config = config

        # Determine which dataset to load
        self.dataset_name = config.training_dataset
        self.dataset_path = self.processed_dir / self.dataset_name

        # Load preprocessed data
        self.data = self._load_preprocessed_data()

    def _load_preprocessed_data(self) -> List[Dict[str, Any]]:
        """Load preprocessed data from disk."""
        data_file = self.dataset_path / "preprocessed_data.pt"

        if not data_file.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {data_file}")

        data = torch.load(data_file)
        return data

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - example1_input: [3, 64, 64] RGB
                - example1_output: [3, 64, 64] RGB
                - example2_input: [3, 64, 64] RGB
                - example2_output: [3, 64, 64] RGB
                - target_input: [1, 30, 30] grayscale
                - target_output: [1, 30, 30] grayscale
        """
        sample = self.data[idx]

        return {
            "example1_input": sample["example1_input"],
            "example1_output": sample["example1_output"],
            "example2_input": sample["example2_input"],
            "example2_output": sample["example2_output"],
            "target_input": sample["target_input"],
            "target_output": sample["target_output"],
        }
