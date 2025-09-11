import torch
from torch.utils.data import Dataset
from pathlib import Path


class HRMDataset(Dataset):
    """dataset for HRM training - converts few-shot samples to standard input-label pairs"""

    def __init__(self, root: str, split: str, dataset: str = "agi2"):
        self.root = Path(root)
        self.split = split
        self.dataset = dataset
        self.files = sorted((self.root / dataset / split).glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load sample
        data = torch.load(self.files[idx])

        # extract data - use query as the main input-label pair
        task_id = data["task_id"]
        inputs = data["query_inp"]  # (L,) LongTensor
        labels = data["query_out"]  # (L,) LongTensor
        
        # also extract support pairs for few-shot learning
        support_pairs = data.get("support_pairs", [])

        # sanity checks
        assert (
            inputs.dtype == torch.long
        ), f"input dtype {inputs.dtype}, expected torch.long"
        assert (
            labels.dtype == torch.long
        ), f"label dtype {labels.dtype}, expected torch.long"
        assert (
            inputs.min() >= 0 and inputs.max() <= 9
        ), f"input values {inputs.min()}-{inputs.max()}, expected [0..9]"
        assert (
            labels.min() >= 0 and labels.max() <= 9
        ), f"label values {labels.min()}-{labels.max()}, expected [0..9]"

        # return standard format: inputs, labels, puzzle_identifiers, support_pairs, file_name
        file_name = self.files[idx].name
        return inputs, labels, task_id, support_pairs, file_name

    def get_eval_samples(self):
        """get samples suitable for evaluation (only train examples as query)"""
        eval_samples = []

        for idx in range(len(self)):
            data = torch.load(self.files[idx])
            query_source = data.get("query_source", "train")

            # only include samples where query is from train
            if query_source == "train":
                eval_samples.append(idx)

        return eval_samples
