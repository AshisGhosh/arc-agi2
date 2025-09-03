import torch
from torch.utils.data import Dataset
from pathlib import Path


class FewShotDataset(Dataset):
    """dataset that loads few-shot learning samples with support pairs and query pairs"""

    def __init__(self, root: str, split: str, dataset: str = "agi2"):
        self.root = Path(root)
        self.split = split
        self.dataset = dataset
        self.files = sorted((self.root / dataset / split).glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load few-shot learning sample
        data = torch.load(self.files[idx])

        # extract data
        task_id = data["task_id"]
        support_pairs = data["support_pairs"]  # list of 2 support pairs
        query_inp = data["query_inp"]  # (L,) LongTensor
        query_out = data["query_out"]  # (L,) LongTensor

        # sanity checks
        assert len(support_pairs) == 2, (
            f"expected 2 support pairs, got {len(support_pairs)}"
        )
        assert query_inp.dtype == torch.long, (
            f"query input dtype {query_inp.dtype}, expected torch.long"
        )
        assert query_out.dtype == torch.long, (
            f"query output dtype {query_out.dtype}, expected torch.long"
        )
        assert query_inp.min() >= 0 and query_inp.max() <= 9, (
            f"query input values {query_inp.min()}-{query_inp.max()}, expected [0..9]"
        )
        assert query_out.min() >= 0 and query_out.max() <= 9, (
            f"query output values {query_out.min()}-{query_out.max()}, expected [0..9]"
        )

        # extract support pairs
        support_inp = []
        support_out = []
        for pair in support_pairs:
            inp = pair["inp"]
            out = pair["out"]

            # sanity checks for support pairs
            assert inp.dtype == torch.long, (
                f"support input dtype {inp.dtype}, expected torch.long"
            )
            assert out.dtype == torch.long, (
                f"support output dtype {out.dtype}, expected torch.long"
            )
            assert inp.min() >= 0 and inp.max() <= 9, (
                f"support input values {inp.min()}-{inp.max()}, expected [0..9]"
            )
            assert out.min() >= 0 and out.max() <= 9, (
                f"support output values {out.min()}-{out.max()}, expected [0..9]"
            )

            support_inp.append(inp)
            support_out.append(out)

        # stack support pairs
        support_inp = torch.stack(support_inp)  # (2, L)
        support_out = torch.stack(support_out)  # (2, L)

        # get source information for evaluation
        support_sources = data.get(
            "support_sources", ["train", "train"]
        )  # default to train
        query_source = data.get("query_source", "train")  # default to train

        return (
            support_inp,
            support_out,
            query_inp,
            query_out,
            task_id,
            support_sources,
            query_source,
        )

    def get_eval_samples(self):
        """get samples suitable for evaluation (only train examples as support)"""
        eval_samples = []

        for idx in range(len(self)):
            data = torch.load(self.files[idx])
            support_sources = data.get("support_sources", ["train", "train"])

            # only include samples where both support pairs are from train
            if all(source == "train" for source in support_sources):
                eval_samples.append(idx)

        return eval_samples
