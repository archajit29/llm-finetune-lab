"""Dataset loading.

Defines a PyTorch Dataset that loads a pre-tokenized/packed Arrow dataset
(the output of data/prepare_dataset.py) and returns model-ready tensors.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_from_disk


class PackedArrowDataset(Dataset):
    """PyTorch Dataset wrapping a pre-tokenized/packed HuggingFace `datasets`
    Arrow dataset on disk.

    Expects each example in the underlying dataset to already contain
    "input_ids", "attention_mask", and "labels" columns (as produced by
    data/prepare_dataset.py), so no further tokenization or packing happens
    here.
    """

    def __init__(self, dataset_path):
        """
        Args:
            dataset_path: Path to a dataset saved with `Dataset.save_to_disk`
                (e.g. the output directory of data/prepare_dataset.py).
        """
        self.dataset = load_from_disk(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return {
            "input_ids": torch.as_tensor(example["input_ids"], dtype=torch.long),
            "attention_mask": torch.as_tensor(example["attention_mask"], dtype=torch.long),
            "labels": torch.as_tensor(example["labels"], dtype=torch.long),
        }