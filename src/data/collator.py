"""
Data collator for causal language model fine-tuning.

Pads a batch of variable-length tokenized examples to the length of the
longest sequence in that batch (dynamic padding), and builds the `labels`
tensor by copying `input_ids` and masking out padding positions with -100
so they're ignored by the loss function.

Compatible with `transformers.Trainer` (pass an instance as `data_collator=`).

Expected input: a list of dicts, each with at least:
    - "input_ids":      List[int] or 1D tensor
    - "attention_mask": List[int] or 1D tensor (optional, will be derived if missing)
    - "labels":         List[int] or 1D tensor (optional; if absent, derived from input_ids)

If an example already carries its own "labels" (e.g. because prompt tokens
were pre-masked with -100 during dataset prep), those are padded with -100
as well, rather than being overwritten -- this preserves prompt-masking
done upstream.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class DataCollatorForCausalLM:
    """
    Args:
        tokenizer: A HuggingFace tokenizer (used for pad_token_id and padding_side).
        label_pad_token_id: Value used to mask padding positions in `labels`.
            Defaults to -100, the value ignored by `torch.nn.CrossEntropyLoss`.
        pad_to_multiple_of: If set, pads sequence length up to the nearest
            multiple of this value (useful for tensor-core alignment on GPU).
    """

    tokenizer: Any
    label_pad_token_id: int = -100
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id is None. Set tokenizer.pad_token "
                "(e.g. tokenizer.pad_token = tokenizer.eos_token) before "
                "constructing the collator."
            )
        self.pad_token_id = self.tokenizer.pad_token_id
        # Default to right padding for causal LM training unless the
        # tokenizer explicitly says otherwise.
        self.padding_side = getattr(self.tokenizer, "padding_side", "right") or "right"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if len(features) == 0:
            raise ValueError("Received an empty batch of features.")

        input_ids_list = [self._to_list(f["input_ids"]) for f in features]

        attention_mask_list = [
            self._to_list(f["attention_mask"])
            if f.get("attention_mask") is not None
            else [1] * len(self._to_list(f["input_ids"]))
            for f in features
        ]

        # Use provided labels if present, otherwise derive from input_ids.
        labels_list = [
            self._to_list(f["labels"]) if f.get("labels") is not None else list(ids)
            for f, ids in zip(features, input_ids_list)
        ]

        max_len = max(len(ids) for ids in input_ids_list)
        if self.pad_to_multiple_of is not None:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                max_len += self.pad_to_multiple_of - remainder

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ids, mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
            pad_len = max_len - len(ids)

            if self.padding_side == "left":
                padded_ids = [self.pad_token_id] * pad_len + ids
                padded_mask = [0] * pad_len + mask
                padded_labels = [self.label_pad_token_id] * pad_len + labels
            else:  # right padding (default)
                padded_ids = ids + [self.pad_token_id] * pad_len
                padded_mask = mask + [0] * pad_len
                padded_labels = labels + [self.label_pad_token_id] * pad_len

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    @staticmethod
    def _to_list(x: Union[List[int], torch.Tensor]) -> List[int]:
        if isinstance(x, torch.Tensor):
            return x.tolist()
        return list(x)