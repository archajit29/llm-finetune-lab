"""
data/prepare_dataset.py

Tokenizes a JSONL dataset and packs sequences to a fixed max_seq_length using
concatenation (no padding), then saves the result to disk in Arrow format for
fast reloading during training.

Usage:
    python prepare_dataset.py --config configs/data.yaml

Expected YAML config keys:
    dataset:
        raw_path: data/raw/train.jsonl      # path to input jsonl file
        text_field: text                    # key in each json line holding the text
    tokenizer:
        name_or_path: meta-llama/Meta-Llama-3-8B
        add_eos_token: true                 # append eos between docs before packing
    packing:
        max_seq_length: 4096
        num_proc: 8                         # parallel workers for map()
        shuffle: true
        seed: 42
    output:
        processed_dir: data/processed/train_packed
"""

import argparse
import sys
from pathlib import Path

import yaml
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def load_config(config_path: str) -> dict:
    """Load and lightly validate the YAML config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    required_top_level = ["dataset", "tokenizer", "packing", "output"]
    for key in required_top_level:
        if key not in config:
            raise KeyError(f"Missing required config section: '{key}'")

    return config


def load_raw_dataset(dataset_cfg: dict) -> Dataset:
    """Load a jsonl dataset from disk."""
    raw_path = dataset_cfg["raw_path"]
    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Dataset path not found: {raw_path}")

    print(f"[1/4] Loading raw jsonl dataset from: {raw_path}")
    ds = load_dataset("json", data_files=raw_path, split="train")
    print(f"      Loaded {len(ds):,} raw examples")
    return ds


def tokenize_dataset(ds: Dataset, tokenizer, dataset_cfg: dict, packing_cfg: dict) -> Dataset:
    """Tokenize the text field of each example. No padding/truncation here —
    packing is handled separately as a second pass over concatenated ids."""
    text_field = dataset_cfg["text_field"]
    add_eos = tokenizer_cfg_get(dataset_cfg, "add_eos_token", True)
    num_proc = packing_cfg.get("num_proc", 1)

    if text_field not in ds.column_names:
        raise KeyError(
            f"text_field '{text_field}' not found in dataset columns: {ds.column_names}"
        )

    eos_id = tokenizer.eos_token_id

    def _tokenize(batch):
        tokenized = tokenizer(
            batch[text_field],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        input_ids = tokenized["input_ids"]
        if add_eos and eos_id is not None:
            input_ids = [ids + [eos_id] for ids in input_ids]
        return {"input_ids": input_ids}

    print("[2/4] Tokenizing text field with tokenizer:", tokenizer.name_or_path)
    tokenized_ds = ds.map(
        _tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )
    return tokenized_ds


def tokenizer_cfg_get(dataset_cfg: dict, key: str, default):
    """Helper to read a nested tokenizer option that may live under
    dataset_cfg or tokenizer_cfg depending on how the user structured the yaml."""
    return dataset_cfg.get(key, default)


def pack_dataset(tokenized_ds: Dataset, packing_cfg: dict) -> Dataset:
    """Concatenate all tokenized sequences into one long stream of token ids,
    then chunk that stream into fixed-size blocks of max_seq_length.
    This avoids padding entirely (standard "packing" approach used for
    efficient pretraining/finetuning)."""
    max_seq_length = packing_cfg["max_seq_length"]
    num_proc = packing_cfg.get("num_proc", 1)

    print(f"[3/4] Packing sequences into blocks of max_seq_length={max_seq_length}")

    def _group_texts(examples):
        # Flatten all input_ids in this batch into one long list
        concatenated = sum(examples["input_ids"], [])
        total_length = len(concatenated)
        # Drop the small remainder that doesn't fill a full block
        total_length = (total_length // max_seq_length) * max_seq_length

        result_input_ids = [
            concatenated[i : i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
        return {
            "input_ids": result_input_ids,
            "labels": [ids.copy() for ids in result_input_ids],
            "attention_mask": [[1] * max_seq_length for _ in result_input_ids],
        }

    packed_ds = tokenized_ds.map(
        _group_texts,
        batched=True,
        num_proc=num_proc,
        remove_columns=tokenized_ds.column_names,
        desc="Packing",
    )

    if packing_cfg.get("shuffle", True):
        seed = packing_cfg.get("seed", 42)
        print(f"      Shuffling packed dataset with seed={seed}")
        packed_ds = packed_ds.shuffle(seed=seed)

    return packed_ds


def save_dataset(packed_ds: Dataset, output_cfg: dict) -> str:
    """Save the packed dataset to disk in Arrow format."""
    processed_dir = output_cfg["processed_dir"]
    Path(processed_dir).parent.mkdir(parents=True, exist_ok=True)

    print(f"[4/4] Saving packed dataset to: {processed_dir}")
    packed_ds.save_to_disk(processed_dir)
    return processed_dir


def print_stats(raw_ds: Dataset, tokenized_ds: Dataset, packed_ds: Dataset, max_seq_length: int) -> None:
    """Print summary stats about the resulting dataset."""
    total_raw_tokens = sum(len(ids) for ids in tokenized_ds["input_ids"])
    avg_tokens_per_raw_example = total_raw_tokens / len(tokenized_ds) if len(tokenized_ds) else 0

    num_packed_examples = len(packed_ds)
    total_packed_tokens = num_packed_examples * max_seq_length

    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Raw examples loaded:            {len(raw_ds):,}")
    print(f"Tokenized examples:              {len(tokenized_ds):,}")
    print(f"Total tokens before packing:     {total_raw_tokens:,}")
    print(f"Avg tokens / raw example:        {avg_tokens_per_raw_example:,.1f}")
    print("-" * 60)
    print(f"Packed sequences (blocks):       {num_packed_examples:,}")
    print(f"Sequence length per block:       {max_seq_length}")
    print(f"Total tokens after packing:      {total_packed_tokens:,}")
    if total_raw_tokens:
        utilization = min(total_packed_tokens / total_raw_tokens, 1.0) * 100
        print(f"Token utilization (no padding):  {utilization:.1f}%")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize and pack a jsonl dataset into Arrow format for training."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/data.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    tokenizer_cfg = config["tokenizer"]
    packing_cfg = config["packing"]
    output_cfg = config["output"]

    print(f"Loading tokenizer: {tokenizer_cfg['name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg["name_or_path"])
    if tokenizer.eos_token_id is None:
        print("WARNING: tokenizer has no eos_token_id; documents will be "
              "concatenated without an explicit separator.", file=sys.stderr)

    # merge add_eos_token setting into dataset_cfg so tokenize_dataset can read it
    dataset_cfg["add_eos_token"] = tokenizer_cfg.get("add_eos_token", True)

    raw_ds = load_raw_dataset(dataset_cfg)
    tokenized_ds = tokenize_dataset(raw_ds, tokenizer, dataset_cfg, packing_cfg)
    packed_ds = pack_dataset(tokenized_ds, packing_cfg)
    save_dataset(packed_ds, output_cfg)
    print_stats(raw_ds, tokenized_ds, packed_ds, packing_cfg["max_seq_length"])


if __name__ == "__main__":
    main()