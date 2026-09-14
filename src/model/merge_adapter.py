"""Merge a trained LoRA adapter into its base model.

Standalone post-training CLI step, intended to run on a single GPU (or CPU)
AFTER distributed training has finished — do not run this inside the
distributed training loop.

Usage:
    python src/model/merge_adapter.py \
        --base_model_name_or_path meta-llama/Meta-Llama-3-8B \
        --adapter_path outputs/checkpoints/lora_adapter \
        --output_dir outputs/merged_model
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a trained LoRA adapter into its base model and save the merged model."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace Hub ID or local path to the base model the adapter was trained on top of.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged full model and tokenizer to.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Dtype to load the base model in before merging.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust custom modeling code from the base model's Hub repo.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype]

    print(f"Loading base model from {args.base_model_name_or_path} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading LoRA adapter from {args.adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging adapter weights into base model ...")
    model = model.merge_and_unload()

    print(f"Saving merged model and tokenizer to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()